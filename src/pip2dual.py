# & 'C:\Users\ASUS\yolovs\Scripts\Activate.ps1'
import os
import time
import pickle
import threading
import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import gc

# 简化配置参数
DIST_TH = 0.60
DIST_TH_SAFE = 0.45
BANK_FILE = "idbank.pkl"
TARGET_FPS = 60
MAX_FEATS = 8
CLEAR_BANK_ON_EXIT = True
TRACKER_WINDOW_NAME = "Tracker View"
TRACKER_WINDOW_SIZE = (800, 450)
HEADROOM_RATIO = 0.20
TRACKER_GRACE_PERIOD = 3.0  # 跟踪器宽限期，目标丢失后的等待时间

# 双摄像头相关变量
cam_lock = threading.RLock()  # 改为RLock避免重入锁问题
queue_lock = threading.Lock()  # 保护退出队列操作
gpu_lock = threading.RLock()    # 保护GPU推理操作
gui_lock = threading.RLock()    # 保护GUI操作
control_lock = threading.RLock()  # 专门的控制权锁
current_cam = -1  # 初始状态：无控制权
cam_detections = {0: [], 2: []}
cam_last_seen = {0: 0.0, 2: 0.0}
cross_cam_features = {}
cam_switch_cooldown = 2.0
control_timeout = 3.0  # 控制权超时时间

# 镜像和位置跟踪相关变量
ENABLE_CAMERA_MIRROR = True
person_positions = {}
cam0_right_exit_queue = []  # 从cam0右侧离开的人员队列 [(sid, exit_time), ...]
cam2_left_exit_queue = []   # 从cam2左侧离开的人员队列 [(sid, exit_time), ...]
position_tracking_window = 5
exit_detection_time_window = 3.0
right_exit_threshold = 0.8  # 右侧离开阈值（相对于画面宽度）
left_exit_threshold = 0.2   # 左侧离开阈值（相对于画面宽度）

# 全局变量
target_id = None
current_detections = []
target_last_seen_time = 0
last_good_box = {}
id_last_seen = {}
stabilized_boxes = {}
stability_buffers = {}
lost_tracklets = {}
max_lost_time = 8.0
next_sid = 1
id_bank = {}
track_id_to_sid = {}

def create_tracker():
    """创建DeepSORT追踪器"""
    return DeepSort(
        max_age=120,
        n_init=2,
        nms_max_overlap=0.4,
        max_cosine_distance=0.35,
        nn_budget=150,
        override_track_class=None,
        embedder=None,
        half=False,
        bgr=True,
        embedder_gpu=False,
        embedder_model_name=None,
        embedder_wts=None,
        polygon=False,
        today=None
    )

# 初始化模型
print("[INFO] Loading models...")
yolo = YOLO("weights/yolo11s.pt")
try:
    providers = ["CUDAExecutionProvider"]
    sess = ort.InferenceSession("weights/osnet_x1_0_msmt17.onnx", providers=providers)
    print("[INFO] Using GPU.")
except Exception:
    providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession("weights/osnet_x0_25_msmt17.onnx", providers=providers)
    print("[WARN] Using CPU.")

inp_name = sess.get_inputs()[0].name
tracker = create_tracker()

def validate_bbox(bbox, frame_w, frame_h, min_size=15):
    """简化的边界框验证"""
    if bbox is None:
        return False
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    
    return (w >= min_size and h >= min_size and 
            x1 >= 0 and y1 >= 0 and x2 <= frame_w and y2 <= frame_h and
            w <= frame_w * 0.95 and h <= frame_h * 0.95)

def get_stabilized_box(sid, new_box, now):
    """简化的防抖函数"""
    if sid not in stability_buffers:
        stability_buffers[sid] = [new_box]
        stabilized_boxes[sid] = new_box
        return new_box
    
    buffer = stability_buffers[sid]
    buffer.append(new_box)
    
    if len(buffer) > 3:
        buffer.pop(0)
    
    if len(buffer) < 3:
        stabilized_boxes[sid] = new_box
        return new_box
    
    # 简单平均
    avg_x1 = sum(box[0] for box in buffer) / len(buffer)
    avg_y1 = sum(box[1] for box in buffer) / len(buffer)
    avg_x2 = sum(box[2] for box in buffer) / len(buffer)
    avg_y2 = sum(box[3] for box in buffer) / len(buffer)
    
    stabilized_box = (int(avg_x1), int(avg_y1), int(avg_x2), int(avg_y2))
    stabilized_boxes[sid] = stabilized_box
    return stabilized_box

def pan_and_scan_16_9(box, frame_w, frame_h, headroom_ratio=0.15):
    """简化的16:9裁剪"""
    x1, y1, x2, y2 = box
    box_w, box_h = x2 - x1, y2 - y1
    
    if box_w <= 0 or box_h <= 0:
        return None
    
    center_x, center_y = x1 + box_w / 2, y1 + box_h / 2
    aspect_ratio_16_9 = 16 / 9
    
    # 计算16:9尺寸
    if (box_w / box_h) > aspect_ratio_16_9:
        new_w, new_h = box_w, int(box_w / aspect_ratio_16_9)
    else:
        new_h, new_w = box_h, int(box_h * aspect_ratio_16_9)
    
    # 应用头部空间
    vertical_offset = int(new_h * headroom_ratio)
    adjusted_center_y = center_y - (vertical_offset / 2)
    
    # 计算位置
    final_x1 = max(0, int(center_x - new_w / 2))
    final_y1 = max(0, int(adjusted_center_y - new_h / 2))
    final_x2 = min(frame_w, final_x1 + new_w)
    final_y2 = min(frame_h, final_y1 + new_h)
    
    # 调整确保在边界内
    if final_x2 > frame_w:
        final_x1 = frame_w - new_w
        final_x2 = frame_w
    if final_y2 > frame_h:
        final_y1 = frame_h - new_h
        final_y2 = frame_h
    
    final_x1 = max(0, final_x1)
    final_y1 = max(0, final_y1)
        
    return final_x1, final_y1, final_x2, final_y2

def extract_feat(bgr: np.ndarray) -> np.ndarray:
    """提取OSNet特征"""
    rgb = cv2.cvtColor(cv2.resize(bgr, (128, 256)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb -= (0.485, 0.456, 0.406)
    rgb /= (0.229, 0.224, 0.225)
    blob = rgb.transpose(2, 0, 1)[None].astype(np.float32)
    feat = sess.run(None, {inp_name: blob})[0][0]
    feat /= (np.linalg.norm(feat) + 1e-6)
    return feat.astype(np.float32)

def extract_color_feature(bgr: np.ndarray) -> np.ndarray:
    """提取颜色特征"""
    h, w = bgr.shape[:2]
    upper_body = bgr[int(h*0.1):int(h*0.6), int(w*0.2):int(w*0.8)]
    
    if upper_body.size == 0:
        return np.zeros(48, dtype=np.float32)
    
    hsv = cv2.cvtColor(upper_body, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256])
    
    h_hist = h_hist.flatten() / (h_hist.sum() + 1e-6)
    s_hist = s_hist.flatten() / (s_hist.sum() + 1e-6)
    v_hist = v_hist.flatten() / (v_hist.sum() + 1e-6)
    
    color_feat = np.concatenate([h_hist, s_hist[:16], v_hist[:16]])
    return color_feat.astype(np.float32)

def extract_combined_feature(bgr: np.ndarray) -> tuple:
    """提取组合特征"""
    return extract_feat(bgr), extract_color_feature(bgr)

def calculate_distance(feat1, feat2):
    """计算特征距离"""
    if isinstance(feat1, tuple) and isinstance(feat2, tuple):
        osnet1, color1 = feat1
        osnet2, color2 = feat2
        osnet_dist = float(np.linalg.norm(osnet1 - osnet2))
        color_dist = float(np.linalg.norm(color1 - color2))
        return osnet_dist * 0.7 + color_dist * 0.3
    else:
        return float(np.linalg.norm(feat1 - feat2))

def assign_id(feat, sids_to_exclude=set()):
    """分配ID"""
    global next_sid
    
    with cam_lock:  # 保护ID分配操作
        if not id_bank:
            id_bank[next_sid] = [feat]
            new_id = next_sid
            next_sid += 1
            return new_id
        
        best_id = None
        best_distance = float('inf')
        
        for sid, feats in id_bank.items():
            if sid in sids_to_exclude or not feats:
                continue
                
            distances = [calculate_distance(feat, f) for f in feats[-3:]]
            min_dist = min(distances)
            
            if min_dist < best_distance and min_dist < DIST_TH:
                best_distance = min_dist
                best_id = sid
        
        if best_id is not None:
            id_bank[best_id].append(feat)
            if len(id_bank[best_id]) > MAX_FEATS:
                id_bank[best_id].pop(0)
            return best_id
        
        # 创建新ID
        new_id = next_sid
        while new_id in id_bank or new_id in sids_to_exclude:
            new_id += 1
        next_sid = new_id + 1
        id_bank[new_id] = [feat]
        return new_id

def update_cross_cam_features(sid, feat, cam_id, timestamp):
    """更新跨摄像头特征"""
    if sid not in cross_cam_features:
        cross_cam_features[sid] = {'features': [], 'last_cam': cam_id, 'last_time': timestamp}
    
    cross_cam_features[sid]['features'].append((feat, timestamp))
    cross_cam_features[sid]['last_cam'] = cam_id
    cross_cam_features[sid]['last_time'] = timestamp
    
    if len(cross_cam_features[sid]['features']) > MAX_FEATS:
        cross_cam_features[sid]['features'].pop(0)

def cross_cam_match_id(new_feat, cam_id, timestamp, relative_x=None):
    """跨摄像头ID匹配，带位置约束"""
    global target_id, current_cam
    
    best_match_sid = None
    best_distance = float('inf')
    
    with cam_lock:  # 保护字典遍历操作
        # 创建字典的副本以避免遍历时修改
        features_copy = dict(cross_cam_features)
    
    for sid, info in features_copy.items():
        if info['last_cam'] == cam_id:
            continue

        time_since_lost = timestamp - info['last_time']
        if time_since_lost > max_lost_time:
            continue

        # 添加位置约束：只有在正确位置的人才能匹配跨摄像头ID
        if relative_x is not None:
            if cam_id == 0:
                # CAM0：进一步放宽约束，允许更多位置的人匹配从CAM2来的ID
                if relative_x <= 0.3:  # 从0.8改为0.3，只排除最左侧的人
                    print(f"[DEBUG] CAM0 位置约束：ID{sid} 位置{relative_x:.2f} <= 0.3，跳过匹配")
                    continue
            elif cam_id == 2:
                # CAM2：进一步放宽约束，允许更多位置的人匹配从CAM0来的ID
                if relative_x >= 0.7:  # 从0.2改为0.7，只排除最右侧的人
                    print(f"[DEBUG] CAM2 位置约束：ID{sid} 位置{relative_x:.2f} >= 0.7，跳过匹配")
                    continue
                
        if info['features']:
            distances = [calculate_distance(new_feat, stored_feat) 
                        for stored_feat, _ in info['features'][-3:]]
            min_dist = min(distances)
            
            threshold = DIST_TH_SAFE if len(cam_detections[cam_id]) > 1 else DIST_TH
            
            if min_dist < threshold and min_dist < best_distance:
                best_distance = min_dist
                best_match_sid = sid
    
    if best_match_sid is not None:
        if relative_x is not None:
            print(f"[DEBUG] 跨摄像头匹配成功：CAM{cam_id} 位置{relative_x:.2f} 匹配到ID{best_match_sid}")
        return best_match_sid
    
    return None

def process_detections_and_tracks(yolo, tracker, frame, frame_w, frame_h):
    """处理检测和追踪"""
    with gpu_lock:  # 保护GPU推理操作
        res = yolo.predict(frame, conf=0.3, classes=[0], iou=0.6, imgsz=640, verbose=False)[0]
    
    dets = []
    embeddings = []
    
    for box in res.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
        conf = float(box.conf.cpu().numpy()[0])
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_w, x2), min(frame_h, y2)
        
        if validate_bbox((x1, y1, x2, y2), frame_w, frame_h):
            try:
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    with gpu_lock:  # 保护特征提取
                        feat = extract_feat(crop)
                    dets.append([[x1, y1, x2-x1, y2-y1], conf, 0])
                    embeddings.append(feat)
            except:
                continue
        
    if dets:
        deepsort_tracks = tracker.update_tracks(dets, embeds=embeddings, frame=frame)
    else:
        deepsort_tracks = []

    tracks = []
    for t in deepsort_tracks:
        if not t.is_confirmed():
            continue
            
        try:
            x1, y1, x2, y2 = map(int, t.to_tlbr())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_w, x2), min(frame_h, y2)
            
            if x2 > x1 and y2 > y1:
                tracks.append([x1, y1, x2, y2, int(t.track_id)])
        except:
            continue
            
    return tracks

def update_person_position(sid, bbox, cam_id, timestamp, frame_w):
    """更新人员位置"""
    global person_positions
    
    x1, y1, x2, y2 = bbox
    x_center = (x1 + x2) / 2
    
    if sid not in person_positions:
        person_positions[sid] = {'positions': [], 'last_cam': cam_id}
    
    person_positions[sid]['positions'].append((x_center, timestamp))
    person_positions[sid]['last_cam'] = cam_id
    
    if len(person_positions[sid]['positions']) > position_tracking_window:
        person_positions[sid]['positions'].pop(0)

def detect_right_exit_from_cam0(sid, timestamp, frame_w):
    """检测从cam0右侧离开"""
    global cam0_right_exit_queue
    
    if sid not in person_positions:
        return False
    
    positions = person_positions[sid]['positions']
    last_cam = person_positions[sid]['last_cam']
    
    if last_cam != 0 or len(positions) < 2:
        return False
    
    recent_positions = positions[-3:] if len(positions) >= 3 else positions
    
    if len(recent_positions) >= 2:
        first_x, first_time = recent_positions[0]
        last_x, last_time = recent_positions[-1]
        
        relative_first_x = first_x / frame_w
        relative_last_x = last_x / frame_w
        
        is_moving_right = relative_last_x > relative_first_x
        is_near_right_edge = relative_last_x > right_exit_threshold
        
        if is_moving_right and is_near_right_edge:
            with queue_lock:
                cam0_right_exit_queue.append((sid, timestamp))
            print(f"[INFO] ID {sid} 从cam0右侧离开")
            return True
    
    return False

def detect_left_exit_from_cam2(sid, timestamp, frame_w):
    """检测从cam2左侧离开"""
    global cam2_left_exit_queue
    
    if sid not in person_positions:
        return False
    
    positions = person_positions[sid]['positions']
    last_cam = person_positions[sid]['last_cam']
    
    if last_cam != 2 or len(positions) < 2:
        return False
    
    recent_positions = positions[-3:] if len(positions) >= 3 else positions
    
    if len(recent_positions) >= 2:
        first_x, first_time = recent_positions[0]
        last_x, last_time = recent_positions[-1]
        
        relative_first_x = first_x / frame_w
        relative_last_x = last_x / frame_w
        
        is_moving_left = relative_last_x < relative_first_x
        is_near_left_edge = relative_last_x < left_exit_threshold
        
        if is_moving_left and is_near_left_edge:
            with queue_lock:
                cam2_left_exit_queue.append((sid, timestamp))
            print(f"[INFO] ID {sid} 从cam2左侧离开")
            return True
    
    return False

def get_next_cam2_assignment():
    """获取cam2新ID的分配"""
    global cam0_right_exit_queue
    
    with queue_lock:
        current_time = time.time()
        
        cam0_right_exit_queue = [
            (sid, exit_time) for sid, exit_time in cam0_right_exit_queue
            if current_time - exit_time <= exit_detection_time_window
        ]
        
        if cam0_right_exit_queue:
            return cam0_right_exit_queue.pop(0)[0]
        
        return None

def get_next_cam0_assignment():
    """获取cam0新ID的分配"""
    global cam2_left_exit_queue
    
    with queue_lock:
        current_time = time.time()
        
        cam2_left_exit_queue = [
            (sid, exit_time) for sid, exit_time in cam2_left_exit_queue
            if current_time - exit_time <= exit_detection_time_window
        ]
        
        if cam2_left_exit_queue:
            return cam2_left_exit_queue.pop(0)[0]
        
        return None

def cleanup_features(current_time):
    """清理过期特征"""
    global cross_cam_features, stabilized_boxes, stability_buffers
    
    expired_sids = []
    for sid, info in cross_cam_features.items():
        if current_time - info['last_time'] > max_lost_time:
            expired_sids.append(sid)
    
    for sid in expired_sids:
        cross_cam_features.pop(sid, None)
        stabilized_boxes.pop(sid, None)
        stability_buffers.pop(sid, None)
    
    if len(cross_cam_features) > 50:
        sorted_items = sorted(cross_cam_features.items(), key=lambda x: x[1]['last_time'])
        for sid, _ in sorted_items[:len(cross_cam_features) - 50]:
            cross_cam_features.pop(sid, None)

def try_acquire_control(cam_id, now):
    """尝试获取控制权，确保只有一个摄像头拥有控制权"""
    global current_cam, target_last_seen_time, cam_last_seen, control_timeout
    
    with control_lock:
        # 如果已经是当前控制摄像头，直接返回成功
        if current_cam == cam_id:
            return True
        
        # 检查当前控制摄像头是否超时
        current_cam_last_seen = cam_last_seen.get(current_cam, 0)
        time_since_current_cam = now - current_cam_last_seen
        time_since_target_seen = now - target_last_seen_time
        
        # 严格的控制权切换条件
        should_switch = (
            time_since_target_seen > control_timeout or  # 目标超时
            time_since_current_cam > cam_switch_cooldown  # 当前摄像头超时
        )
        
        if should_switch:
            old_cam = current_cam
            current_cam = cam_id
            print(f"[CONTROL] 控制权从CAM{old_cam}切换到CAM{cam_id}")
            return True
        
        return False

def release_control():
    """释放控制权"""
    global current_cam
    with control_lock:
        print(f"[CONTROL] 释放CAM{current_cam}的控制权")
        current_cam = -1  # 无控制权状态

def select_target_id(event, x, y, flags, param):
    """选择目标ID"""
    global target_id, target_last_seen_time, last_good_box, current_cam
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if target_id is not None:
            last_good_box.pop(target_id, None)
            release_control()  # 释放旧target的控制权
        target_id = None
        
        with cam_lock:
            for cam_id, detections in cam_detections.items():
                for det in detections:
                    x1, y1, x2, y2, sid = det
                    if x1 < x < x2 and y1 < y < y2:
                        target_id = sid
                        # 选择新target时尝试获取控制权
                        if try_acquire_control(cam_id, time.time()):
                            target_last_seen_time = time.time()
                            cam_last_seen[cam_id] = time.time()
                            print(f"[INFO] 目标锁定 ID: {sid} 在 CAM {cam_id}")
                        return
                        
    elif event == cv2.EVENT_RBUTTONDOWN:
        print("[INFO] 目标解锁")
        if target_id is not None:
            last_good_box.pop(target_id, None)
            release_control()  # 释放控制权
        target_id = None
        try:
            with gui_lock:  # 保护GUI操作
                cv2.destroyWindow(TRACKER_WINDOW_NAME)
        except:
            pass

def process_camera(cam_id):
    """处理摄像头"""
    global target_id, current_cam, target_last_seen_time, last_good_box
    
    window_name = f"CAM {cam_id} - Tracking"
    with gui_lock:  # 保护GUI操作
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, select_target_id)
    
    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开摄像头 {cam_id}")
        return
    
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    tracker = create_tracker()
    track_id_to_sid = {}
    
    interval = 1.0 / TARGET_FPS
    last_t = 0.0
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    print(f"[INFO] CAM {cam_id} 启动: {frame_w}x{frame_h}")
    
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if ENABLE_CAMERA_MIRROR:
                frame = cv2.flip(frame, 1)
            
            now = time.time()
            if now - last_t < interval:
                time.sleep(max(0, interval-(now-last_t)))
                continue
            last_t = now
            
            frame_count += 1
            
            with cam_lock:
                cam_detections[cam_id].clear()
            
            tracks = process_detections_and_tracks(yolo, tracker, frame, frame_w, frame_h)
            
            cam_active_sids = set()
            target_found_this_frame = False
            
            for track in tracks:
                x1, y1, x2, y2, track_id = map(int, track)
                
                if not validate_bbox((x1, y1, x2, y2), frame_w, frame_h):
                    continue
                
                try:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    with gpu_lock:  # 保护GPU特征提取操作
                        current_feature = extract_combined_feature(crop)
                except:
                    continue
                
                # ID匹配逻辑
                matched_sid = None
                
                # cam0优先获取从cam2左侧离开的人员ID，但只有从右侧进入的人才能获得
                if cam_id == 0 and track_id not in track_id_to_sid:
                    # 检查检测框位置，放宽约束条件
                    x_center = (x1 + x2) / 2
                    relative_x = x_center / frame_w
                    
                    # 放宽约束：右侧区域的人优先获得从cam2来的ID，但不完全限制
                    if relative_x > 0.6:  # 从0.8放宽到0.6
                        pending_cam2_sid = get_next_cam0_assignment()
                        if pending_cam2_sid is not None:
                            matched_sid = pending_cam2_sid
                            track_id_to_sid[track_id] = matched_sid
                            print(f"[INFO] CAM0右侧检测到从CAM2来的ID: {matched_sid}")
                
                # cam2优先获取从cam0右侧离开的人员ID，但只有从左侧进入的人才能获得
                if cam_id == 2 and track_id not in track_id_to_sid:
                    # 检查检测框位置，放宽约束条件
                    x_center = (x1 + x2) / 2
                    relative_x = x_center / frame_w
                    
                    # 放宽约束：左侧区域的人优先获得从cam0来的ID，但不完全限制
                    if relative_x < 0.4:  # 从0.2放宽到0.4
                        pending_cam0_sid = get_next_cam2_assignment()
                        if pending_cam0_sid is not None:
                            matched_sid = pending_cam0_sid
                            track_id_to_sid[track_id] = matched_sid
                            print(f"[INFO] CAM2左侧检测到从CAM0来的ID: {matched_sid}")
                
                if matched_sid is None:
                    # 重新计算位置信息传给cross_cam_match_id
                    x_center = (x1 + x2) / 2
                    relative_x = x_center / frame_w
                    try:
                        matched_sid = cross_cam_match_id(current_feature, cam_id, now, relative_x)
                    except Exception as e:
                        print(f"[ERROR] 跨摄像头匹配失败: {e}")
                        matched_sid = None
                
                if matched_sid is not None:
                    sid = matched_sid
                    track_id_to_sid[track_id] = sid
                    
                    # 标记找到匹配的target_id，控制权检查统一在显示处理时进行
                elif track_id in track_id_to_sid:
                    sid = track_id_to_sid[track_id]
                else:
                    sid = assign_id(current_feature, cam_active_sids)
                    track_id_to_sid[track_id] = sid
                
                cam_active_sids.add(sid)
                
                update_cross_cam_features(sid, current_feature, cam_id, now)
                
                stabilized_box = get_stabilized_box(sid, (x1, y1, x2, y2), now)
                
                update_person_position(sid, stabilized_box, cam_id, now, frame_w)
                
                # 检测离开行为
                if cam_id == 0:
                    detect_right_exit_from_cam0(sid, now, frame_w)  # 检测从cam0右侧离开
                elif cam_id == 2:
                    detect_left_exit_from_cam2(sid, now, frame_w)   # 检测从cam2左侧离开
                
                adj_box = pan_and_scan_16_9(stabilized_box, frame_w, frame_h, HEADROOM_RATIO)
                if adj_box is None:
                    continue
                
                adj_x1, adj_y1, adj_x2, adj_y2 = adj_box
                
                with cam_lock:
                    cam_detections[cam_id].append((adj_x1, adj_y1, adj_x2, adj_y2, sid))
                    
                    # 在同一个锁内处理target相关逻辑，避免重复获取锁
                    if sid == target_id:
                        # 尝试获取控制权
                        control_acquired = try_acquire_control(cam_id, now)
                        
                        # 只有获取控制权后才设置target_found_this_frame
                        if control_acquired:
                            target_found_this_frame = True
                            target_last_seen_time = now
                            cam_last_seen[cam_id] = now
                            last_good_box[target_id] = adj_box
                            
                            # 更新tracker window
                            target_crop = frame[adj_y1:adj_y2, adj_x1:adj_x2]
                            if target_crop.size > 0:
                                # 裁剪中心部分填满窗口(无黑边)
                                crop_h, crop_w = target_crop.shape[:2]
                                target_w, target_h = TRACKER_WINDOW_SIZE
                                
                                # 计算裁剪比例，使图像能完全填满窗口
                                scale = max(target_w / crop_w, target_h / crop_h)
                                new_w = int(crop_w * scale)
                                new_h = int(crop_h * scale)
                                
                                # 先resize到更大尺寸
                                resized_crop = cv2.resize(target_crop, (new_w, new_h))
                                
                                # 裁剪中心部分
                                start_x = max(0, (new_w - target_w) // 2)
                                start_y = max(0, (new_h - target_h) // 2)
                                end_x = start_x + target_w
                                end_y = start_y + target_h
                                
                                display_frame = resized_crop[start_y:end_y, start_x:end_x]
                                
                                try:
                                    with gui_lock:  # 保护GUI操作
                                        cv2.imshow(TRACKER_WINDOW_NAME, display_frame)
                                except Exception as e:
                                    print(f"[ERROR] 显示tracker窗口失败: {e}")
                                    pass
                
                # 绘制边界框 - 严格控制权检查
                is_target = (sid == target_id)
                
                # 使用控制权锁进行严格检查
                with control_lock:
                    has_control = (current_cam == cam_id)
                
                if is_target and has_control:
                    # 只有拥有控制权的摄像头才显示红色target框
                    label_color = (0, 0, 255)  # 红色
                    cv2.rectangle(frame, (adj_x1, adj_y1), (adj_x2, adj_y2), label_color, 3)
                    label_text = f"ID:{sid} [TARGET-CONTROL]"
                elif is_target:
                    # 其他摄像头检测到target但没有控制权，显示蓝色
                    label_color = (255, 0, 0)  # 蓝色
                    cv2.rectangle(frame, (adj_x1, adj_y1), (adj_x2, adj_y2), label_color, 2)
                    label_text = f"ID:{sid} [DETECTED]"
                else:
                    label_color = (0, 255, 0)  # 绿色，普通检测
                    cv2.rectangle(frame, (adj_x1, adj_y1), (adj_x2, adj_y2), label_color, 2)
                    label_text = f"ID:{sid}"
                
                cv2.putText(frame, label_text, (adj_x1, adj_y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)
            
            # 清理过期数据
            if frame_count % 120 == 0:
                cleanup_features(now)
                gc.collect()
            
            # 自动切换逻辑：作为备用方案，确保控制权能够切换
            if target_id is not None and not target_found_this_frame:
                with cam_lock:
                    # 检查当前摄像头是否有target
                    current_cam_has_target = False
                    for det in cam_detections[cam_id]:
                        _, _, _, _, detected_sid = det
                        if detected_sid == target_id:
                            current_cam_has_target = True
                            break
                    
                    # 如果当前摄像头有target，尝试获取控制权
                    if current_cam_has_target:
                        if try_acquire_control(cam_id, now):
                            target_found_this_frame = True
                            target_last_seen_time = now
                            cam_last_seen[cam_id] = now
                            print(f"[CONTROL] CAM{cam_id} 备用切换获取控制权")
            
            # 处理目标丢失的Re-acquiring显示 - 使用严格的控制权检查
            if target_id is not None and not target_found_this_frame:
                with control_lock:
                    has_control = (current_cam == cam_id)
                
                if has_control:
                    time_since_lost = now - target_last_seen_time
                    if time_since_lost > TRACKER_GRACE_PERIOD:
                        try:
                            with gui_lock:  # 保护GUI操作
                                cv2.destroyWindow(TRACKER_WINDOW_NAME)
                        except:
                            pass
                        # 超时后释放控制权
                        release_control()
                    else:
                        # 只有在目标确实丢失超过0.5秒后才显示Re-acquiring
                        if time_since_lost > 0.5:
                            last_box = last_good_box.get(target_id)
                            if last_box:
                                lx1, ly1, lx2, ly2 = last_box
                                if lx2 > lx1 and ly2 > ly1:
                                    live_crop = frame[ly1:ly2, lx1:lx2]
                                    if live_crop.size > 0:
                                        # 裁剪中心部分填满窗口(无黑边)
                                        crop_h, crop_w = live_crop.shape[:2]
                                        target_w, target_h = TRACKER_WINDOW_SIZE
                                        
                                        # 计算裁剪比例，使图像能完全填满窗口
                                        scale = max(target_w / crop_w, target_h / crop_h)
                                        new_w = int(crop_w * scale)
                                        new_h = int(crop_h * scale)
                                        
                                        # 先resize到更大尺寸
                                        resized_crop = cv2.resize(live_crop, (new_w, new_h))
                                        
                                        # 裁剪中心部分
                                        start_x = max(0, (new_w - target_w) // 2)
                                        start_y = max(0, (new_h - target_h) // 2)
                                        end_x = start_x + target_w
                                        end_y = start_y + target_h
                                        
                                        lost_view = resized_crop[start_y:end_y, start_x:end_x]
                                        
                                        # 显示剩余时间，让用户了解状态
                                        remaining_time = TRACKER_GRACE_PERIOD - time_since_lost
                                        status_text = f"Re-acquiring... ({remaining_time:.1f}s)"
                                        cv2.putText(lost_view, status_text, (20, 40), 
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                                        
                                        try:
                                            with gui_lock:  # 保护GUI操作
                                                cv2.imshow(TRACKER_WINDOW_NAME, lost_view)
                                        except Exception as e:
                                            print(f"[ERROR] 显示Re-acquiring窗口失败: {e}")
                                            pass
                        else:
                            # 短时间丢失时，继续显示最后的目标窗口但不显示Re-acquiring
                            last_box = last_good_box.get(target_id)
                            if last_box:
                                lx1, ly1, lx2, ly2 = last_box
                                if lx2 > lx1 and ly2 > ly1:
                                    live_crop = frame[ly1:ly2, lx1:lx2]
                                    if live_crop.size > 0:
                                        # 裁剪中心部分填满窗口(无黑边)
                                        crop_h, crop_w = live_crop.shape[:2]
                                        target_w, target_h = TRACKER_WINDOW_SIZE
                                        
                                        # 计算裁剪比例，使图像能完全填满窗口
                                        scale = max(target_w / crop_w, target_h / crop_h)
                                        new_w = int(crop_w * scale)
                                        new_h = int(crop_h * scale)
                                        
                                        # 先resize到更大尺寸
                                        resized_crop = cv2.resize(live_crop, (new_w, new_h))
                                        
                                        # 裁剪中心部分
                                        start_x = max(0, (new_w - target_w) // 2)
                                        start_y = max(0, (new_h - target_h) // 2)
                                        end_x = start_x + target_w
                                        end_y = start_y + target_h
                                        
                                        lost_view = resized_crop[start_y:end_y, start_x:end_x]
                                        
                                        try:
                                            with gui_lock:  # 保护GUI操作
                                                cv2.imshow(TRACKER_WINDOW_NAME, lost_view)
                                        except Exception as e:
                                            print(f"[ERROR] 显示短时间丢失窗口失败: {e}")
                                            pass
        
            # 显示信息
            mirror_info = " (镜像)" if ENABLE_CAMERA_MIRROR else ""
            
            # 使用控制权锁获取当前控制权状态
            with control_lock:
                control_status = f"Control: CAM{current_cam}" if current_cam >= 0 else "Control: NONE"
            
            cam_info = f"CAM {cam_id}{mirror_info} | Active: {len(cam_active_sids)} | {control_status}"
            cv2.putText(frame, cam_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示离开队列状态
            if cam_id == 0:
                if len(cam0_right_exit_queue) > 0:
                    exit_info = f"Right Exit Queue: {len(cam0_right_exit_queue)}"
                    cv2.putText(frame, exit_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            elif cam_id == 2:
                if len(cam2_left_exit_queue) > 0:
                    exit_info = f"Left Exit Queue: {len(cam2_left_exit_queue)}"
                    cv2.putText(frame, exit_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            with gui_lock:  # 保护GUI操作
                cv2.imshow(window_name, frame)
            
            with gui_lock:  # 保护GUI操作
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                
            with gui_lock:  # 保护GUI操作
                key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
                
    except Exception as e:
        print(f"[ERROR] CAM {cam_id} 错误: {e}")
    finally:
        try:
            cap.release()
            with gui_lock:  # 保护GUI操作
                cv2.destroyWindow(window_name)
        except:
            pass
        print(f"[INFO] CAM {cam_id} 结束")

# 初始化ID库
if os.path.exists(BANK_FILE) and not CLEAR_BANK_ON_EXIT:
    try:
        with open(BANK_FILE, "rb") as f:
            id_bank = pickle.load(f)
            next_sid = max(id_bank.keys()) + 1 if id_bank else 1
    except:
        id_bank, next_sid = {}, 1
else:
    id_bank, next_sid = {}, 1

# 主程序
print("[INFO] 启动双摄像头跟踪系统...")
print("[INFO] 摄像头配置: CAM0(左侧), CAM2(右侧)")
if ENABLE_CAMERA_MIRROR:
    print("[INFO] 镜像功能已启用")
print("[INFO] 位置跟踪功能已启用:")
print("[INFO] - 从cam0右侧离开的人员将自动分配给cam2新ID")
print("[INFO] - 从cam2左侧离开的人员将自动分配给cam0新ID")
print("[INFO] 位置约束已启用:")
print(f"[INFO] - CAM0只有右侧位置(>{right_exit_threshold})的人才能获得[DETECTED]状态")
print(f"[INFO] - CAM2只有左侧位置(<{left_exit_threshold})的人才能获得[DETECTED]状态")
print(f"[INFO] 跟踪器宽限期: {TRACKER_GRACE_PERIOD}秒")
print("[INFO] 目标丢失时将显示Re-acquiring状态")
print("[INFO] 左键选择目标，右键取消，Q键退出")

# 启动摄像头线程
camera_threads = []
for cam_id in [0, 2]:
    thread = threading.Thread(target=process_camera, args=(cam_id,))
    thread.daemon = True
    thread.start()
    camera_threads.append(thread)

# 等待线程完成
try:
    for thread in camera_threads:
        thread.join()
except KeyboardInterrupt:
    print("\n[INFO] 收到中断信号，正在关闭...")

# 清理
with gui_lock:  # 保护GUI操作
    cv2.destroyAllWindows()
    if CLEAR_BANK_ON_EXIT:
        if os.path.exists(BANK_FILE):
            os.remove(BANK_FILE)
    else:
        with open(BANK_FILE, "wb") as f:
            pickle.dump(id_bank, f)

print("[INFO] 程序结束")