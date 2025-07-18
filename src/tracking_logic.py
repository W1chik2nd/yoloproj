# tracking_logic.py (v6 - 严格遵循用户提供的 pip2dual.py 逻辑)
# 本文件是用户提供的 pip2dual.py 的直接模块化转换，未修改任何核心算法。

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

# --- 配置参数 (来自用户文件) ---
DIST_TH = 0.60
DIST_TH_SAFE = 0.45
BANK_FILE = "idbank.pkl"
TARGET_FPS = 60
MAX_FEATS = 8
CLEAR_BANK_ON_EXIT = True
TRACKER_WINDOW_NAME = "Tracker View"
TRACKER_WINDOW_SIZE = (800, 450)
HEADROOM_RATIO = 0.20
TRACKER_GRACE_PERIOD = 3.0

# --- 全局状态变量和锁 (来自用户文件) ---
global_lock = threading.RLock()
gpu_lock = threading.RLock()

# 模块内部全局变量
yolo, sess, inp_name = None, None, None
id_bank, next_sid = {}, 1
target_id, current_cam = None, -1
cam_detections = {0: [], 2: []}
cam_last_seen = {0: 0.0, 2: 0.0}
cross_cam_features = {}
cam_switch_cooldown, control_timeout = 1.0, 2.0
ENABLE_CAMERA_MIRROR = True
person_positions = {}
cam0_right_exit_queue, cam2_left_exit_queue = [], []
position_tracking_window, exit_detection_time_window = 5, 3.0
right_exit_threshold, left_exit_threshold = 0.8, 0.2
target_last_seen_time = 0
last_good_box = {}
stabilized_boxes, stability_buffers = {}, {}
max_lost_time = 8.0

# --- 核心功能函数 (严格来自用户文件) ---
def create_tracker():
    return DeepSort(max_age=60, n_init=3, nms_max_overlap=0.4, max_cosine_distance=0.35, nn_budget=150)

def validate_bbox(bbox, frame_w, frame_h, min_size=15):
    if bbox is None: return False
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    return (w >= min_size and h >= min_size and x1 >= 0 and y1 >= 0 and x2 <= frame_w and y2 <= frame_h and w <= frame_w * 0.95 and h <= frame_h * 0.95)

def get_stabilized_box(sid, new_box, now):
    if sid not in stability_buffers: stability_buffers[sid] = []
    buffer = stability_buffers[sid]
    buffer.append(new_box)
    if len(buffer) > 3: buffer.pop(0)
    avg_box = np.mean(buffer, axis=0).astype(int)
    stabilized_boxes[sid] = tuple(avg_box)
    return stabilized_boxes[sid]

def pan_and_scan_16_9(box, frame_w, frame_h, headroom_ratio=0.15):
    x1, y1, x2, y2 = box
    box_w, box_h = x2 - x1, y2 - y1
    if box_w <= 0 or box_h <= 0: return None
    center_x, center_y = x1 + box_w / 2, y1 + box_h / 2
    aspect_ratio_16_9 = 16 / 9
    if (box_w / box_h) > aspect_ratio_16_9:
        new_w, new_h = box_w, int(box_w / aspect_ratio_16_9)
    else:
        new_h, new_w = box_h, int(box_h * aspect_ratio_16_9)
    adjusted_center_y = center_y - (int(new_h * headroom_ratio) / 2)
    final_x1 = max(0, int(center_x - new_w / 2))
    final_y1 = max(0, int(adjusted_center_y - new_h / 2))
    final_x2 = min(frame_w, final_x1 + new_w)
    final_y2 = min(frame_h, final_y1 + new_h)
    if final_x2 > frame_w: final_x1 = frame_w - new_w; final_x2 = frame_w
    if final_y2 > frame_h: final_y1 = frame_h - new_h; final_y2 = frame_h
    final_x1 = max(0, final_x1)
    final_y1 = max(0, final_y1)
    return final_x1, final_y1, final_x2, final_y2

def extract_feat(bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(cv2.resize(bgr, (128, 256)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb -= (0.485, 0.456, 0.406)
    rgb /= (0.229, 0.224, 0.225)
    blob = rgb.transpose(2, 0, 1)[None].astype(np.float32)
    feat = sess.run(None, {inp_name: blob})[0][0]
    feat /= (np.linalg.norm(feat) + 1e-6)
    return feat.astype(np.float32)

def extract_color_feature(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    upper_body = bgr[int(h*0.1):int(h*0.6), int(w*0.2):int(w*0.8)]
    if upper_body.size == 0: return np.zeros(48, dtype=np.float32)
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
    return extract_feat(bgr), extract_color_feature(bgr)

def calculate_distance(feat1, feat2):
    if isinstance(feat1, tuple) and isinstance(feat2, tuple):
        osnet_dist = float(np.linalg.norm(feat1[0] - feat2[0]))
        color_dist = float(np.linalg.norm(feat1[1] - feat2[1]))
        return osnet_dist * 0.7 + color_dist * 0.3
    return float(np.linalg.norm(feat1 - feat2))

def assign_id(feat, sids_to_exclude=set()):
    global next_sid, id_bank
    if not id_bank:
        id_bank[next_sid] = [feat]; new_id = next_sid; next_sid += 1; return new_id
    best_id, best_distance = None, float('inf')
    for sid, feats in id_bank.items():
        if sid in sids_to_exclude or not feats: continue
        min_dist = min([calculate_distance(feat, f) for f in feats[-3:]])
        if min_dist < best_distance and min_dist < DIST_TH:
            best_distance, best_id = min_dist, sid
    if best_id is not None:
        id_bank[best_id].append(feat)
        if len(id_bank[best_id]) > MAX_FEATS: id_bank[best_id].pop(0)
        return best_id
    new_id = next_sid
    while new_id in id_bank or new_id in sids_to_exclude: new_id += 1
    next_sid = new_id + 1; id_bank[new_id] = [feat]
    return new_id

def update_cross_cam_features(sid, feat, cam_id, timestamp):
    if sid not in cross_cam_features:
        cross_cam_features[sid] = {'features': [], 'last_cam': cam_id, 'last_time': timestamp}
    cross_cam_features[sid]['features'].append((feat, timestamp))
    cross_cam_features[sid]['last_cam'], cross_cam_features[sid]['last_time'] = cam_id, timestamp
    if len(cross_cam_features[sid]['features']) > MAX_FEATS: cross_cam_features[sid]['features'].pop(0)

def cross_cam_match_id(new_feat, cam_id, timestamp, relative_x=None):
    best_match_sid, best_distance = None, float('inf')
    features_copy = dict(cross_cam_features)
    for sid, info in features_copy.items():
        if info['last_cam'] == cam_id or timestamp - info['last_time'] > max_lost_time: continue
        if relative_x is not None:
            if cam_id == 0 and relative_x <= 0.3: continue
            if cam_id == 2 and relative_x >= 0.7: continue
        if info['features']:
            min_dist = min([calculate_distance(new_feat, sf) for sf, _ in info['features'][-3:]])
            threshold = DIST_TH_SAFE if len(cam_detections[cam_id]) > 1 else DIST_TH
            if min_dist < threshold and min_dist < best_distance:
                best_distance, best_match_sid = min_dist, sid
    return best_match_sid

def process_detections_and_tracks(yolo, tracker, frame, frame_w, frame_h):
    with gpu_lock:
        res = yolo.predict(frame, conf=0.3, classes=[0], iou=0.6, imgsz=640, verbose=False)[0]
    dets, embeddings = [], []
    for box in res.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
        conf = float(box.conf.cpu().numpy()[0])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_w, x2), min(frame_h, y2)
        if validate_bbox((x1, y1, x2, y2), frame_w, frame_h):
            try:
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    with gpu_lock: feat = extract_feat(crop)
                    dets.append([[x1, y1, x2-x1, y2-y1], conf, 0])
                    embeddings.append(feat)
            except: continue
    deepsort_tracks = tracker.update_tracks(dets, embeds=embeddings, frame=frame) if dets else []
    tracks = []
    for t in deepsort_tracks:
        if not t.is_confirmed(): continue
        try:
            x1, y1, x2, y2 = map(int, t.to_tlbr())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_w, x2), min(frame_h, y2)
            if x2 > x1 and y2 > y1: tracks.append([x1, y1, x2, y2, int(t.track_id)])
        except: continue
    return tracks

def update_person_position(sid, bbox, cam_id, timestamp, frame_w):
    x1, y1, x2, y2 = bbox
    x_center = (x1 + x2) / 2
    if sid not in person_positions:
        person_positions[sid] = {'positions': [], 'last_cam': cam_id}
    person_positions[sid]['positions'].append((x_center, timestamp))
    person_positions[sid]['last_cam'] = cam_id
    if len(person_positions[sid]['positions']) > position_tracking_window:
        person_positions[sid]['positions'].pop(0)

def detect_right_exit_from_cam0(sid, timestamp, frame_w):
    if sid not in person_positions: return False
    positions, last_cam = person_positions[sid]['positions'], person_positions[sid]['last_cam']
    if last_cam != 0 or len(positions) < 2: return False
    recent_positions = positions[-3:] if len(positions) >= 3 else positions
    if len(recent_positions) >= 2:
        first_x, _ = recent_positions[0]; last_x, _ = recent_positions[-1]
        if last_x / frame_w > right_exit_threshold and last_x > first_x:
            cam0_right_exit_queue.append((sid, timestamp)); print(f"[INFO] ID {sid} 从cam0右侧离开")
            return True
    return False

def detect_left_exit_from_cam2(sid, timestamp, frame_w):
    if sid not in person_positions: return False
    positions, last_cam = person_positions[sid]['positions'], person_positions[sid]['last_cam']
    if last_cam != 2 or len(positions) < 2: return False
    recent_positions = positions[-3:] if len(positions) >= 3 else positions
    if len(recent_positions) >= 2:
        first_x, _ = recent_positions[0]; last_x, _ = recent_positions[-1]
        if last_x / frame_w < left_exit_threshold and last_x < first_x:
            cam2_left_exit_queue.append((sid, timestamp)); print(f"[INFO] ID {sid} 从cam2左侧离开")
            return True
    return False

def get_next_cam2_assignment():
    current_time = time.time()
    cam0_right_exit_queue[:] = [(sid, exit_time) for sid, exit_time in cam0_right_exit_queue if current_time - exit_time <= exit_detection_time_window]
    return cam0_right_exit_queue.pop(0)[0] if cam0_right_exit_queue else None

def get_next_cam0_assignment():
    current_time = time.time()
    cam2_left_exit_queue[:] = [(sid, exit_time) for sid, exit_time in cam2_left_exit_queue if current_time - exit_time <= exit_detection_time_window]
    return cam2_left_exit_queue.pop(0)[0] if cam2_left_exit_queue else None

def cleanup_features(current_time):
    with global_lock:
        expired_sids = [sid for sid, info in list(cross_cam_features.items()) if current_time - info['last_time'] > max_lost_time]
        for sid in expired_sids:
            cross_cam_features.pop(sid, None); stabilized_boxes.pop(sid, None); stability_buffers.pop(sid, None)
        if len(cross_cam_features) > 50:
            sids_to_remove = sorted(cross_cam_features, key=lambda s: cross_cam_features[s]['last_time'])[:len(cross_cam_features) - 50]
            for sid in sids_to_remove: cross_cam_features.pop(sid, None)

def try_acquire_control(cam_id, now):
    global current_cam, target_last_seen_time, cam_last_seen
    with global_lock:
        if current_cam == cam_id: return True
        time_since_current_cam = now - cam_last_seen.get(current_cam, 0)
        time_since_target_seen = now - target_last_seen_time
        if time_since_target_seen > control_timeout or time_since_current_cam > cam_switch_cooldown:
            old_cam = current_cam; current_cam = cam_id
            print(f"[CONTROL] 控制权从CAM{old_cam}切换到CAM{cam_id}")
            return True
        return False

def release_control():
    global current_cam
    with global_lock:
        print(f"[CONTROL] 释放CAM{current_cam}的控制权"); current_cam = -1

# --- GUI 接口函数 ---
def initialize():
    global yolo, sess, inp_name, id_bank, next_sid
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
    with global_lock:
        if os.path.exists(BANK_FILE) and not CLEAR_BANK_ON_EXIT:
            try:
                with open(BANK_FILE, "rb") as f: id_bank = pickle.load(f)
                next_sid = max(id_bank.keys()) + 1 if id_bank else 1
            except: id_bank, next_sid = {}, 1
        else: id_bank, next_sid = {}, 1
    print("[INFO] 追踪模块初始化完成。")

def set_target(new_sid):
    global target_id, target_last_seen_time
    with global_lock:
        target_id = new_sid
        target_last_seen_time = time.time()
        last_cam = cross_cam_features.get(new_sid, {}).get('last_cam')
        if last_cam is not None: try_acquire_control(last_cam, target_last_seen_time)
        print(f"[INFO] GUI 设置追踪目标为 ID: {target_id}")

def clear_target():
    global target_id
    with global_lock:
        if target_id is not None:
            last_good_box.pop(target_id, None); release_control(); target_id = None
            print("[INFO] GUI 取消追踪目标")

def cleanup_and_save():
    if not CLEAR_BANK_ON_EXIT:
        with global_lock:
            with open(BANK_FILE, "wb") as f: pickle.dump(id_bank, f)
            print(f"[INFO] ID库已保存到 {BANK_FILE}")
    elif os.path.exists(BANK_FILE): os.remove(BANK_FILE)
    print("[INFO] 模块清理完成。")

def reset_state():
    global id_bank, next_sid, target_id, current_cam, cam_detections
    global cam_last_seen, cross_cam_features, person_positions
    global cam0_right_exit_queue, cam2_left_exit_queue
    global last_good_box, target_last_seen_time, stabilized_boxes, stability_buffers

    print("[INFO] 重置状态")    
    with global_lock:
        id_bank.clear()
        cross_cam_features.clear()
        person_positions.clear()
        cam_detections = {0: [], 2: []}
        cam_last_seen = {0: 0, 2: 0}
        cam0_right_exit_queue.clear()
        cam2_left_exit_queue.clear()
        last_good_box.clear()
        stabilized_boxes.clear()
        stability_buffers.clear()

        next_sid = 1
        target_id = None
        current_cam = -1
        target_last_seen_time = 0
        print("[INFO] 状态已重置")

# --- 核心处理生成器 ---
def process_camera(cam_id):
    # 此函数内所有逻辑均严格复制自用户提供的 pip2dual.py
    global target_id, current_cam, target_last_seen_time, last_good_box
    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    if not cap.isOpened(): print(f"[ERROR] 无法打开摄像头 {cam_id}"); return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    tracker_instance = create_tracker()
    track_id_to_sid = {}

    interval = 1.0 / TARGET_FPS; last_t = 0.0
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count = 0
    print(f"[INFO] CAM {cam_id} 处理线程启动: {frame_w}x{frame_h}")

    while True:
        ret, frame = cap.read()
        if not ret: break
        if ENABLE_CAMERA_MIRROR: frame = cv2.flip(frame, 1)
        now = time.time()
        if now - last_t < interval:
            time.sleep(max(0, interval - (now - last_t))); continue
        last_t = now
        frame_count += 1
        
        tracker_view_to_show = None
        should_close_tracker_window = False
        boxes_to_draw = {}
        local_cam_detections = []
        cam_active_sids = set()
        target_found_this_frame = False
        
        tracks = process_detections_and_tracks(yolo, tracker_instance, frame, frame_w, frame_h)
        
        if global_lock.acquire(timeout=0.05):
            try:
                for track_item in tracks:
                    x1, y1, x2, y2, track_id = track_item
                    
                    try:
                        crop = frame[y1:y2, x1:x2]
                        if crop.size == 0: continue
                        with gpu_lock: current_feature = extract_combined_feature(crop)
                    except: continue

                    # <<< 关键: 严格还原用户原版的ID匹配逻辑 >>>
                    matched_sid = None
                    if cam_id == 0 and track_id not in track_id_to_sid:
                        if (x1 + x2) / 2 / frame_w > 0.6:
                            pending_sid = get_next_cam0_assignment()
                            if pending_sid is not None:
                                matched_sid = pending_sid
                                print(f"[INFO] CAM0右侧检测到从CAM2来的ID: {matched_sid}")
                    
                    if cam_id == 2 and track_id not in track_id_to_sid:
                        if (x1 + x2) / 2 / frame_w < 0.4:
                            pending_sid = get_next_cam2_assignment()
                            if pending_sid is not None:
                                matched_sid = pending_sid
                                print(f"[INFO] CAM2左侧检测到从CAM0来的ID: {matched_sid}")

                    if matched_sid is None:
                        relative_x = (x1 + x2) / 2 / frame_w
                        try:
                            matched_sid = cross_cam_match_id(current_feature, cam_id, now, relative_x)
                        except Exception as e:
                            print(f"[ERROR] 跨摄像头匹配失败: {e}")

                    if matched_sid is not None:
                        sid = matched_sid
                        track_id_to_sid[track_id] = sid
                    elif track_id in track_id_to_sid:
                        sid = track_id_to_sid[track_id]
                    else:
                        sid = assign_id(current_feature, cam_active_sids)
                        track_id_to_sid[track_id] = sid
                    
                    cam_active_sids.add(sid)
                    update_cross_cam_features(sid, current_feature, cam_id, now)
                    stabilized_box = get_stabilized_box(sid, (x1, y1, x2, y2), now)
                    update_person_position(sid, stabilized_box, cam_id, now, frame_w)

                    if cam_id == 0: detect_right_exit_from_cam0(sid, now, frame_w)
                    elif cam_id == 2: detect_left_exit_from_cam2(sid, now, frame_w)

                    adj_box = pan_and_scan_16_9(stabilized_box, frame_w, frame_h, HEADROOM_RATIO)
                    if adj_box is None: continue
                    adj_x1, adj_y1, adj_x2, adj_y2 = adj_box
                    local_cam_detections.append((adj_x1, adj_y1, adj_x2, adj_y2, sid))
                    
                    if sid == target_id:
                        control_acquired = try_acquire_control(cam_id, now)
                        if control_acquired:
                            target_found_this_frame = True
                            target_last_seen_time = now; cam_last_seen[cam_id] = now
                            last_good_box[target_id] = adj_box
                            
                            target_crop = frame[adj_y1:adj_y2, adj_x1:adj_x2]
                            if target_crop.size > 0:
                                crop_h, crop_w = target_crop.shape[:2]
                                target_w, target_h = TRACKER_WINDOW_SIZE
                                # <<< 关键: 严格还原用户原版的Tracker View缩放逻辑 (max) >>>
                                scale = max(target_w / crop_w, target_h / crop_h)
                                new_w, new_h = int(crop_w * scale), int(crop_h * scale)
                                resized_crop = cv2.resize(target_crop, (new_w, new_h))
                                start_x = max(0, (new_w - target_w) // 2)
                                start_y = max(0, (new_h - target_h) // 2)
                                tracker_view_to_show = resized_crop[start_y:start_y + target_h, start_x:start_x + target_w]
                    
                    is_target, has_control_for_draw = (sid == target_id), (current_cam == cam_id)
                    if is_target and has_control_for_draw: color, label, thick = (0,0,255), f"ID:{sid} [TARGET-CONTROL]", 3
                    elif is_target: color, label, thick = (255,0,0), f"ID:{sid} [DETECTED]", 2
                    else: color, label, thick = (0,255,0), f"ID:{sid}", 2
                    
                    # <<< 关键: 严格还原用户原版绘制16:9 adj_box的逻辑 >>>
                    box_info = {'box': adj_box, 'label': label, 'color': color, 'thickness': thick}
                    if sid not in boxes_to_draw or (adj_box[2]-adj_box[0])*(adj_box[3]-adj_box[1]) > (boxes_to_draw[sid]['box'][2]-boxes_to_draw[sid]['box'][0])*(boxes_to_draw[sid]['box'][3]-boxes_to_draw[sid]['box'][1]):
                         boxes_to_draw[sid] = box_info
                
                cam_detections[cam_id] = local_cam_detections
                if frame_count % 120 == 0: cleanup_features(now)

                if target_id is not None and not target_found_this_frame:
                    if any(det[4] == target_id for det in cam_detections[cam_id]):
                        if try_acquire_control(cam_id, now):
                            target_found_this_frame = True; target_last_seen_time = now; cam_last_seen[cam_id] = now
                            print(f"[CONTROL] CAM{cam_id} 备用切换获取控制权")

                if target_id is not None and not target_found_this_frame:
                    has_control = (current_cam == cam_id)
                    if has_control:
                        time_since_lost = now - target_last_seen_time
                        if time_since_lost > TRACKER_GRACE_PERIOD:
                            release_control(); should_close_tracker_window = True
                        else:
                            last_box = last_good_box.get(target_id)
                            if last_box:
                                lx1, ly1, lx2, ly2 = last_box
                                if lx2 > lx1 and ly2 > ly1:
                                    live_crop = frame[ly1:ly2, lx1:lx2]
                                    if live_crop.size > 0:
                                        crop_h, crop_w = live_crop.shape[:2]
                                        target_w, target_h = TRACKER_WINDOW_SIZE
                                        scale = max(target_w / crop_w, target_h / crop_h)
                                        new_w, new_h = int(crop_w * scale), int(crop_h * scale)
                                        resized = cv2.resize(live_crop, (new_w, new_h))
                                        start_x = max(0, (new_w - target_w) // 2)
                                        start_y = max(0, (new_h - target_h) // 2)
                                        lost_view = resized[start_y:start_y + target_h, start_x:start_x + target_w]
                                        if time_since_lost > 0.5:
                                            text = f"Re-acquiring... ({TRACKER_GRACE_PERIOD - time_since_lost:.1f}s)"
                                            cv2.putText(lost_view, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                                        tracker_view_to_show = lost_view

                control_status = f"Control: CAM{current_cam}" if current_cam != -1 else "Control: NONE"
            finally:
                global_lock.release()
        else: 
            control_status = "Control: [LOCKED]"

        for sid, info in boxes_to_draw.items():
            x1, y1, x2, y2 = info['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), info['color'], info['thickness'])
            cv2.putText(frame, info['label'], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, info['color'], 2)
        
        if frame_count % 120 == 0: gc.collect()

        mirror_info = " (镜像)" if ENABLE_CAMERA_MIRROR else ""
        cam_info = f"CAM {cam_id}{mirror_info} | Active: {len(cam_active_sids)} | {control_status}"
        cv2.putText(frame, cam_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if cam_id == 0 and len(cam0_right_exit_queue) > 0:
            exit_info = f"Right Exit Queue: {len(cam0_right_exit_queue)}"
            cv2.putText(frame, exit_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        elif cam_id == 2 and len(cam2_left_exit_queue) > 0:
            exit_info = f"Left Exit Queue: {len(cam2_left_exit_queue)}"
            cv2.putText(frame, exit_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        yield { "main_frame": frame, "tracker_view": tracker_view_to_show, 
                "should_close_tracker": should_close_tracker_window, "active_sids": cam_active_sids }

    cap.release()
    print(f"[INFO] CAM {cam_id} 生成器结束")