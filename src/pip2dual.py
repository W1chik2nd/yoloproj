# & 'C:\Users\ASUS\yolovs\Scripts\Activate.ps1'
import os
import time
import pickle
import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

DIST_TH    = 0.85             # 进一步放宽距离阈值，提高重识别成功率
DIST_TH_RELAXED = 1.5        # 降低放宽的距离阈值
CROSS_CAMERA_TH = 2.0        # 跨相机匹配的放宽阈值
BANK_FILE  = "idbank.pkl"
TARGET_FPS = 60
MAX_FEATS  = 15              # 减少每个ID保存的最大特征数，保持特征库纯净度
CLEAR_BANK_ON_EXIT = True
TRACKER_WINDOW_NAME_CAM0 = "Camera 0"
TRACKER_WINDOW_NAME_CAM1 = "Camera 1"
TRACKER_WINDOW_SIZE = (800, 450)
HEADROOM_RATIO = 0.20
TRACKER_GRACE_PERIOD = 1.5
JITTER_THRESHOLD = 1.5
FILTER_MIN_CUTOFF = 0.4
FILTER_BETA = 1.0

# 增强防抖参数
MOTION_DETECTION_THRESHOLD = 3.0  # 运动检测阈值
STATIC_FILTER_STRENGTH = 0.1      # 静止时的强过滤
SLOW_MOTION_FILTER_STRENGTH = 0.3 # 缓慢移动时的过滤强度
FAST_MOTION_FILTER_STRENGTH = 0.8 # 快速移动时的过滤强度
SIZE_CHANGE_THRESHOLD = 0.15      # 尺寸变化阈值(15%)
STABILITY_FRAMES = 5              # 稳定状态判定帧数
REENTRY_TIME_THRESHOLD = 3.0
PREDICTION_CONFIDENCE_THRESHOLD = 0.4  # 预测置信度阈值
USE_PREDICTION_WHEN_LOST = True  # 是否在目标丢失时使用预测

# 双相机配置
CAMERA_0_ID = 0  # 左侧相机
CAMERA_1_ID = 1  # 右侧相机
CROSS_CAMERA_MATCH_TIME = 5.0  # 跨相机匹配的时间窗口
EDGE_ZONE_WIDTH = 0.15  # 边缘区域宽度比例


# 双相机全局变量
target_id = None
current_detections_cam0 = []
current_detections_cam1 = []
target_last_seen_time = 0
last_good_box = {}
stabilized_boxes = {}
filter_sets = {}
id_last_seen = {}
id_absence_duration = {}
lost_tracklets = {}  # 存储丢失的tracklet信息 {sid: {'last_box': box, 'last_time': time, 'feature': feat, 'confidence': conf, 'camera': cam_id}}
max_lost_time = 8.0  # 增加最大丢失时间，更好处理长时间遮挡和出入画面

# 运动预测相关数据结构
motion_history = {}  # {sid: [(time, center_x, center_y, w, h, camera), ...]} 存储运动历史
prediction_cache = {}  # {sid: {'predicted_box': box, 'confidence': conf, 'time': time}} 预测缓存
motion_models = {}  # {sid: {'velocity': (vx, vy), 'acceleration': (ax, ay), 'last_update': time}} 运动模型

# 增强防抖数据结构
motion_states = {}  # {sid: {'state': 'static/slow/fast', 'stable_count': int, 'last_positions': []}}
adaptive_filters = {}  # {sid: {'position': filter, 'size': filter, 'last_update': time}}

# 跨相机匹配相关
cross_camera_candidates = {}  # {sid: {'last_pos': (x, y), 'last_time': time, 'camera': cam_id, 'velocity': (vx, vy)}}
edge_exits = {}  # {sid: {'exit_time': time, 'exit_camera': cam_id, 'exit_side': 'left'/'right', 'velocity': (vx, vy)}}
global_id_counter = 1  # 全局ID计数器
track_id_to_sid = {}  # tracker ID到系统ID的映射

def update_motion_history(sid, bbox, timestamp, camera_id=0):
    """更新运动历史记录"""
    if sid not in motion_history:
        motion_history[sid] = []
    
    x1, y1, x2, y2 = bbox
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1
    
    motion_history[sid].append((timestamp, center_x, center_y, w, h, camera_id))
    
    # 保持最近10个位置记录
    if len(motion_history[sid]) > 10:
        motion_history[sid].pop(0)

def detect_edge_exit(sid, bbox, frame_w, frame_h, camera_id):
    """检测目标是否从边缘离开"""
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    
    # 检测是否在边缘区域
    left_edge = center_x < frame_w * EDGE_ZONE_WIDTH
    right_edge = center_x > frame_w * (1 - EDGE_ZONE_WIDTH)
    
    if left_edge or right_edge:
        # 获取运动方向
        model = calculate_motion_model(sid)
        if model is not None:
            vx, vy = model['velocity']
            # 检测是否向边缘移动
            if (left_edge and vx < -10) or (right_edge and vx > 10):
                exit_side = 'left' if left_edge else 'right'
                edge_exits[sid] = {
                    'exit_time': time.time(),
                    'exit_camera': camera_id,
                    'exit_side': exit_side,
                    'velocity': (vx, vy),
                    'last_pos': (center_x, (y1 + y2) / 2)
                }
                print(f"[INFO] ID {sid} 从相机{camera_id} {exit_side}侧离开")
                return True
    return False

def match_cross_camera_entry(new_tracks, frame_w, frame_h, camera_id, now):
    """匹配跨相机进入的目标"""
    global global_id_counter
    matched_tracks = []
    
    for track in new_tracks:
        x1, y1, x2, y2, track_id = track
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # 检测是否从边缘进入
        from_left = center_x < frame_w * EDGE_ZONE_WIDTH
        from_right = center_x > frame_w * (1 - EDGE_ZONE_WIDTH)
        
        if from_left or from_right:
            # 寻找匹配的离开记录
            best_match_sid = None
            best_match_score = float('inf')
            
            for exit_sid, exit_info in list(edge_exits.items()):
                # 检查时间窗口
                if now - exit_info['exit_time'] > CROSS_CAMERA_MATCH_TIME:
                    del edge_exits[exit_sid]
                    continue
                
                # 检查相机和方向匹配
                expected_entry_side = 'left' if exit_info['exit_side'] == 'right' else 'right'
                actual_entry_side = 'left' if from_left else 'right'
                
                # 相机0右侧离开 -> 相机1左侧进入
                # 相机1左侧离开 -> 相机0右侧进入
                camera_match = False
                if exit_info['exit_camera'] == 0 and camera_id == 1 and exit_info['exit_side'] == 'right' and from_left:
                    camera_match = True
                elif exit_info['exit_camera'] == 1 and camera_id == 0 and exit_info['exit_side'] == 'left' and from_right:
                    camera_match = True
                
                if camera_match:
                    # 计算速度匹配度
                    exit_vx, exit_vy = exit_info['velocity']
                    exit_speed = np.sqrt(exit_vx*exit_vx + exit_vy*exit_vy)
                    
                    # 估算当前速度（如果有足够的历史数据）
                    current_speed = 0
                    if track_id in motion_history and len(motion_history[track_id]) >= 2:
                        recent = motion_history[track_id][-2:]
                        dt = recent[1][0] - recent[0][0]
                        if dt > 0:
                            dx = recent[1][1] - recent[0][1]
                            dy = recent[1][2] - recent[0][2]
                            current_speed = np.sqrt(dx*dx + dy*dy) / dt
                    
                    # 速度匹配评分
                    speed_diff = abs(exit_speed - current_speed) if current_speed > 0 else exit_speed * 0.5
                    
                    # Y位置匹配评分
                    y_diff = abs(center_y - exit_info['last_pos'][1])
                    
                    # 综合评分
                    match_score = speed_diff * 0.6 + y_diff * 0.4
                    
                    if match_score < best_match_score:
                        best_match_score = match_score
                        best_match_sid = exit_sid
            
            # 如果找到匹配，使用原有ID
            if best_match_sid is not None and best_match_score < 100:  # 阈值可调
                # 更新track_id映射
                track_id_to_sid[track_id] = best_match_sid
                del edge_exits[best_match_sid]
                print(f"[INFO] 跨相机匹配成功: ID {best_match_sid} 从相机{camera_id}进入, 评分:{best_match_score:.1f}")
                matched_tracks.append((x1, y1, x2, y2, track_id))
            else:
                matched_tracks.append(track)
        else:
            matched_tracks.append(track)
    
    return matched_tracks

def calculate_motion_model(sid):
    """计算运动模型（速度和加速度）"""
    if sid not in motion_history or len(motion_history[sid]) < 3:
        return None
    
    history = motion_history[sid]
    
    # 计算速度（使用最近3个点）
    recent_points = history[-3:]
    
    # 计算平均速度
    velocities = []
    for i in range(1, len(recent_points)):
        dt = recent_points[i][0] - recent_points[i-1][0]
        if dt > 0:
            # 现在记录格式是 (time, x, y, w, h, camera_id)
            vx = (recent_points[i][1] - recent_points[i-1][1]) / dt
            vy = (recent_points[i][2] - recent_points[i-1][2]) / dt
            velocities.append((vx, vy))
    
    if not velocities:
        return None
    
    # 平均速度
    avg_vx = sum(v[0] for v in velocities) / len(velocities)
    avg_vy = sum(v[1] for v in velocities) / len(velocities)
    
    # 计算加速度（如果有足够的速度数据）
    ax, ay = 0, 0
    if len(velocities) >= 2:
        ax = (velocities[-1][0] - velocities[0][0]) / (len(velocities) - 1)
        ay = (velocities[-1][1] - velocities[0][1]) / (len(velocities) - 1)
    
    return {
        'velocity': (avg_vx, avg_vy),
        'acceleration': (ax, ay),
        'last_update': history[-1][0]
    }

def predict_position(sid, target_time):
    """基于运动模型预测位置"""
    if sid not in motion_history or len(motion_history[sid]) < 2:
        return None, 0.0
    
    # 获取或计算运动模型
    model = calculate_motion_model(sid)
    if model is None:
        return None, 0.0
    
    # 获取最后已知位置
    last_record = motion_history[sid][-1]
    last_time, last_x, last_y, last_w, last_h, last_camera = last_record
    
    # 预测时间差
    dt = target_time - last_time
    if dt <= 0:
        return None, 0.0
    
    # 基于运动模型预测位置
    vx, vy = model['velocity']
    ax, ay = model['acceleration']
    
    # 使用运动学公式：x = x0 + v*t + 0.5*a*t²
    pred_x = last_x + vx * dt + 0.5 * ax * dt * dt
    pred_y = last_y + vy * dt + 0.5 * ay * dt * dt
    
    # 预测边界框（假设尺寸保持相对稳定）
    pred_x1 = int(pred_x - last_w / 2)
    pred_y1 = int(pred_y - last_h / 2)
    pred_x2 = int(pred_x + last_w / 2)
    pred_y2 = int(pred_y + last_h / 2)
    
    # 计算预测置信度
    confidence = calculate_prediction_confidence(sid, dt, model)
    
    return (pred_x1, pred_y1, pred_x2, pred_y2), confidence

def calculate_prediction_confidence(sid, dt, model):
    """计算预测置信度"""
    if model is None:
        return 0.0
    
    # 基础置信度（随时间衰减）
    time_factor = max(0, 1.0 - dt / 2.0)  # 2秒后置信度降为0
    
    # 运动稳定性因子
    vx, vy = model['velocity']
    speed = np.sqrt(vx*vx + vy*vy)
    
    # 速度稳定性：适中的速度有更好的预测性
    if speed < 10:  # 太慢，可能是静止或抖动
        speed_factor = 0.5
    elif speed < 50:  # 适中速度
        speed_factor = 1.0
    else:  # 太快，预测不稳定
        speed_factor = max(0.3, 1.0 - (speed - 50) / 100)
    
    # 历史数据丰富度
    history_factor = min(1.0, len(motion_history.get(sid, [])) / 5.0)
    
    # 综合置信度
    confidence = time_factor * speed_factor * history_factor
    
    return max(0.0, min(1.0, confidence))

def get_predicted_box_if_lost(sid, current_time):
    """获取丢失目标的预测边界框"""
    if not USE_PREDICTION_WHEN_LOST:
        return None, 0.0
    
    # 检查是否在缓存中
    if sid in prediction_cache:
        cache_entry = prediction_cache[sid]
        if current_time - cache_entry['time'] < 0.1:  # 缓存100ms内有效
            return cache_entry['predicted_box'], cache_entry['confidence']
    
    # 计算新的预测
    predicted_box, confidence = predict_position(sid, current_time)
    
    # 更新缓存
    if predicted_box is not None:
        prediction_cache[sid] = {
            'predicted_box': predicted_box,
            'confidence': confidence,
            'time': current_time
        }
    
    return predicted_box, confidence

def cleanup_prediction_data(active_sids):
    """清理预测相关数据"""
    global motion_history, prediction_cache, motion_models
    
    # 清理不活跃的运动历史
    for sid in list(motion_history.keys()):
        if sid not in active_sids:
            del motion_history[sid]
    
    # 清理预测缓存
    for sid in list(prediction_cache.keys()):
        if sid not in active_sids:
            del prediction_cache[sid]
    
    # 清理运动模型
    for sid in list(motion_models.keys()):
        if sid not in active_sids:
            del motion_models[sid]


def alpha(cutoff, freq):
    te = 1.0 / freq
    tau = 1.0 / (2 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / te)

class OneEuroFilter:
    def __init__(self, freq, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.freq, self.min_cutoff, self.beta, self.d_cutoff = freq, min_cutoff, beta, d_cutoff
        self.x_prev, self.dx_prev, self.t_prev = None, None, None
        
    def __call__(self, x, t=None):
        if t is None: 
            t = time.time()
        if self.t_prev is None: 
            self.t_prev = t
        t_e = t - self.t_prev
        if t_e < 1e-6: 
            return self.x_prev if self.x_prev is not None else x
        freq = 1.0 / t_e
        if self.x_prev is None: 
            self.x_prev, self.dx_prev = x, 0.0
            self.t_prev = t
            return x
        dx = (x - self.x_prev) / t_e
        a_d = alpha(self.d_cutoff, freq)
        if self.dx_prev is None: 
            self.dx_prev = dx
        self.dx_prev = (1.0 - a_d) * self.dx_prev + a_d * dx
        cutoff = self.min_cutoff + self.beta * abs(self.dx_prev)
        a = alpha(cutoff, freq)
        x_filtered = (1.0 - a) * self.x_prev + a * x
        self.x_prev, self.t_prev = x_filtered, t
        return x_filtered

def validate_bbox(bbox, frame_w, frame_h, min_size=20, allow_large_person=True):
    """优化的边界框验证，减少计算复杂度"""
    if bbox is None:
        return False
        
    x1, y1, x2, y2 = bbox
    
    # 简化的边界检查
    w, h = x2 - x1, y2 - y1
    
    # 快速检查：尺寸和位置
    if (w < min_size or h < min_size or 
        x1 < 0 or y1 < 0 or x2 > frame_w or y2 > frame_h):
        return False
    
    # 简化的宽高比检查
    if w > h * 5 or h > w * 5:  # 简化为倍数检查，避免除法
        return False
    
    # 简化的面积检查
    if allow_large_person:
        # 对人体检测，只检查是否过大
        if w > frame_w * 0.9 and h > frame_h * 0.9:
            return False
    else:
        # 其他情况的面积限制
        if w * h > frame_w * frame_h * 0.6:
            return False
        
    return True

def create_optimized_tracker():
    """创建优化的DeepSORT追踪器，参考最新的StrongSORT和Basketball-SORT改进"""
    tracker = DeepSort(
        max_age=60,           # 进一步增加最大年龄，更好处理长时间遮挡
        n_init=1,             # 更快的初始化，减少延迟
        nms_max_overlap=0.5,  # 更严格的NMS，减少重复框
        max_cosine_distance=0.5,  # 进一步放宽特征匹配，适应外观变化
        nn_budget=200,        # 增加特征预算，储存更多样的特征
        override_track_class=None,
        embedder="mobilenet", # 使用轻量级嵌入器
        half=True,           # 使用半精度加速
        bgr=True,            # 输入为BGR格式
        embedder_gpu=True,   # 使用GPU加速特征提取
        embedder_model_name=None,
        embedder_wts=None,
        polygon=False,
        today=None
    )
    return tracker

print("[INFO] Loading models and tracker...")
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
tracker = create_optimized_tracker()
track_id_to_sid = {}
print("[INFO] Models loaded.")
print(f"[INFO] 运动预测功能: {'启用' if USE_PREDICTION_WHEN_LOST else '禁用'}")
print(f"[INFO] 预测置信度阈值: {PREDICTION_CONFIDENCE_THRESHOLD}")
print(f"[INFO] 按 'H' 键查看控制帮助")

def get_hybrid_stabilized_box(sid, new_box, now):
    if sid not in filter_sets:
        freq = TARGET_FPS
        filter_sets[sid] = {
            'cx': OneEuroFilter(freq, FILTER_MIN_CUTOFF, FILTER_BETA),
            'cy': OneEuroFilter(freq, FILTER_MIN_CUTOFF, FILTER_BETA),
            'w': OneEuroFilter(freq, FILTER_MIN_CUTOFF * 0.5, FILTER_BETA),  # 对尺寸变化更敏感
            'h': OneEuroFilter(freq, FILTER_MIN_CUTOFF * 0.5, FILTER_BETA)   # 对尺寸变化更敏感
        }
        stabilized_boxes[sid] = new_box
        return new_box
    
    last_box = stabilized_boxes.get(sid, new_box)
    last_cx, last_cy = (last_box[0] + last_box[2]) / 2, (last_box[1] + last_box[3]) / 2
    new_cx, new_cy = (new_box[0] + new_box[2]) / 2, (new_box[1] + new_box[3]) / 2
    dist = np.sqrt((last_cx - new_cx)**2 + (last_cy - new_cy)**2)
    
    # 检查尺寸变化
    last_w, last_h = last_box[2] - last_box[0], last_box[3] - last_box[1]
    new_w, new_h = new_box[2] - new_box[0], new_box[3] - new_box[1]
    size_change_ratio = max(new_w/max(last_w, 1), new_h/max(last_h, 1), 
                           last_w/max(new_w, 1), last_h/max(new_h, 1))
    
    # 如果位置变化小且尺寸变化不大，保持稳定
    if dist < JITTER_THRESHOLD and size_change_ratio < 1.5:
        return last_box
    else:
        box_w, box_h = new_box[2] - new_box[0], new_box[3] - new_box[1]
        smooth_cx = filter_sets[sid]['cx'](new_cx, now)
        smooth_cy = filter_sets[sid]['cy'](new_cy, now)
        smooth_w = filter_sets[sid]['w'](box_w, now)
        smooth_h = filter_sets[sid]['h'](box_h, now)
        sx1, sy1 = int(smooth_cx - smooth_w/2), int(smooth_cy - smooth_h/2)
        sx2, sy2 = int(smooth_cx + smooth_w/2), int(smooth_cy + smooth_h/2)
        stabilized_box = (sx1, sy1, sx2, sy2)
        stabilized_boxes[sid] = stabilized_box
        return stabilized_box

def pan_and_scan_16_9(box, frame_w, frame_h, headroom_ratio=0.15):
    x1, y1, x2, y2 = box
    box_w, box_h = x2 - x1, y2 - y1
    if box_w <= 0 or box_h <= 0:
        return None
    
    # 确保输入边界框在合理范围内
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_w-1, x2)
    y2 = min(frame_h-1, y2)
    box_w, box_h = x2 - x1, y2 - y1
    
    center_x, center_y = x1 + box_w / 2, y1 + box_h / 2
    aspect_ratio_16_9 = 16 / 9
    
    # 计算理想的16:9尺寸
    if (box_w / box_h) > aspect_ratio_16_9:
        new_w, new_h = box_w, int(box_w / aspect_ratio_16_9)
    else:
        new_h, new_w = box_h, int(box_h * aspect_ratio_16_9)
    
    # 应用头部空间偏移
    vertical_offset = int(new_h * headroom_ratio)
    adjusted_center_y = center_y - (vertical_offset / 2)
    
    # 计算理想位置
    ideal_x1, ideal_y1 = int(center_x - new_w / 2), int(adjusted_center_y - new_h / 2)
    ideal_x2, ideal_y2 = int(center_x + new_w / 2), int(adjusted_center_y + new_h / 2)
    
    # 检查是否超出边界，如果超出则需要重新计算以保持16:9比例
    if ideal_x1 < 0 or ideal_x2 >= frame_w or ideal_y1 < 0 or ideal_y2 >= frame_h:
        # 计算可用的最大尺寸
        max_w = frame_w
        max_h = frame_h
        
        # 根据16:9比例计算实际可用尺寸
        if max_w / max_h > aspect_ratio_16_9:
            # 受高度限制
            actual_h = max_h
            actual_w = int(actual_h * aspect_ratio_16_9)
        else:
            # 受宽度限制
            actual_w = max_w
            actual_h = int(actual_w / aspect_ratio_16_9)
        
        # 如果计算出的尺寸比理想尺寸小，使用实际尺寸
        if actual_w < new_w or actual_h < new_h:
            new_w, new_h = actual_w, actual_h
            # 重新计算偏移
            vertical_offset = int(new_h * headroom_ratio)
            adjusted_center_y = center_y - (vertical_offset / 2)
            ideal_x1, ideal_y1 = int(center_x - new_w / 2), int(adjusted_center_y - new_h / 2)
            ideal_x2, ideal_y2 = int(center_x + new_w / 2), int(adjusted_center_y + new_h / 2)
    
    # 平移调整（保持尺寸不变）
    dx = 0
    if ideal_x1 < 0:
        dx = -ideal_x1
    elif ideal_x2 >= frame_w:
        dx = frame_w - 1 - ideal_x2
    
    dy = 0
    if ideal_y1 < 0:
        dy = -ideal_y1
    elif ideal_y2 >= frame_h:
        dy = frame_h - 1 - ideal_y2
    
    final_x1, final_y1 = ideal_x1 + dx, ideal_y1 + dy
    final_x2, final_y2 = ideal_x2 + dx, ideal_y2 + dy
    
    # 最终边界检查和16:9比例验证
    final_x1 = max(0, final_x1)
    final_y1 = max(0, final_y1)
    final_x2 = min(frame_w-1, final_x2)
    final_y2 = min(frame_h-1, final_y2)
    
    # 确保边界框有效
    if final_x2 <= final_x1 or final_y2 <= final_y1:
        return None
    
    # 验证最终比例是否接近16:9
    final_w, final_h = final_x2 - final_x1, final_y2 - final_y1
    final_ratio = final_w / final_h
    
    # 如果比例偏差太大，进行最后的调整
    if abs(final_ratio - aspect_ratio_16_9) > 0.1:
        # 以较小的维度为基准重新计算
        if final_w < final_h * aspect_ratio_16_9:
            # 宽度受限，调整高度
            corrected_h = int(final_w / aspect_ratio_16_9)
            if corrected_h <= final_h:
                center_y_final = (final_y1 + final_y2) / 2
                final_y1 = int(center_y_final - corrected_h / 2)
                final_y2 = int(center_y_final + corrected_h / 2)
                
                # 确保不超出边界
                if final_y1 < 0:
                    final_y2 += -final_y1
                    final_y1 = 0
                elif final_y2 >= frame_h:
                    final_y1 -= (final_y2 - frame_h + 1)
                    final_y2 = frame_h - 1
        else:
            # 高度受限，调整宽度
            corrected_w = int(final_h * aspect_ratio_16_9)
            if corrected_w <= final_w:
                center_x_final = (final_x1 + final_x2) / 2
                final_x1 = int(center_x_final - corrected_w / 2)
                final_x2 = int(center_x_final + corrected_w / 2)
                
                # 确保不超出边界
                if final_x1 < 0:
                    final_x2 += -final_x1
                    final_x1 = 0
                elif final_x2 >= frame_w:
                    final_x1 -= (final_x2 - frame_w + 1)
                    final_x2 = frame_w - 1
    
    # 最终安全检查
    final_x1 = max(0, final_x1)
    final_y1 = max(0, final_y1)
    final_x2 = min(frame_w-1, final_x2)
    final_y2 = min(frame_h-1, final_y2)
    
    if final_x2 <= final_x1 or final_y2 <= final_y1:
        return None
        
    return final_x1, final_y1, final_x2, final_y2

def extract_color_feature(bgr: np.ndarray) -> np.ndarray:
    """提取衣服颜色特征"""
    h, w = bgr.shape[:2]
    # 提取上半身区域（衣服区域）
    upper_body = bgr[int(h*0.1):int(h*0.6), int(w*0.2):int(w*0.8)]
    
    if upper_body.size == 0:
        return np.zeros(64, dtype=np.float32)
    
    # 转换到HSV空间
    hsv = cv2.cvtColor(upper_body, cv2.COLOR_BGR2HSV)
    
    # 计算HSV直方图
    h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256])
    
    # 归一化并合并
    h_hist = h_hist.flatten() / (h_hist.sum() + 1e-6)
    s_hist = s_hist.flatten() / (s_hist.sum() + 1e-6)
    v_hist = v_hist.flatten() / (v_hist.sum() + 1e-6)
    
    # 合并HSV特征
    color_feat = np.concatenate([h_hist, s_hist[:16], v_hist[:16]])
    return color_feat.astype(np.float32)

def extract_feat(bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(cv2.resize(bgr, (128, 256)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb -= (0.485, 0.456, 0.406)
    rgb /= (0.229, 0.224, 0.225)
    blob = np.repeat(rgb.transpose(2, 0, 1)[None], 16, axis=0)
    feat = sess.run(None, {inp_name: blob})[0][0]
    feat /= (np.linalg.norm(feat) + 1e-6)
    return feat.astype(np.float32)

def extract_combined_feature(bgr: np.ndarray) -> tuple:
    """提取组合特征：OSNet + 颜色"""
    osnet_feat = extract_feat(bgr)
    color_feat = extract_color_feature(bgr)
    return osnet_feat, color_feat

def calculate_combined_distance(feat1_tuple, feat2_tuple, color_weight=0.3):
    """计算组合特征距离"""
    osnet1, color1 = feat1_tuple
    osnet2, color2 = feat2_tuple
    
    # OSNet特征距离
    osnet_dist = np.linalg.norm(osnet1 - osnet2)
    
    # 颜色特征距离（使用巴氏距离）
    bhattacharyya_coeff = np.sum(np.sqrt(color1 * color2))
    color_dist = np.sqrt(max(0, 1 - bhattacharyya_coeff))
    
    # 加权组合
    combined_dist = (1 - color_weight) * osnet_dist + color_weight * color_dist
    return combined_dist

if os.path.exists(BANK_FILE) and not CLEAR_BANK_ON_EXIT:
    try:
        with open(BANK_FILE, "rb") as f:
            id_bank = pickle.load(f)
            next_sid = max(id_bank.keys()) + 1 if id_bank else 1
    except:
        id_bank, next_sid = {}, 1
else:
    id_bank, next_sid = {}, 1

def assign_id_with_confidence(feat: tuple, bbox_area: float, detection_conf: float, sids_to_exclude=set()) -> tuple:
    """优化的ID分配，减少计算复杂度"""
    global next_sid
    if not id_bank:
        id_bank[next_sid] = [feat]
        new_id = next_sid
        next_sid += 1
        return new_id, 1.0
    
    best_id, best_d = None, float("inf")
    
    # 优化：只检查最近的N个特征，减少计算量
    max_features_to_check = 3  # 只检查每个ID的最近3个特征
    
    for sid, feats in id_bank.items():
        if sid in sids_to_exclude:
            continue
            
        if not feats:
            continue
        
        # 只检查最新的几个特征，大幅减少计算量
        recent_feats = feats[-max_features_to_check:] if len(feats) > max_features_to_check else feats
        
        # 简化距离计算：只用最小距离，移除复杂的harmonic mean
        min_dist = min(calculate_combined_distance(feat, f) for f in recent_feats)
        
        if min_dist < best_d:
            best_d, best_id = min_dist, sid
    
    # 简化的匹配置信度计算
    if best_id is not None and best_d < DIST_TH:
        # 简化置信度计算，减少分支判断
        distance_conf = max(0, 1.0 - best_d / DIST_TH)
        match_confidence = distance_conf * 0.7 + detection_conf * 0.3
        
        if match_confidence > 0.5:  # 稍微提高阈值，减少误匹配
            id_bank[best_id].append(feat)
            if len(id_bank[best_id]) > MAX_FEATS:
                id_bank[best_id].pop(0)
            return best_id, match_confidence
    
    # 创建新ID
    new_id = next_sid
    while new_id in id_bank:
        new_id += 1
    next_sid = new_id + 1
    id_bank[new_id] = [feat]
    return new_id, 1.0

def assign_id(feat: np.ndarray, sids_to_exclude=set()) -> int:
    """保持向后兼容的简化版本"""
    bbox_area = 1000  # 默认面积
    detection_conf = 0.8  # 默认置信度
    sid, _ = assign_id_with_confidence(feat, bbox_area, detection_conf, sids_to_exclude)
    return sid

def process_detections_and_tracks(yolo, tracker, frame, frame_w, frame_h):
    """处理YOLO检测和DeepSORT追踪"""
    # YOLO检测
    res = yolo.predict(frame, conf=0.45, classes=[0], iou=0.6, imgsz=640, verbose=False)[0]  # 优化参数并设置较小的图像尺寸以提升速度
    dets = []
    
    # 首先收集所有有效检测并按置信度排序
    valid_detections = []
    for box in res.boxes:
        x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
        conf = float(box.conf.cpu().numpy()[0])
        
        # 确保检测框在合理范围内
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(frame_w-1, int(x2)), min(frame_h-1, int(y2))
        
        # 检查检测框是否有效
        if x2 > x1 and y2 > y1:
            w, h = x2 - x1, y2 - y1
            # 额外验证检测框质量
            if validate_bbox((x1, y1, x2, y2), frame_w, frame_h):
                valid_detections.append(((x1, y1, w, h), conf))
    
    # 按置信度排序，保留高质量检测
    valid_detections.sort(key=lambda x: x[1], reverse=True)
    
    # 优化的NMS：简化重叠检测
    filtered_detections = []
    for (x1, y1, w, h), conf in valid_detections:
        x2, y2 = x1 + w, y1 + h
        
        # 简化的重叠检查：只检查中心点距离，避免复杂的IoU计算
        cx, cy = x1 + w/2, y1 + h/2
        overlap_found = False
        
        for (px1, py1, pw, ph), _ in filtered_detections:
            pcx, pcy = px1 + pw/2, py1 + ph/2
            
            # 简化的距离检查，比IoU计算快得多
            distance = ((cx - pcx)**2 + (cy - pcy)**2)**0.5
            min_distance = min(w, h, pw, ph) * 0.5  # 基于最小尺寸的距离阈值
            
            if distance < min_distance:
                overlap_found = True
                break
        
        if not overlap_found:
            filtered_detections.append(((x1, y1, w, h), conf))
    
    # 限制最大检测数量
    max_detections = min(10, len(filtered_detections))  # 动态调整最大数量
    
    for (x1, y1, w, h), conf in filtered_detections[:max_detections]:
        # DeepSORT期望的格式: [x1, y1, w, h]
        dets.append([[x1, y1, w, h], conf, 0])

    # 更新追踪器
    if len(dets) > 0:
        deepsort_tracks = tracker.update_tracks(dets, frame=frame)
    else:
        deepsort_tracks = []

    tracks = []
    for t in deepsort_tracks:
        if not t.is_confirmed():
            continue
            
        # 更安全的边界框获取方式
        try:
            # 优先使用to_tlbr()方法
            if hasattr(t, 'to_tlbr'):
                x1, y1, x2, y2 = t.to_tlbr()
            elif hasattr(t, 'to_tlwh'):
                tlwh = t.to_tlwh()
                x1, y1 = tlwh[0], tlwh[1]
                x2, y2 = tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]
            else:
                # 如果都没有，跳过这个track
                continue
                
            # 确保坐标在合理范围内
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(frame_w-1, int(x2)), min(frame_h-1, int(y2))
            
            # 检查边界框是否有效
            if x2 > x1 and y2 > y1:
                tracks.append([x1, y1, x2, y2, int(t.track_id)])
                
        except Exception as e:
            print(f"[WARN] Error processing track {t.track_id}: {e}")
            continue
    
    return tracks

def try_recover_lost_tracklets(new_tracks, frame, frame_w, frame_h, now):
    """优化的tracklet恢复机制，集成运动预测功能"""
    global lost_tracklets, next_sid
    
    # 如果没有丢失的tracklets，直接返回
    if not lost_tracklets:
        return new_tracks, []
    
    # 限制恢复频率，每5帧才尝试一次恢复，大幅减少计算量
    if int(now * TARGET_FPS) % 5 != 0:
        return new_tracks, []
    
    recovered_sids = []
    new_tracks_enhanced = []
    predicted_tracks = []  # 存储预测的轨迹
    
    # 首先清理过期的丢失tracklets
    expired_sids = []
    for lost_sid, lost_info in lost_tracklets.items():
        if now - lost_info['last_time'] > max_lost_time:
            expired_sids.append(lost_sid)
    for sid in expired_sids:
        del lost_tracklets[sid]
    
    # 如果清理后没有丢失的tracklets，直接返回
    if not lost_tracklets:
        return new_tracks, []
    
    # 生成预测轨迹（如果启用预测功能）
    if USE_PREDICTION_WHEN_LOST:
        for lost_sid in lost_tracklets.keys():
            predicted_box, pred_confidence = get_predicted_box_if_lost(lost_sid, now)
            if predicted_box is not None and pred_confidence >= PREDICTION_CONFIDENCE_THRESHOLD:
                # 验证预测边界框是否在合理范围内
                if validate_bbox(predicted_box, frame_w, frame_h):
                    predicted_tracks.append({
                        'sid': lost_sid,
                        'bbox': predicted_box,
                        'confidence': pred_confidence,
                        'type': 'predicted'
                    })
    
    # 只对前3个最新的tracks尝试恢复，减少计算量
    tracks_to_check = new_tracks[:3] if len(new_tracks) > 3 else new_tracks
    
    for track in tracks_to_check:
        x1, y1, x2, y2, track_id = map(int, track)
        
        # 快速验证边界框
        if x2 <= x1 or y2 <= y1:
            new_tracks_enhanced.append(track)
            continue
        
        # 简化特征提取：只在合理大小的框上提取
        bbox_area = (x2 - x1) * (y2 - y1)
        if bbox_area < 400 or bbox_area > 100000:  # 跳过过小或过大的框
            new_tracks_enhanced.append(track)
            continue
            
        try:
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                new_tracks_enhanced.append(track)
                continue
            current_feature = extract_combined_feature(crop)
        except:
            new_tracks_enhanced.append(track)
            continue
        
        # 增强的匹配算法：结合特征距离和预测位置
        best_lost_sid = None
        best_score = float('inf')
        matching_method = "feature"
        
        for lost_sid, lost_info in lost_tracklets.items():
            if lost_info['feature'] is None:
                continue
            
            # 1. 特征距离匹配
            feat_distance = calculate_combined_distance(current_feature, lost_info['feature'])
            
            # 2. 预测位置匹配（如果可用）
            spatial_score = 0.0
            if USE_PREDICTION_WHEN_LOST:
                predicted_box, pred_confidence = get_predicted_box_if_lost(lost_sid, now)
                if predicted_box is not None and pred_confidence >= PREDICTION_CONFIDENCE_THRESHOLD:
                    # 计算预测位置与当前检测的空间距离
                    pred_cx = (predicted_box[0] + predicted_box[2]) / 2
                    pred_cy = (predicted_box[1] + predicted_box[3]) / 2
                    curr_cx = (x1 + x2) / 2
                    curr_cy = (y1 + y2) / 2
                    
                    spatial_distance = np.sqrt((pred_cx - curr_cx)**2 + (pred_cy - curr_cy)**2)
                    spatial_score = spatial_distance / 100.0  # 归一化空间距离
                    
                    # 综合评分：特征距离 + 空间距离（加权）
                    combined_score = feat_distance * 0.7 + spatial_score * 0.3
                    
                    if combined_score < best_score and feat_distance < DIST_TH_RELAXED:
                        best_score = combined_score
                        best_lost_sid = lost_sid
                        matching_method = "feature+prediction"
                        continue
            
            # 纯特征匹配（fallback）
            # 检查是否为跨相机匹配，使用更宽松的阈值
            threshold = CROSS_CAMERA_TH if lost_sid in edge_exits else DIST_TH * 1.2
            if feat_distance < best_score and feat_distance < threshold:
                best_score = feat_distance
                best_lost_sid = lost_sid
                matching_method = "feature"
        
        # 恢复匹配的tracklet
        final_threshold = CROSS_CAMERA_TH if best_lost_sid in edge_exits else DIST_TH
        if best_lost_sid is not None and best_score < final_threshold:
            track_id_to_sid[track_id] = best_lost_sid
            recovered_sids.append(best_lost_sid)
            
            # 更新特征库
            if best_lost_sid in id_bank:
                id_bank[best_lost_sid].append(current_feature)
                if len(id_bank[best_lost_sid]) > MAX_FEATS:
                    id_bank[best_lost_sid].pop(0)
            
            # 从丢失列表中移除
            del lost_tracklets[best_lost_sid]
            # 减少输出频率
            if len(recovered_sids) <= 2:  # 只输出前2个恢复的ID
                print(f"[INFO] 检测到人物重新进入画面 {best_lost_sid} 使用 {matching_method}")
                # 打印预测置信度
                if matching_method == "feature":
                    predicted_box, pred_confidence = get_predicted_box_if_lost(best_lost_sid, now)
                    if predicted_box:
                        print(f"[INFO] 预测置信度: {pred_confidence:.3f}")
        
        new_tracks_enhanced.append(track)
    
    # 将剩余的tracks直接添加
    if len(new_tracks) > 3:
        new_tracks_enhanced.extend(new_tracks[3:])
    
    return new_tracks_enhanced, recovered_sids

def process_tracks_safely(tracks, frame, frame_w, frame_h, now, camera_id=0):
    """安全地处理tracks，避免异常边界框"""
    global target_id, target_last_seen_time, last_good_box, lost_tracklets
    
    # 首先尝试恢复丢失的tracklets
    tracks, recovered_sids = try_recover_lost_tracklets(tracks, frame, frame_w, frame_h, now)
    
    active_track_ids = {int(t[4]) for t in tracks}
    active_sids = set()
    
    # 记录即将丢失的tracklets
    for tid, sid in list(track_id_to_sid.items()):
        if tid not in active_track_ids and sid not in recovered_sids:
            # 如果这个SID有有效信息，保存到丢失列表
            if sid in stabilized_boxes and sid in id_bank:
                lost_tracklets[sid] = {
                    'last_box': stabilized_boxes[sid],
                    'last_time': now,
                    'feature': id_bank[sid][-1] if id_bank[sid] else None,
                    'confidence': 0.8  # 默认置信度
                }
                # 减少日志输出频率
            del track_id_to_sid[tid]
    
    # 清理无效的track_id映射
    for tid in list(track_id_to_sid.keys()):
        if tid not in active_track_ids:
            del track_id_to_sid[tid]
    
    sids_in_current_frame = set()
    target_found_this_frame = False
    
    for track in tracks:
        x1, y1, x2, y2, track_id = map(int, track)
        
        # 验证边界框
        if not validate_bbox((x1, y1, x2, y2), frame_w, frame_h):
            continue  # 静默跳过无效框，减少日志输出
            
        sid = track_id_to_sid.get(track_id, -1)
        
        if sid == -1:
            # 提取特征前再次验证裁剪区域
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0 and crop.shape[0] > 0 and crop.shape[1] > 0:
                try:
                    feature = extract_combined_feature(crop)
                    bbox_area = (x2 - x1) * (y2 - y1)
                    
                    # 使用改进的ID分配方法，假设检测置信度为0.8
                    sid, match_confidence = assign_id_with_confidence(
                        feature, bbox_area, 0.8, sids_to_exclude=sids_in_current_frame
                    )
                    track_id_to_sid[track_id] = sid
                    
                    # 减少低置信度警告的输出频率
                    if match_confidence < 0.5 and int(now * TARGET_FPS) % 30 == 0:
                        print(f"[WARN] Low confidence match: ID {sid} (conf: {match_confidence:.2f})")
                        
                except Exception as e:
                    print(f"[WARN] Feature extraction failed for track {track_id}: {e}")
                    continue
            else:
                continue
        
        if sid == -1:
            continue
            
        sids_in_current_frame.add(sid)
        active_sids.add(sid)
        
        # 使用稳定化的边界框
        stabilized_box = get_hybrid_stabilized_box(sid, (x1, y1, x2, y2), now)
        
        # 验证稳定化后的边界框
        if not validate_bbox(stabilized_box, frame_w, frame_h):
            continue
            
        adj_box = pan_and_scan_16_9(stabilized_box, frame_w, frame_h, headroom_ratio=HEADROOM_RATIO)
        if adj_box is None:
            continue
            
        adj_x1, adj_y1, adj_x2, adj_y2 = adj_box
        
        # 最终验证调整后的边界框
        if not validate_bbox(adj_box, frame_w, frame_h):
            continue
            
        current_detections = current_detections_cam0 if camera_id == 0 else current_detections_cam1
        current_detections.append((adj_x1, adj_y1, adj_x2, adj_y2, sid))
        
        # 处理目标追踪
        if sid == target_id:
            target_found_this_frame = True
            target_last_seen_time = time.time()
            last_good_box[target_id] = adj_box
            
            target_crop = frame[adj_y1:adj_y2, adj_x1:adj_x2]
            if target_crop.size > 0:
                window_name = TRACKER_WINDOW_NAME_CAM0 if camera_id == 0 else TRACKER_WINDOW_NAME_CAM1
                cv2.imshow(window_name, cv2.resize(target_crop, TRACKER_WINDOW_SIZE))
        
        # 更新运动历史（用于预测）
        update_motion_history(sid, adj_box, now, camera_id)
        
        # 检测边缘离开
        detect_edge_exit(sid, adj_box, frame_w, frame_h, camera_id)
        
        # 绘制边界框
        label_color = (0, 0, 255) if sid == target_id else (0, 255, 0)
        cv2.rectangle(frame, (adj_x1, adj_y1), (adj_x2, adj_y2), label_color, 2)
        
        # 显示ID和预测信息
        label_text = f"ID:{sid}"
        if USE_PREDICTION_WHEN_LOST and sid in motion_history:
            model = calculate_motion_model(sid)
            if model is not None:
                vx, vy = model['velocity']
                speed = np.sqrt(vx*vx + vy*vy)
                if speed > 5:  # 只显示有意义的速度
                    label_text += f" v:{speed:.0f}"
        
        cv2.putText(frame, label_text, (adj_x1, adj_y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)
    
    # 绘制预测轨迹（如果启用）
    if USE_PREDICTION_WHEN_LOST:
        for lost_sid in lost_tracklets.keys():
            predicted_box, pred_confidence = get_predicted_box_if_lost(lost_sid, now)
            if predicted_box is not None and pred_confidence >= PREDICTION_CONFIDENCE_THRESHOLD:
                if validate_bbox(predicted_box, frame_w, frame_h):
                    px1, py1, px2, py2 = predicted_box
                    # 绘制预测边界框（虚线效果）
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 255, 0), 1)
                    cv2.putText(frame, f"P-ID:{lost_sid} ({pred_confidence:.2f})", 
                               (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    return target_found_this_frame, active_sids

def select_target_id(event, x, y, flags, param, camera_id=0):
    global target_id, target_last_seen_time, last_good_box, filter_sets
    if event == cv2.EVENT_LBUTTONDOWN:
        if target_id is not None:
            if target_id in last_good_box:
                del last_good_box[target_id]
            if target_id in filter_sets:
                del filter_sets[target_id]
        target_id = None
        current_detections = current_detections_cam0 if camera_id == 0 else current_detections_cam1
        for det in current_detections:
            x1, y1, x2, y2, sid = det
            if x1 < x < x2 and y1 < y < y2:
                target_id = sid
                print(f"[INFO] Target logged on ID: {sid} from camera {camera_id}")
                target_last_seen_time = time.time()
                break
    elif event == cv2.EVENT_RBUTTONDOWN:
        print("[INFO] Target unlocked.")
        if target_id is not None:
            if target_id in last_good_box:
                del last_good_box[target_id]
            if target_id in filter_sets:
                del filter_sets[target_id]
        target_id = None
        try:
            cv2.destroyWindow(TRACKER_WINDOW_NAME_CAM0)
            cv2.destroyWindow(TRACKER_WINDOW_NAME_CAM1)
        except:
            pass

def debug_detection_info(tracks, dets, frame_w, frame_h):
    """调试函数，打印检测和追踪信息"""
    print(f"\n[DEBUG] Frame size: {frame_w}x{frame_h}")
    print(f"[DEBUG] Detections: {len(dets)}")
    print(f"[DEBUG] Tracks: {len(tracks)}")
    
    abnormal_count = 0
    for i, track in enumerate(tracks):
        x1, y1, x2, y2, track_id = track
        w, h = x2 - x1, y2 - y1
        aspect_ratio = w / h if h > 0 else 0
        
        # 检查是否有异常大小的边界框
        is_abnormal = False
        # 调整大框警告阈值，允许正常的站立姿势
        if w > frame_w * 0.9 or h > frame_h * 0.9:
            print(f"[WARN] Track {track_id} has abnormally large bbox: {w}x{h}")
            is_abnormal = True
        if w < 20 or h < 20:
            print(f"[WARN] Track {track_id} has abnormally small bbox: {w}x{h}")
            is_abnormal = True
        # 调整宽高比警告阈值，允许站立时的高瘦形状
        if aspect_ratio > 4.0 or aspect_ratio < 0.25:
            print(f"[WARN] Track {track_id} has abnormal aspect ratio: {aspect_ratio:.2f}")
            is_abnormal = True
            
        if is_abnormal:
            abnormal_count += 1
            print(f"  -> Track {track_id}: bbox=({x1},{y1},{x2},{y2}), size={w}x{h}, ratio={aspect_ratio:.2f}")
    
    if abnormal_count > 0:
        print(f"[WARN] Found {abnormal_count} abnormal tracks out of {len(tracks)} total")
    print("=" * 50)

# 主程序 - 双相机
MAIN_WINDOW_NAME_CAM0 = "Camera 0 - YOLO Tracking"
MAIN_WINDOW_NAME_CAM1 = "Camera 1 - YOLO Tracking"
cv2.namedWindow(MAIN_WINDOW_NAME_CAM0)
cv2.namedWindow(MAIN_WINDOW_NAME_CAM1)
cv2.setMouseCallback(MAIN_WINDOW_NAME_CAM0, lambda *args: select_target_id(*args, camera_id=0))
cv2.setMouseCallback(MAIN_WINDOW_NAME_CAM1, lambda *args: select_target_id(*args, camera_id=1))

# 初始化双相机
cap0 = cv2.VideoCapture(CAMERA_0_ID, cv2.CAP_DSHOW)
cap1 = cv2.VideoCapture(CAMERA_1_ID, cv2.CAP_DSHOW)
interval = 1.0 / TARGET_FPS
last_t = 0.0

# 获取相机分辨率
frame_h0, frame_w0 = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h1, frame_w1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))

try:
    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        
        if not ret0 or not ret1:
            break
            
        now = time.time()
        if now - last_t < interval:
            time.sleep(max(0, interval-(now-last_t)))
            continue
        last_t = now
        
        current_detections_cam0.clear()
        current_detections_cam1.clear()
        
        # 处理相机0
        tracks0 = process_detections_and_tracks(yolo, tracker, frame0, frame_w0, frame_h0)
        tracks0 = match_cross_camera_entry(tracks0, frame_w0, frame_h0, 0, now)
        
        # 处理相机1  
        tracks1 = process_detections_and_tracks(yolo, tracker, frame1, frame_w1, frame_h1)
        tracks1 = match_cross_camera_entry(tracks1, frame_w1, frame_h1, 1, now)
        
        # 安全处理tracks
        target_found_this_frame0, active_sids0 = process_tracks_safely(
            tracks0, frame0, frame_w0, frame_h0, now, camera_id=0
        )
        target_found_this_frame1, active_sids1 = process_tracks_safely(
            tracks1, frame1, frame_w1, frame_h1, now, camera_id=1
        )
        
        # 合并活跃ID
        active_sids = active_sids0.union(active_sids1)
        target_found_this_frame = target_found_this_frame0 or target_found_this_frame1
        
        # 清理无效的滤波器和稳定化框
        for sid_to_clean in list(filter_sets.keys()):
            if sid_to_clean not in active_sids:
                del filter_sets[sid_to_clean]
        for sid_to_clean in list(stabilized_boxes.keys()):
            if sid_to_clean not in active_sids:
                del stabilized_boxes[sid_to_clean]
        
        # 清理预测相关数据
        cleanup_prediction_data(active_sids)
        
        # 清理过期的丢失tracklets (降低频率)
        if int(now * TARGET_FPS) % 60 == 0:  # 每60帧清理一次，减少计算频率
            expired_count = 0
            for lost_sid in list(lost_tracklets.keys()):
                if now - lost_tracklets[lost_sid]['last_time'] > max_lost_time:
                    del lost_tracklets[lost_sid]
                    expired_count += 1
            
            # 批量输出统计信息，减少I/O
            if expired_count > 0 or len(lost_tracklets) > 0:
                print(f"[INFO] Active: {len(active_sids)}, Lost: {len(lost_tracklets)}, Expired: {expired_count}")
        
        # 处理目标丢失的情况
        if target_id is not None and not target_found_this_frame:
            time_since_lost = time.time() - target_last_seen_time
            if time_since_lost > TRACKER_GRACE_PERIOD:
                try:
                    cv2.destroyWindow(TRACKER_WINDOW_NAME_CAM0)
                    cv2.destroyWindow(TRACKER_WINDOW_NAME_CAM1)
                except:
                    pass
        
        # 显示双相机画面
        cv2.imshow(MAIN_WINDOW_NAME_CAM0, frame0)
        cv2.imshow(MAIN_WINDOW_NAME_CAM1, frame1)
        
        key = cv2.waitKey(1) & 0xFF
        
        # 键盘控制
        if key in (ord("q"), 27):  # Q键或ESC退出
            break
        elif key == ord("p"):  # P键切换预测功能
            globals()['USE_PREDICTION_WHEN_LOST'] = not USE_PREDICTION_WHEN_LOST
            status = "启用" if USE_PREDICTION_WHEN_LOST else "禁用"
            print(f"[INFO] 运动预测功能已{status}")
        elif key == ord("h"):  # H键显示帮助
            print("\n=== 双相机键盘控制 ===")
            print("Q/ESC: 退出程序")
            print("P: 切换运动预测功能")
            print("H: 显示帮助信息")
            print("鼠标左键: 选择目标(任一相机)")
            print("鼠标右键: 取消目标")
            print("支持跨相机ID追踪")
            print("==================\n")
            
finally:
    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()
    if CLEAR_BANK_ON_EXIT:
        if os.path.exists(BANK_FILE):
            os.remove(BANK_FILE)
    else:
        with open(BANK_FILE, "wb") as f:
            pickle.dump(id_bank, f)