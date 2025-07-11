# & 'C:\Users\ASUS\yolovs\Scripts\Activate.ps1'
import os
import time
import pickle
import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

DIST_TH    = 0.35             # 纯OSNet L2距离阈值：同人≈0.15-0.35，异人>0.6
DIST_TH_RELAXED = 0.45        # 放宽的L2距离阈值  
DIST_TH_SAFE = 0.30          # 多人场景使用的距离阈值
BANK_FILE  = "idbank.pkl"
TARGET_FPS = 60
MAX_FEATS  = 15              # 减少每个ID保存的最大特征数，保持特征库纯净度
CLEAR_BANK_ON_EXIT = True
TRACKER_WINDOW_NAME = "Tracker View"
TRACKER_WINDOW_SIZE = (800, 450)
HEADROOM_RATIO = 0.20
TRACKER_GRACE_PERIOD = 3.0          # 延长宽限期，减少频繁的Re-acquiring显示
JITTER_THRESHOLD = 1.5              # 放宽抖动阈值，减少过度稳定化
STABILITY_BUFFER_SIZE = 2           # 减少缓冲区大小，提高响应性
REENTRY_TIME_THRESHOLD = 3.0
PREDICTION_CONFIDENCE_THRESHOLD = 0.4  # 预测置信度阈值
USE_PREDICTION_WHEN_LOST = True  # 是否在目标丢失时使用预测
MIN_LOST_TIME_FOR_RELAXED_MATCHING = 3.0  # 增加最小丢失时间，多人场景下更保守


target_id = None
current_detections = []
target_last_seen_time = 0
last_good_box = {}
stabilized_boxes = {}
id_last_seen = {}
id_absence_duration = {}
stability_buffers = {}  # 稳定性缓冲区 {sid: [box1, box2, box3, ...]}
lost_tracklets = {}  # 存储丢失的tracklet信息 {sid: {'last_box': box, 'last_time': time, 'feature': feat, 'confidence': conf}}
max_lost_time = 8.0  # 增加最大丢失时间，更好处理长时间遮挡和出入画面

# 运动预测相关数据结构
motion_history = {}  # {sid: [(time, center_x, center_y, w, h), ...]} 存储运动历史
prediction_cache = {}  # {sid: {'predicted_box': box, 'confidence': conf, 'time': time}} 预测缓存
motion_models = {}  # {sid: {'velocity': (vx, vy), 'acceleration': (ax, ay), 'last_update': time}} 运动模型

def update_motion_history(sid, bbox, timestamp):
    """更新运动历史记录"""
    if sid not in motion_history:
        motion_history[sid] = []
    
    x1, y1, x2, y2 = bbox
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1
    
    motion_history[sid].append((timestamp, center_x, center_y, w, h))
    
    # 保持最近10个位置记录
    if len(motion_history[sid]) > 10:
        motion_history[sid].pop(0)

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
    last_time, last_x, last_y, last_w, last_h = last_record
    
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




def validate_bbox(bbox, frame_w, frame_h, min_size=15, allow_large_person=True):
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
    
    # 简化的宽高比检查 - 大幅放宽
    if w > h * 8 or h > w * 8:  # 放宽宽高比限制
        return False
    
    # 简化的面积检查 - 支持近距离大框
    if allow_large_person:
        # 对人体检测，允许更大的框（近距离）
        if w > frame_w * 0.95 and h > frame_h * 0.95:
            return False
    else:
        # 其他情况的面积限制
        if w * h > frame_w * frame_h * 0.8:
            return False
        
    return True

def create_optimized_tracker():
    """创建平衡的DeepSORT追踪器，兼顾ID稳定性和多人场景控制"""
    tracker = DeepSort(
        max_age=120,          # 增加最大年龄，2秒内消失都算同一轨迹
        n_init=2,             # 降低初始化要求，提高ID稳定性
        nms_max_overlap=0.4,  # 适中的NMS阈值
        max_cosine_distance=0.35,  # OSNet L2距离阈值
        nn_budget=150,        # 适中的特征预算
        override_track_class=None,
        embedder=None,        # 关闭内置mobilenet，使用外部OSNet
        half=False,          # 外部特征不需要半精度
        bgr=True,            # 输入为BGR格式
        embedder_gpu=False,  # 外部特征提取
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
print(f"[INFO] === 统一OSNet特征架构 ===")
print(f"[INFO] DeepSORT使用OSNet特征，max_cosine_distance=0.35")
print(f"[INFO] 距离阈值 - 标准:{DIST_TH}, 安全:{DIST_TH_SAFE}")
print(f"[INFO] 运动预测功能: {'启用' if USE_PREDICTION_WHEN_LOST else '禁用'}")
print(f"[INFO] 预测置信度阈值: {PREDICTION_CONFIDENCE_THRESHOLD}")
print(f"[INFO] 跟踪器宽限期: {TRACKER_GRACE_PERIOD}秒")
print(f"[INFO] 抖动阈值: {JITTER_THRESHOLD}")
print(f"[INFO] 稳定性缓冲区: {STABILITY_BUFFER_SIZE}帧")
print(f"[INFO] === 防闪烁优化已启用 ===")
print(f"[INFO] 按 'H' 键查看控制帮助")

def get_stabilized_box(sid, new_box, now):
    """优化的防抖函数，减少过度稳定化导致的闪烁"""
    # 初始化缓冲区
    if sid not in stability_buffers:
        stability_buffers[sid] = [new_box]
        stabilized_boxes[sid] = new_box
        return new_box
    
    buffer = stability_buffers[sid]
    buffer.append(new_box)
    
    # 保持缓冲区大小
    if len(buffer) > STABILITY_BUFFER_SIZE:
        buffer.pop(0)
    
    # 如果缓冲区未满，使用当前帧结果（提高响应性）
    if len(buffer) < STABILITY_BUFFER_SIZE:
        stabilized_boxes[sid] = new_box
        return new_box
    
    # 计算缓冲区内的变化幅度
    centers = []
    sizes = []
    for box in buffer:
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        w, h = box[2] - box[0], box[3] - box[1]
        centers.append((cx, cy))
        sizes.append((w, h))
    
    # 计算位置变化的标准差
    cx_std = np.std([c[0] for c in centers])
    cy_std = np.std([c[1] for c in centers])
    pos_stability = np.sqrt(cx_std*cx_std + cy_std*cy_std)
    
    # 计算尺寸变化的标准差
    w_std = np.std([s[0] for s in sizes])
    h_std = np.std([s[1] for s in sizes])
    size_stability = max(w_std, h_std)
    
    last_box = stabilized_boxes.get(sid, new_box)
    
    # 放宽的稳定性判断逻辑
    if pos_stability < JITTER_THRESHOLD * 0.5 and size_stability < 3:
        # 非常稳定，使用平均值
        avg_cx = np.mean([c[0] for c in centers])
        avg_cy = np.mean([c[1] for c in centers])
        avg_w = np.mean([s[0] for s in sizes])
        avg_h = np.mean([s[1] for s in sizes])
        
        # 构建稳定化边界框
        sx1, sy1 = int(avg_cx - avg_w/2), int(avg_cy - avg_h/2)
        sx2, sy2 = int(avg_cx + avg_w/2), int(avg_cy + avg_h/2)
        stabilized_box = (sx1, sy1, sx2, sy2)
    elif pos_stability < JITTER_THRESHOLD and size_stability < 8:
        # 适度稳定，轻微平滑
        new_cx, new_cy = (new_box[0] + new_box[2]) / 2, (new_box[1] + new_box[3]) / 2
        new_w, new_h = new_box[2] - new_box[0], new_box[3] - new_box[1]
        last_cx, last_cy = (last_box[0] + last_box[2]) / 2, (last_box[1] + last_box[3]) / 2
        last_w, last_h = last_box[2] - last_box[0], last_box[3] - last_box[1]
        
        # 轻微平滑（70%新值 + 30%旧值）
        smooth_cx = new_cx * 0.7 + last_cx * 0.3
        smooth_cy = new_cy * 0.7 + last_cy * 0.3
        smooth_w = new_w * 0.8 + last_w * 0.2
        smooth_h = new_h * 0.8 + last_h * 0.2
        
        sx1, sy1 = int(smooth_cx - smooth_w/2), int(smooth_cy - smooth_h/2)
        sx2, sy2 = int(smooth_cx + smooth_w/2), int(smooth_cy + smooth_h/2)
        stabilized_box = (sx1, sy1, sx2, sy2)
    else:
        # 快速移动或变化，直接使用当前帧
        stabilized_box = new_box
    
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
    """修复：使用单张图片输入，避免16张重复图片的错误"""
    rgb = cv2.cvtColor(cv2.resize(bgr, (128, 256)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb -= (0.485, 0.456, 0.406)
    rgb /= (0.229, 0.224, 0.225)
    # 修复：直接使用单张图片，不重复16次
    blob = rgb.transpose(2, 0, 1)[None].astype(np.float32)  # 1×3×256×128
    feat = sess.run(None, {inp_name: blob})[0][0]
    feat /= (np.linalg.norm(feat) + 1e-6)
    return feat.astype(np.float32)

def extract_combined_feature(bgr: np.ndarray) -> tuple:
    """提取组合特征：OSNet + 颜色"""
    osnet_feat = extract_feat(bgr)
    color_feat = extract_color_feature(bgr)
    return osnet_feat, color_feat

def calculate_combined_distance(feat1_tuple, feat2_tuple):
    """计算OSNet特征距离 - 简化版本，只使用OSNet特征"""
    try:
        # 安全地提取OSNet特征
        if isinstance(feat1_tuple, tuple) and len(feat1_tuple) >= 1:
            osnet1 = feat1_tuple[0]
        else:
            osnet1 = feat1_tuple
        
        if isinstance(feat2_tuple, tuple) and len(feat2_tuple) >= 1:
            osnet2 = feat2_tuple[0]
        else:
            osnet2 = feat2_tuple
        
        # 确保是numpy数组并转换类型
        osnet1 = np.asarray(osnet1)
        osnet2 = np.asarray(osnet2)
        
        # 纯OSNet L2距离
        return float(np.linalg.norm(osnet1 - osnet2))
    except Exception as e:
        print(f"[WARN] Distance calculation failed: {e}")
        return 999.0

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
    """优化的ID分配，防止ID冲突，在多人场景下使用严格策略"""
    global next_sid
    if not id_bank:
        id_bank[next_sid] = [feat]
        new_id = next_sid
        next_sid += 1
        return new_id, 1.0
    
    
    best_id, best_d = None, float("inf")
    
    # 优化：只检查最近的N个特征，减少计算量
    max_features_to_check = 3  # 只检查每个ID的最近3个特征
    
    # 判断是否为多人场景
    is_multi_person = len(sids_to_exclude) >= 2
    
    for sid, feats in id_bank.items():
        # 强制排除当前活跃的ID，防止ID冲突
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
    
    # 多人场景下使用更严格的匹配条件
    if is_multi_person:
        # 多人场景：使用较严格的阈值
        distance_threshold = DIST_TH_SAFE
        min_confidence_threshold = 0.35  # 降低置信度要求
    else:
        # 单人场景：使用标准阈值
        distance_threshold = DIST_TH
        min_confidence_threshold = 0.35 if len(sids_to_exclude) > 0 else 0.3
    
    print(f"[DBG] best_d={best_d:.3f}  dist_th={distance_threshold:.3f}  best_id={best_id}")
    
    # 加强匹配条件，防止误分配
    if best_id is not None and best_d < distance_threshold and best_id not in sids_to_exclude:
        # 简化置信度计算，减少分支判断
        distance_conf = max(0, 1.0 - best_d / distance_threshold)
        match_confidence = distance_conf * 0.8 + detection_conf * 0.2  # 更重视特征匹配
        
        if match_confidence > min_confidence_threshold:
            print(f"[DBG] SAME id={best_id}  d={best_d:.3f}  conf={match_confidence:.3f}")
            id_bank[best_id].append(feat)
            if len(id_bank[best_id]) > MAX_FEATS:
                id_bank[best_id].pop(0)
            return best_id, match_confidence
    
    # 创建新ID，确保不与现有ID冲突
    new_id = next_sid
    while new_id in id_bank or new_id in sids_to_exclude:
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
    """处理YOLO检测和DeepSORT追踪，平衡检测质量和ID稳定性"""
    # YOLO检测 - 交回NMS给YOLO处理
    res = yolo.predict(frame, conf=0.5, classes=[0], iou=0.6, imgsz=640, verbose=False)[0]  # 提高阈值，让YOLO做NMS
    
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
            # 放宽检测框质量验证，支持近距离
            if validate_bbox((x1, y1, x2, y2), frame_w, frame_h, min_size=20):  # 降低最小尺寸，支持近距离小框
                # 放宽面积和长宽比检查
                area = w * h
                aspect_ratio = w / h if h > 0 else 0
                # 大幅放宽限制，支持近距离大框检测
                if 500 < area < 150000 and 0.2 < aspect_ratio < 5.0:  # 大幅放宽面积和长宽比，支持近距离
                    valid_detections.append(((x1, y1, w, h), conf))
    
    # 按置信度排序，只保留高质量检测
    valid_detections.sort(key=lambda x: x[1], reverse=True)
    
    # 简化过滤：主要依赖YOLO的NMS，只做基本验证
    filtered_detections = valid_detections  # 直接使用YOLO的NMS结果
    
    # 减少检测数量，提高性能
    max_detections = 8  # 适当增加检测数量，支持多人场景
    
    dets = []          # 只放 bbox / 置信度 / 类别
    embeddings = []    # 与 dets 一一对应的 512 维特征
    
    for (x1, y1, w, h), conf in filtered_detections[:max_detections]:
        try:
            x2, y2 = x1 + w, y1 + h
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            osnet_feat = extract_feat(crop)          # 512 维
            dets.append([ [x1, y1, w, h], conf, 0 ]) # 只有三项
            embeddings.append(osnet_feat)
        except Exception as e:
            print(f"[WARN] Failed to extract feature for detection: {e}")
            continue

    # 更新追踪器
    if dets:
        deepsort_tracks = tracker.update_tracks(
            dets,
            embeds=embeddings,   # 关键：把特征列表传进来
            frame=frame
        )
        # 添加调试信息：显示DeepSORT的匹配结果
        print(f"[DBG] DeepSORT: 检测数={len(dets)}, 轨迹数={len(deepsort_tracks)}")
        for i, track in enumerate(deepsort_tracks):
            if hasattr(track, 'track_id') and hasattr(track, 'is_confirmed'):
                status = "confirmed" if track.is_confirmed() else "tentative"
                print(f"[DBG] Track {track.track_id}: {status}")
        
        # 添加特征匹配调试信息
        debug_feature_matching(embeddings, deepsort_tracks)
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
    """平衡的tracklet恢复策略，多人场景下使用保守策略"""
    global lost_tracklets, next_sid, track_id_to_sid
    
    # 统计当前帧中的人数
    current_frame_person_count = len(new_tracks)
    
    # 多人场景下使用严格策略，但保留恢复功能
    if current_frame_person_count >= 4:
        # 4人以上场景才完全禁用恢复
        if lost_tracklets:
            lost_tracklets.clear()
        return new_tracks, []
    
    # 调整恢复频率 - 保持合理的恢复机会
    recovery_interval = 15 if current_frame_person_count >= 3 else (8 if current_frame_person_count >= 2 else 5)
    if int(now * TARGET_FPS) % recovery_interval != 0:
        return new_tracks, []
    
    # 如果没有丢失的tracklets，直接返回
    if not lost_tracklets:
        return new_tracks, []
    
    # 获取当前活跃的SID列表（简化版本）
    current_active_sids = set(int(track[4]) for track in new_tracks)
    
    recovered_sids = []
    
    # 首先清理过期的丢失tracklets - 给足够时间恢复
    max_lost_time = 10.0 if current_frame_person_count >= 3 else (8.0 if current_frame_person_count >= 2 else 6.0)
    expired_sids = []
    for lost_sid, lost_info in lost_tracklets.items():
        if now - lost_info['last_time'] > max_lost_time:
            expired_sids.append(lost_sid)
    for sid in expired_sids:
        del lost_tracklets[sid]
    
    # 清理已经被重新激活的ID（防止冲突）
    conflicted_sids = []
    for lost_sid in lost_tracklets.keys():
        if lost_sid in current_active_sids:
            conflicted_sids.append(lost_sid)
            print(f"[WARN] 检测到ID冲突，移除丢失列表中的ID {lost_sid}（已被重新使用）")
    
    for sid in conflicted_sids:
        del lost_tracklets[sid]
    
    # 如果清理后没有丢失的tracklets，直接返回
    if not lost_tracklets:
        return new_tracks, []
    
    # 限制处理的track数量 - 保持恢复能力
    max_tracks_to_check = 1 if current_frame_person_count >= 3 else (2 if current_frame_person_count >= 2 else 3)
    tracks_to_check = new_tracks[:max_tracks_to_check]
    
    for track in tracks_to_check:
        x1, y1, x2, y2, track_id = map(int, track)
        
        # 快速验证边界框
        if x2 <= x1 or y2 <= y1:
            continue
        
        # 适中的特征提取条件
        bbox_area = (x2 - x1) * (y2 - y1)
        if bbox_area < 800 or bbox_area > 50000:  # 适中的面积限制
            continue
            
        try:
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            current_feature = extract_combined_feature(crop)
        except:
            continue
        
        # 适中的匹配算法
        best_lost_sid = None
        best_score = float('inf')
        
        for lost_sid, lost_info in lost_tracklets.items():
            if lost_info['feature'] is None:
                continue
            
            # 确保要恢复的ID不在当前活跃列表中
            if lost_sid in current_active_sids:
                continue
                
            # 调整最小丢失时间要求 - 确保稳定性
            min_lost_time = 4.0 if current_frame_person_count >= 3 else (3.0 if current_frame_person_count >= 2 else 2.0)
            lost_time = now - lost_info['last_time']
            if lost_time < min_lost_time:
                continue
            
            # 特征距离匹配
            feat_distance = calculate_combined_distance(current_feature, lost_info['feature'])
            
            # 适中的尺寸一致性检查
            current_w, current_h = x2 - x1, y2 - y1
            if 'last_box' in lost_info:
                last_x1, last_y1, last_x2, last_y2 = lost_info['last_box']
                last_w, last_h = last_x2 - last_x1, last_y2 - last_y1
                
                # 计算尺寸变化比例
                w_ratio = max(current_w, last_w) / min(current_w, last_w) if min(current_w, last_w) > 0 else 10
                h_ratio = max(current_h, last_h) / min(current_h, last_h) if min(current_h, last_h) > 0 else 10
                
                # 适中的尺寸检查
                max_size_ratio = 1.4 if current_frame_person_count >= 2 else 1.6
                if w_ratio > max_size_ratio or h_ratio > max_size_ratio:
                    continue
            
            # 分层特征匹配阈值 - 修复：调整为L2距离的正确阈值
            threshold = 0.25 if current_frame_person_count >= 3 else (0.3 if current_frame_person_count >= 2 else 0.35)
            if feat_distance < best_score and feat_distance < threshold:
                best_score = feat_distance
                best_lost_sid = lost_sid
        
        # 最终检查：确保要恢复的ID确实不在活跃列表中
        if (best_lost_sid is not None and 
            best_lost_sid not in current_active_sids):
            
            # 注意：由于现在track_id直接等于SID，无需映射
            # 这里的恢复逻辑需要重新设计，暂时简化
            recovered_sids.append(best_lost_sid)
            
            # 更新特征库
            if best_lost_sid in id_bank:
                id_bank[best_lost_sid].append(current_feature)
                if len(id_bank[best_lost_sid]) > MAX_FEATS:
                    id_bank[best_lost_sid].pop(0)
            
            # 从丢失列表中移除
            del lost_tracklets[best_lost_sid]
            
            scene_type = "多人" if current_frame_person_count >= 2 else "单人"
            print(f"[INFO] {scene_type}场景下恢复ID {best_lost_sid}，匹配分数: {best_score:.3f}")
            
            # 3人以上场景只恢复一个ID，避免混乱
            if current_frame_person_count >= 3:
                break
    
    return new_tracks, recovered_sids

def process_tracks_safely(tracks, frame, frame_w, frame_h, now):
    """安全地处理tracks，避免异常边界框，防止ID冲突"""
    global target_id, target_last_seen_time, last_good_box, lost_tracklets
    
    # 首先尝试恢复丢失的tracklets
    tracks, recovered_sids = try_recover_lost_tracklets(tracks, frame, frame_w, frame_h, now)
    
    active_track_ids = {int(t[4]) for t in tracks}
    active_sids = set()
    
    # 记录即将丢失的tracklets（简化版本）
    for track in tracks:
        active_track_ids.add(int(track[4]))
    
    # 清理过期的丢失tracklets
    for sid in list(lost_tracklets.keys()):
        if sid in active_track_ids:
            # 如果ID重新出现，从丢失列表中移除
            del lost_tracklets[sid]
    
    sids_in_current_frame = set()
    target_found_this_frame = False
    
    for track in tracks:
        x1, y1, x2, y2, track_id = map(int, track)
        
        # 验证边界框
        if not validate_bbox((x1, y1, x2, y2), frame_w, frame_h):
            continue  # 静默跳过无效框，减少日志输出
            
        # 简化ID映射：直接使用track_id作为SID
        sid = track_id
            
        sids_in_current_frame.add(sid)
        active_sids.add(sid)
        
        # 使用稳定化的边界框
        stabilized_box = get_stabilized_box(sid, (x1, y1, x2, y2), now)
        
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
            
        current_detections.append((adj_x1, adj_y1, adj_x2, adj_y2, sid))
        
        # 处理目标追踪
        if sid == target_id:
            target_found_this_frame = True
            target_last_seen_time = time.time()
            last_good_box[target_id] = adj_box
            
            target_crop = frame[adj_y1:adj_y2, adj_x1:adj_x2]
            if target_crop.size > 0:
                cv2.imshow(TRACKER_WINDOW_NAME, cv2.resize(target_crop, TRACKER_WINDOW_SIZE))
        
        # 更新运动历史（用于预测）
        update_motion_history(sid, adj_box, now)
        
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

def select_target_id(event, x, y, flags, param):
    global target_id, target_last_seen_time, last_good_box, filter_sets
    if event == cv2.EVENT_LBUTTONDOWN:
        if target_id is not None:
            if target_id in last_good_box:
                del last_good_box[target_id]
        target_id = None
        for det in current_detections:
            x1, y1, x2, y2, sid = det
            if x1 < x < x2 and y1 < y < y2:
                target_id = sid
                print(f"[INFO] Target locked on ID: {sid}")
                target_last_seen_time = time.time()
                break
    elif event == cv2.EVENT_RBUTTONDOWN:
        print("[INFO] Target unlocked.")
        if target_id is not None:
            if target_id in last_good_box:
                del last_good_box[target_id]
        target_id = None
        try:
            cv2.destroyWindow(TRACKER_WINDOW_NAME)
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

# 主程序
MAIN_WINDOW_NAME = "YOLO Tracking"
cv2.namedWindow(MAIN_WINDOW_NAME)
cv2.setMouseCallback(MAIN_WINDOW_NAME, select_target_id)


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
interval = 1.0 / TARGET_FPS
last_t = 0.0
frame_h, frame_w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        now = time.time()
        if now - last_t < interval:
            time.sleep(max(0, interval-(now-last_t)))
            continue
        last_t = now
        
        current_detections.clear()
        
        # 处理检测和追踪
        tracks = process_detections_and_tracks(yolo, tracker, frame, frame_w, frame_h)
        
        # 降低调试信息频率，减少I/O开销
        if len(tracks) > 12 and int(now * TARGET_FPS) % 30 == 0:  # 每30帧才输出一次调试信息
            debug_detection_info(tracks, [], frame_w, frame_h)
        
        # 安全处理tracks
        target_found_this_frame, active_sids = process_tracks_safely(
            tracks, frame, frame_w, frame_h, now
        )
        
        # 清理无效的稳定化框和缓冲区
        for sid_to_clean in list(stabilized_boxes.keys()):
            if sid_to_clean not in active_sids:
                del stabilized_boxes[sid_to_clean]
        for sid_to_clean in list(stability_buffers.keys()):
            if sid_to_clean not in active_sids:
                del stability_buffers[sid_to_clean]
        
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
        
        # 处理目标丢失的情况 - 优化显示逻辑，减少闪烁
        if target_id is not None and not target_found_this_frame:
            time_since_lost = time.time() - target_last_seen_time
            if time_since_lost > TRACKER_GRACE_PERIOD:
                try:
                    cv2.destroyWindow(TRACKER_WINDOW_NAME)
                except:
                    pass
            else:
                # 只有在目标确实丢失超过0.5秒后才显示Re-acquiring
                if time_since_lost > 0.5:
                    last_box = last_good_box.get(target_id)
                    if last_box:
                        lx1, ly1, lx2, ly2 = last_box
                        if lx2 > lx1 and ly2 > ly1:
                            live_crop = frame[ly1:ly2, lx1:lx2]
                            if live_crop.size > 0:
                                lost_view = cv2.resize(live_crop, TRACKER_WINDOW_SIZE)
                                # 显示剩余时间，让用户了解状态
                                remaining_time = TRACKER_GRACE_PERIOD - time_since_lost
                                status_text = f"Re-acquiring... ({remaining_time:.1f}s)"
                                cv2.putText(lost_view, status_text, (20, 40), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                                cv2.imshow(TRACKER_WINDOW_NAME, lost_view)
                else:
                    # 短时间丢失时，继续显示最后的目标窗口但不显示Re-acquiring
                    last_box = last_good_box.get(target_id)
                    if last_box:
                        lx1, ly1, lx2, ly2 = last_box
                        if lx2 > lx1 and ly2 > ly1:
                            live_crop = frame[ly1:ly2, lx1:lx2]
                            if live_crop.size > 0:
                                lost_view = cv2.resize(live_crop, TRACKER_WINDOW_SIZE)
                                cv2.imshow(TRACKER_WINDOW_NAME, lost_view)
        
        cv2.imshow(MAIN_WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        
        # 键盘控制
        if key in (ord("q"), 27):  # Q键或ESC退出
            break
        elif key == ord("p"):  # P键切换预测功能
            globals()['USE_PREDICTION_WHEN_LOST'] = not USE_PREDICTION_WHEN_LOST
            status = "启用" if USE_PREDICTION_WHEN_LOST else "禁用"
            print(f"[INFO] 运动预测功能已{status}")
        elif key == ord("h"):  # H键显示帮助
            print("\n=== 键盘控制 ===")
            print("Q/ESC: 退出程序")
            print("P: 切换运动预测功能")
            print("H: 显示帮助信息")
            print("鼠标左键: 选择目标")
            print("鼠标右键: 取消目标")
            print("==================\n")
            
finally:
    cap.release()
    cv2.destroyAllWindows()
    if CLEAR_BANK_ON_EXIT:
        if os.path.exists(BANK_FILE):
            os.remove(BANK_FILE)
    else:
        with open(BANK_FILE, "wb") as f:
            pickle.dump(id_bank, f)