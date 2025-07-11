# yolo_sort_osnet_dualcam_color_dedup.py
# ---------------------------------------------------------------------------
# 双摄像头 + Tracker View，ReID 特征 + 衣服颜色直方图融合判 ID（同帧去重版）
# ---------------------------------------------------------------------------
import os
import time
import pickle
import threading
import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
from sort import Sort

# ---------------------------------------------------------------------------
# 参数区
# ---------------------------------------------------------------------------
DIST_TH            = 0.70            # 融合距离阈值（更严格）
_COLOR_WEIGHT      = 0.50            # 颜色直方图权重
BANK_FILE          = "idbank.pkl"
TARGET_FPS         = 60
MAX_FEATS          = 10
CLEAR_BANK_ON_EXIT = True
TRACKER_WINDOW_NAME = "Tracker View"
TRACKER_WINDOW_SIZE = (800, 450)     # 16:9
HEADROOM_RATIO     = 0.20

# ---------------------------------------------------------------------------
# 加载 YOLO + ReID 模型
# ---------------------------------------------------------------------------
print("[INFO] Loading models...")
yolo = YOLO("weights/yolo11s.pt")

try:
    providers = ["CUDAExecutionProvider"]
    sess = ort.InferenceSession("weights/osnet_x1_0_msmt17.onnx", providers=providers)
    print("[INFO] ReID on GPU.")
except Exception:
    providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession("weights/osnet_x0_25_msmt17.onnx", providers=providers)
    print("[WARN] ReID on CPU.")

inp_name = sess.get_inputs()[0].name
print("[INFO] Models loaded.")

# ---------------------------------------------------------------------------
# 全局变量（线程安全访问请配合 bank_lock）
# ---------------------------------------------------------------------------
# id_bank[sid] = [(reid_feat, color_hist), ...]
id_bank: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
next_sid = 1
target_id: int | None = None
current_detections_per_cam: dict[int, list[tuple[int, int, int, int, int]]] = {}
tracker_window_owner: int | None = None
bank_lock = threading.Lock()

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def pan_and_scan_16_9(box, frame_w, frame_h, headroom_ratio=0.20):
    x1, y1, x2, y2 = box
    box_w, box_h = x2 - x1, y2 - y1
    if box_w <= 0 or box_h <= 0:
        return None
    center_x, center_y = x1 + box_w / 2, y1 + box_h / 2
    asp = 16 / 9
    if box_w / box_h > asp:
        new_w, new_h = box_w, int(box_w / asp)
    else:
        new_h, new_w = box_h, int(box_h * asp)
    vertical_offset = int(new_h * headroom_ratio)
    adjusted_center_y = center_y - vertical_offset / 2
    ix1, iy1 = int(center_x - new_w / 2), int(adjusted_center_y - new_h / 2)
    ix2, iy2 = int(center_x + new_w / 2), int(adjusted_center_y + new_h / 2)

    dx = 0
    if ix1 < 0:
        dx = -ix1
    if ix2 >= frame_w:
        dx = frame_w - 1 - ix2
    dy = 0
    if iy1 < 0:
        dy = -iy1
    if iy2 >= frame_h:
        dy = frame_h - 1 - iy2

    final_x1 = max(0, ix1 + dx)
    final_y1 = max(0, iy1 + dy)
    final_x2 = min(frame_w - 1, ix2 + dx)
    final_y2 = min(frame_h - 1, iy2 + dy)
    return final_x1, final_y1, final_x2, final_y2

def create_letterboxed_image(image, target_size):
    target_w, target_h = target_size
    h, w, _ = image.shape
    if w == 0 or h == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_offset, y_offset = (target_w - new_w) // 2, (target_h - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas

# -------------------- 特征提取 --------------------
def extract_feat(bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(cv2.resize(bgr, (128, 256)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb -= (0.485, 0.456, 0.406)
    rgb /= (0.229, 0.224, 0.225)
    blob = np.repeat(rgb.transpose(2, 0, 1)[None], 16, axis=0)
    feat = sess.run(None, {inp_name: blob})[0][0]
    return feat.astype(np.float32) / (np.linalg.norm(feat) + 1e-6)

def extract_color_hist(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, w, _ = bgr.shape
    crop = hsv[int(h*0.5):, :]                # 仅下半身/衣服区域
    hist = cv2.calcHist([crop], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    hvec = hist.flatten().astype(np.float32)
    return hvec / (np.linalg.norm(hvec) + 1e-6)

# -------------------- 距离与 ID 分配 --------------------
def _combo_dist(f1, c1, f2, c2):
    d_reid = np.linalg.norm(f1 - f2)
    d_col  = cv2.compareHist(c1, c2, cv2.HISTCMP_BHATTACHARYYA)
    return (1 - _COLOR_WEIGHT) * d_reid + _COLOR_WEIGHT * d_col

def assign_id(feat: np.ndarray, col: np.ndarray) -> int:
    global next_sid
    if not id_bank:
        id_bank[next_sid] = [(feat, col)]
        next_sid += 1
        return next_sid - 1

    best_id, best_d = None, float("inf")
    for sid, lst in id_bank.items():
        d = min(_combo_dist(feat, col, f, c) for f, c in lst)
        if d < best_d:
            best_d, best_id = d, sid

    if best_id is not None and best_d < DIST_TH:
        id_bank[best_id].append((feat, col))
        if len(id_bank[best_id]) > MAX_FEATS:
            id_bank[best_id].pop(0)
        return best_id

    id_bank[next_sid] = [(feat, col)]
    next_sid += 1
    return next_sid - 1

# -------------------- 鼠标锁定 / 解锁 --------------------
def select_target_id(event, x, y, flags, cam_idx):
    global target_id
    if event == cv2.EVENT_LBUTTONDOWN:
        with bank_lock:
            target_id = None
            if cam_idx in current_detections_per_cam:
                for x1, y1, x2, y2, sid in current_detections_per_cam[cam_idx]:
                    if x1 < x < x2 and y1 < y < y2:
                        target_id = sid
                        print(f"[INFO] CAM {cam_idx} | Target locked on ID: {sid}")
                        break
    elif event == cv2.EVENT_RBUTTONDOWN:
        with bank_lock:
            if target_id is not None:
                print(f"[INFO] CAM {cam_idx} | Target unlocked.")
                target_id = None

# ---------------------------------------------------------------------------
# 摄像头线程
# ---------------------------------------------------------------------------
def process_camera(camera_index: int):
    global target_id, tracker_window_owner

    main_window_name = f"Camera {camera_index}"
    cv2.namedWindow(main_window_name)
    cv2.setMouseCallback(main_window_name, select_target_id, param=camera_index)

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {camera_index}.")
        return

    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    track_id_to_sid: dict[int, int] = {}

    while True:
        if cv2.getWindowProperty(main_window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        ret, frame = cap.read()
        if not ret:
            break

        # -------------------- 本帧已用 SID 集合 --------------------
        frame_used_sids = set()

        local_detections: list[tuple[int, int, int, int, int]] = []
        target_found_this_frame = False

        # -------------------- YOLO 检测 --------------------
        res = yolo.predict(frame, conf=0.25, classes=[0], verbose=False)[0]
        dets_to_sort = np.array(
            [b.xyxy.cpu().numpy()[0].tolist() + [b.conf.cpu().numpy()[0]]
             for b in res.boxes], dtype=np.float32)

        tracks = tracker.update(dets_to_sort) if dets_to_sort.size else tracker.update(np.empty((0, 5)))

        active_track_ids = {int(t[4]) for t in tracks}
        track_id_to_sid = {tid: sid for tid, sid in track_id_to_sid.items() if tid in active_track_ids}

        # -------------------- 遍历每个跟踪框 --------------------
        for tr in tracks:
            x1, y1, x2, y2, track_id = map(int, tr)
            adj_box = pan_and_scan_16_9((x1, y1, x2, y2), frame_w, frame_h, headroom_ratio=HEADROOM_RATIO)
            if adj_box is None:
                continue
            ax1, ay1, ax2, ay2 = adj_box

            sid = track_id_to_sid.get(track_id, -1)
            if sid == -1:
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    reid_f = extract_feat(crop)
                    col_h  = extract_color_hist(crop)
                    with bank_lock:
                        sid = assign_id(reid_f, col_h)
                track_id_to_sid[track_id] = sid

            # -------- 同帧去重：若已被占用，强制新号 --------
            if sid in frame_used_sids:
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    reid_f = extract_feat(crop)
                    col_h  = extract_color_hist(crop)
                    with bank_lock:
                        sid = assign_id(reid_f, col_h)
                    track_id_to_sid[track_id] = sid
            frame_used_sids.add(sid)

            local_detections.append((ax1, ay1, ax2, ay2, sid))

            # -------------------- 绘制 & Tracker View --------------------
            with bank_lock:
                is_target = (sid == target_id)

            if is_target:
                target_found_this_frame = True
                with bank_lock:
                    if tracker_window_owner in (None, camera_index):
                        tracker_window_owner = camera_index
                        update_tracker_view = True
                    else:
                        update_tracker_view = False

                if update_tracker_view:
                    target_crop = frame[ay1:ay2, ax1:ax2]
                    if target_crop.shape[0] > 0 and target_crop.shape[1] > 0:
                        letterboxed = create_letterboxed_image(target_crop, TRACKER_WINDOW_SIZE)
                        cv2.imshow(TRACKER_WINDOW_NAME, letterboxed)

            label_color = (0, 0, 255) if is_target else (0, 255, 0)
            cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), label_color, 2)
            cv2.putText(frame, f"ID:{sid}", (ax1, ay1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)

        # -------------------- 更新全局检测缓存 --------------------
        with bank_lock:
            current_detections_per_cam[camera_index] = local_detections
            if target_id is not None and not target_found_this_frame:
                if tracker_window_owner == camera_index:
                    tracker_window_owner = None
                    try:
                        cv2.destroyWindow(TRACKER_WINDOW_NAME)
                    except Exception:
                        pass

        cv2.imshow(main_window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(main_window_name)
    print(f"[INFO] Thread for Camera {camera_index} finished.")

# ---------------------------------------------------------------------------
# 主程序入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # ------- 读取旧 id_bank（如需保留）-------
    if os.path.exists(BANK_FILE) and not CLEAR_BANK_ON_EXIT:
        try:
            with open(BANK_FILE, "rb") as f:
                id_bank = pickle.load(f)
                if id_bank:
                    any_feat = next(iter(id_bank.values()))[0]
                    if not isinstance(any_feat, tuple):
                        print("[WARN] 旧格式 idbank 检测到，已忽略并重新开始。")
                        id_bank = {}
                        next_sid = 1
                    else:
                        next_sid = max(id_bank.keys()) + 1
                print(f"[INFO] Loaded {len(id_bank)} IDs from {BANK_FILE}")
        except Exception as e:
            print(f"[WARN] Failed to load {BANK_FILE}: {e}")
            id_bank, next_sid = {}, 1

    # ------- 启动线程 -------
    camera_indices = [0, 1]
    threads = []
    print(f"[INFO] Starting {len(camera_indices)} camera threads...")
    for idx in camera_indices:
        t = threading.Thread(target=process_camera, args=(idx,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    # ------- 退出处理 -------
    if CLEAR_BANK_ON_EXIT:
        if os.path.exists(BANK_FILE):
            os.remove(BANK_FILE)
    else:
        with open(BANK_FILE, "wb") as f:
            pickle.dump(id_bank, f)

    print("[INFO] Program terminated.")