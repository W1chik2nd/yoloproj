# & 'C:\Users\ASUS\yolovs\Scripts\Activate.ps1'
import cv2
import time
import os
import pickle
import numpy as np
from sort import Sort
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# ---------- 参数配置 ----------
DIST_TH       = 1.08            # 人脸 embedding 距离阈值
BANK_FILE     = "face_idbank.pkl"
TARGET_FPS    = 30              # 目标帧率
CLEAR_ON_EXIT = True            # 退出时清空/删除 id 库

# 模型加载
# YOLOv8-Face 用于人脸检测
yolo_face = YOLO("weights/yolov8n-face-lindevs.pt")

# InsightFace ArcFace 用于人脸 embedding
fa = FaceAnalysis(
    allowed_modules=['detection', 'recognition'],
    providers=['CUDAExecutionProvider']  # 无 GPU 用 ['CPUExecutionProvider']
)
fa.prepare(ctx_id=0, det_size=(160, 160))

# SORT 跟踪器，用于短时跟踪
tracker = Sort(max_age=5, min_hits=1, iou_threshold=0.3)

# 人脸 ID 库持久化加载
if os.path.exists(BANK_FILE):
    with open(BANK_FILE, "rb") as f:
        raw = pickle.load(f)
    id_bank = {
        int(fid): (vals if isinstance(vals, list) else [vals])
        for fid, vals in raw.items()
    }
    next_id = max(id_bank.keys()) + 1
    print(f"[INFO] Loaded {len(id_bank)} face IDs, next = {next_id}")
else:
    id_bank, next_id = {}, 1
    print("[INFO] Starting new face ID database")

MAX_FEATS_PER_ID = 5  # 每个 ID 最多保留 N 条特征

def assign_id(feat: np.ndarray) -> int:
    """
    “多特征相册”策略：
    与库中每个ID的所有特征进行比对，取最小距离。
    匹配成功则将新特征加入该ID的“相册”中，以增强其多样性。
    """
    global next_id
    
    # 如果ID库为空，直接创建新ID
    if not id_bank:
        id_bank[next_id] = [feat]
        new_id = next_id
        next_id += 1
        print(f"[INFO] ID Bank is empty. Creating new ID: {new_id}")
        return new_id
    
    best_id, best_d = None, float("inf")
    
    # 遍历相册中的每一个ID
    for fid, saved_feats_list in id_bank.items():
        # 计算新特征与该ID“相册”中所有照片的最小距离
        min_dist_to_this_id = min(np.linalg.norm(feat - saved_f) for saved_f in saved_feats_list)
        
        # 寻找全局最小的距离
        if min_dist_to_this_id < best_d:
            best_d, best_id = min_dist_to_this_id, fid
            
    print(f"[DEBUG] Best match is ID {best_id} with distance {best_d:.4f}. Threshold is {DIST_TH}")

    # 如果最小距离小于阈值，则认为是同一个人
    if best_id is not None and best_d < DIST_TH:
        print(f"[INFO] Match found! Assigning existing ID: {best_id}")
        # 将这张新“照片”加入对应ID的相册
        id_bank[best_id].append(feat)
        # 如果相册超过上限，则移除最早的那张照片 (FIFO)
        if len(id_bank[best_id]) > MAX_FEATS_PER_ID:
            id_bank[best_id].pop(0)
        return best_id
    
    # 如果最小距离仍然大于阈值，则确认为一个新人
    new_id = next_id
    print(f"[INFO] No suitable match found. Creating new ID: {new_id}")
    id_bank[new_id] = [feat] # 为新人创建他自己的第一张照片
    next_id += 1
    return new_id

# track_id -> face_id 映射
track2face = {}

# 打开摄像头 
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
interval, last_t = 1 / TARGET_FPS, 0.0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        if now - last_t < interval:
            continue
        last_t = now

        # —— 1. YOLO-Face 检测 —— 
        results = yolo_face.predict(frame,
                                    imgsz=320,
                                    conf=0.25,
                                    classes=[0],
                                    verbose=False)[0]

        # 准备 SORT 输入：[[x1,y1,x2,y2,conf], ...]
        dets = []
        for box in results.boxes:
            coords = box.xyxy.cpu().numpy()[0]
            x1, y1, x2, y2 = coords.astype(int)
            conf = float(box.conf.cpu().numpy()[0])
            dets.append([x1, y1, x2, y2, conf])
        # 转为NumPy并fix形状
        dets = np.array(dets, dtype=np.float32)
        if dets.size == 0:
            dets = np.zeros((0, 5), dtype=np.float32)
        elif dets.ndim == 1:
            dets = dets.reshape(1, 5)

        # 更新短时跟踪器
        tracks = tracker.update(dets)
        # 先清理失效的 track2face 条目
        active_tids = set(int(t[4]) for t in tracks)
        for tid in list(track2face):
            if tid not in active_tids:
                del track2face[tid]

        # 遍历每一路 track
        for *xyxy, track_id in tracks:
            x1, y1, x2, y2 = map(int, xyxy)
            # 人脸区域
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            # 只在新 track 上做一次 embedding+assign
            if track_id not in track2face:
                faces = fa.get(face)
                if not faces:
                    continue
                emb = faces[0].embedding
                emb /= (np.linalg.norm(emb) + 1e-6)
                face_id = assign_id(emb)
                track2face[track_id] = face_id
            else:
                face_id = track2face[track_id]

            # 绘制结果 
            label = f"ID:{face_id}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("YOLO-Face + ArcFace + SORT", frame)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

finally:
    cap.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass

    # 退出时持久化 or 清空数据库  
    if CLEAR_ON_EXIT:
        track2face.clear()
        id_bank.clear()
        if os.path.exists(BANK_FILE):
            os.remove(BANK_FILE)
        print("[INFO] Cleared face ID database on exit")
    else:
        with open(BANK_FILE, "wb") as f:
            pickle.dump(id_bank, f)
        print(f"[INFO] Saved {len(id_bank)} face IDs to {BANK_FILE}")
