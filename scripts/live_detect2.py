import cv2
import time
import os
import pickle
import numpy as np
import threading  # 引入多线程库
from sort import Sort
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# ---------- 1. 全局共享资源定义 ----------
DIST_TH = 1.08             # 人脸 embedding 距离阈值 (根据您的测试进行微调)
BANK_FILE = "face_idbank.pkl"
TARGET_FPS = 30            # 目标帧率
CLEAR_ON_EXIT = True       # 退出时清空/删除 id 库
MAX_FEATS_PER_ID = 5       # 每个ID最多保留N条特征

# --- 模型加载 (这些模型实例将被所有线程共享) ---
print("[INFO] Loading models...")
yolo_face = YOLO("weights/yolov8n-face-lindevs.pt")
fa = FaceAnalysis(
    allowed_modules=['detection', 'recognition'],
    providers=['CPUExecutionProvider']
)
fa.prepare(ctx_id=0, det_size=(160, 160))
print("[INFO] Models loaded.")

# --- 共享ID库和线程锁 ---
id_bank = {}
next_id = 1
bank_lock = threading.Lock() # 关键：用于保护 id_bank 和 next_id 的访问

# ID分配函数 (与之前版本相同，但现在操作的是全局变量)
def assign_id(feat: np.ndarray) -> int:
    global next_id # 声明操作的是全局变量
    
    # 注意：这个函数的所有调用都必须被 "with bank_lock:" 包裹
    
    if not id_bank:
        id_bank[next_id] = [feat]
        new_id = next_id
        next_id += 1
        return new_id
    
    best_id, best_d = None, float("inf")
    for fid, saved_feats_list in id_bank.items():
        min_dist_to_this_id = min(np.linalg.norm(feat - saved_f) for saved_f in saved_feats_list)
        if min_dist_to_this_id < best_d:
            best_d, best_id = min_dist_to_this_id, fid
            
    if best_id is not None and best_d < DIST_TH:
        id_bank[best_id].append(feat)
        if len(id_bank[best_id]) > MAX_FEATS_PER_ID:
            id_bank[best_id].pop(0)
        return best_id
    
    new_id = next_id
    id_bank[new_id] = [feat]
    next_id += 1
    return new_id

# ---------- 2. 摄像头处理核心函数 ----------
def process_camera(camera_index: int):
    """
    为单个摄像头处理视频流、检测、跟踪和识别的核心函数。
    每个线程运行此函数的一个实例。
    """
    print(f"[INFO] Thread for Camera {camera_index} started.")
    
    # 每个线程拥有自己独立的短时跟踪器和ID映射
    tracker = Sort(max_age=5, min_hits=1, iou_threshold=0.3)
    track2face = {}

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {camera_index}.")
        return

    interval, last_t = 1 / TARGET_FPS, 0.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[INFO] Camera {camera_index} stream ended.")
            break

        now = time.time()
        if now - last_t < interval:
            time.sleep(interval - (now - last_t)) # 稍微等待，避免CPU空转
            continue
        last_t = now
        
        # YOLO 检测
        results = yolo_face.predict(frame, imgsz=320, conf=0.25, classes=[0], verbose=False)[0]
        
        # 准备 SORT 输入
        dets = np.array([box.xyxy.cpu().numpy()[0].tolist() + [box.conf.cpu().numpy()[0]] for box in results.boxes], dtype=np.float32)
        if dets.size == 0:
            dets = np.zeros((0, 5), dtype=np.float32)

        # 更新短时跟踪器
        tracks = tracker.update(dets)
        
        # 清理失效的 track2face 条目
        active_tids = {int(t[4]) for t in tracks}
        for tid in list(track2face.keys()):
            if tid not in active_tids:
                del track2face[tid]

        # 遍历每一路 track
        for *xyxy, track_id in tracks:
            track_id = int(track_id)
            x1, y1, x2, y2 = map(int, xyxy)
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                continue

            face_id = -1 # 默认值
            if track_id not in track2face:
                faces = fa.get(face_crop)
                if faces:
                    emb = faces[0].embedding
                    emb /= (np.linalg.norm(emb) + 1e-6)
                    
                    # 关键：在访问共享ID库之前加锁！
                    with bank_lock:
                        face_id = assign_id(emb)
                    
                    track2face[track_id] = face_id
            else:
                face_id = track2face[track_id]

            # 绘制结果
            if face_id != -1:
                label = f"CAM {camera_index} | ID: {face_id}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow(f"Camera {camera_index}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(f"Camera {camera_index}")
    print(f"[INFO] Thread for Camera {camera_index} finished.")

# ---------- 3. 主程序入口 ----------
if __name__ == "__main__":
    # 启动前加载已有的ID库
    if os.path.exists(BANK_FILE) and not CLEAR_ON_EXIT:
        with open(BANK_FILE, "rb") as f:
            id_bank = pickle.load(f)
        next_id = max(id_bank.keys()) + 1 if id_bank else 1
        print(f"[INFO] Loaded {len(id_bank)} face IDs from {BANK_FILE}")
    
    # 定义要启动的摄像头列表
    camera_indices = [0, 1] # 假定您有两个摄像头，编号为0和1
    threads = []
    
    # 为每个摄像头创建一个线程
    for index in camera_indices:
        thread = threading.Thread(target=process_camera, args=(index,))
        threads.append(thread)
        thread.start()
        
    # 等待所有线程执行完毕 (即所有摄像头窗口被关闭)
    for thread in threads:
        thread.join()
        
    # 在所有线程结束后，统一处理数据库的保存或清理事宜
    if CLEAR_ON_EXIT:
        if os.path.exists(BANK_FILE):
            os.remove(BANK_FILE)
        print("[INFO] Cleared face ID database on exit.")
    else:
        with open(BANK_FILE, "wb") as f:
            pickle.dump(id_bank, f)
        print(f"[INFO] Saved {len(id_bank)} face IDs to {BANK_FILE}")

    print("[INFO] All camera threads have been closed. Program terminated.")