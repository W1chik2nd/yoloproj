# & 'C:\Users\ASUS\yolovs\Scripts\Activate.ps1'
import os
import time
import pickle
import threading  # 引入多线程库

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

DIST_TH    = 0.90          # 特征距离阈值
BANK_FILE  = "idbank.pkl"  # 特征库文件
TARGET_FPS = 30            # 目标帧率
MAX_FEATS  = 10            # 每个 ID 最多保留 N 条特征
CLEAR_BANK_ON_EXIT = True  # 设置为 True, 退出时删除ID库; 设置为 False, 退出时保存

# 模型加载 (全局加载，所有线程共享)
print("[INFO] Loading models...")
yolo = YOLO("weights/yolo11s.pt")
try:
    providers = ["CUDAExecutionProvider"]
    sess = ort.InferenceSession("weights/osnet_x1_0_msmt17.onnx", providers=providers)
    print("[INFO] Using GPU (CUDAExecutionProvider).")
except Exception:
    providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession("weights/osnet_x1_0_msmt17.onnx", providers=providers)
    print("[WARN] CUDA not available. Using CPU with a smaller OSNet model.")
inp_name = sess.get_inputs()[0].name
print("[INFO] Models loaded.")

# 共享ID库和线程锁
id_bank = {}
next_sid = 1
bank_lock = threading.Lock() # 用于保护 id_bank 和 next_sid 的访问

# 特征提取函数 (全局)
def extract_feat(bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(cv2.resize(bgr, (128, 256)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb -= (0.485, 0.456, 0.406)
    rgb /= (0.229, 0.224, 0.225)
    blob = np.repeat(rgb.transpose(2, 0, 1)[None], 16, axis=0)
    feat = sess.run(None, {inp_name: blob})[0][0]
    feat /= (np.linalg.norm(feat) + 1e-6)
    return feat.astype(np.float32)

# ID分配函数
def assign_id(feat: np.ndarray) -> int:
    global next_sid
    if not id_bank:
        id_bank[next_sid] = [feat]
        new_id = next_sid; next_sid += 1
        print(f"[INFO] CAM ALL | ID Bank empty. Creating new ID: {new_id}")
        return new_id
    best_id, best_d = None, float("inf")
    for sid, feats in id_bank.items():
        valid = [f for f in feats if isinstance(f, np.ndarray) and f.ndim == 1]
        if not valid: continue
        d = min(np.linalg.norm(feat - f) for f in valid)
        if d < best_d:
            best_d, best_id = d, sid
    if best_id is not None and best_d < DIST_TH:
        id_bank[best_id].append(feat)
        if len(id_bank[best_id]) > MAX_FEATS: id_bank[best_id].pop(0)
        return best_id
    new_id = next_sid
    id_bank[new_id] = [feat]
    next_sid += 1
    print(f"[INFO] CAM ALL | No match found. Creating new ID: {new_id}")
    return new_id

# 摄像头处理函数 (每个线程的核心)
def process_camera(camera_index: int):
    print(f"[INFO] Thread for Camera {camera_index} started.")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {camera_index}.")
        return

    interval, last_t = 1.0 / TARGET_FPS, 0.0
    while True:
        ret, frame = cap.read()
        if not ret: break

        now = time.time()
        if now - last_t < interval:
            time.sleep(max(0, interval - (now - last_t)))
            continue
        last_t = now

        res = yolo.predict(frame[..., ::-1], conf=0.25, classes=[0], verbose=False)[0]
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0: continue

            feature = extract_feat(crop)
            
            # 关键：在访问共享ID库以分配ID前，必须加锁
            with bank_lock:
                assigned_id = assign_id(feature)

            label = f"CAM {camera_index} | ID: {assigned_id}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 128, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow(f"Person Re-ID Camera {camera_index}", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
            
    cap.release()
    cv2.destroyWindow(f"Person Re-ID Camera {camera_index}")
    print(f"[INFO] Thread for Camera {camera_index} finished.")

# 主程序入口
if __name__ == "__main__":
    if os.path.exists(BANK_FILE) and not CLEAR_BANK_ON_EXIT:
        try:
            with open(BANK_FILE, "rb") as f:
                id_bank = pickle.load(f)
            next_sid = max(id_bank.keys()) + 1 if id_bank else 1
            print(f"[INFO] Loaded {len(id_bank)} IDs from {BANK_FILE}")
        except Exception as e:
            print(f"[WARN] Failed to load ID bank: {e}")
            id_bank = {}
            next_sid = 1
    
    # 定义要启动的摄像头列表
    camera_indices = [0, 1] # 假定您有两个摄像头，编号为0和1
    threads = []

    print(f"[INFO] Starting {len(camera_indices)} camera threads...")
    for index in camera_indices:
        thread = threading.Thread(target=process_camera, args=(index,))
        threads.append(thread)
        thread.start()

    # 等待所有摄像头线程结束
    for thread in threads:
        thread.join()

    print("[INFO] All threads have been closed.")
    
    # 在所有线程结束后，统一处理ID库的保存/删除
    if CLEAR_BANK_ON_EXIT:
        if os.path.exists(BANK_FILE):
            os.remove(BANK_FILE)
        print(f"[INFO] Cleared and removed ID bank file: {BANK_FILE}")
    else:
        with open(BANK_FILE, "wb") as f:
            pickle.dump(id_bank, f)
        print(f"[INFO] Saved {len(id_bank)} IDs to {BANK_FILE}")

    print("[INFO] Program terminated.")