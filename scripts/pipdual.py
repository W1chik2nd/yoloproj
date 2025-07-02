import os
import time
import pickle
import threading
import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
from sort import Sort

DIST_TH    = 0.91
BANK_FILE  = "idbank.pkl"
TARGET_FPS = 60
MAX_FEATS  = 10
CLEAR_BANK_ON_EXIT = True
TRACKER_WINDOW_NAME = "Tracker View"
TRACKER_WINDOW_SIZE = (800, 450) # 16:9
HEADROOM_RATIO = 0.20

print("[INFO] Loading models...")
yolo = YOLO("weights/yolo11s.pt")
try:
    providers = ["CUDAExecutionProvider"]; sess = ort.InferenceSession("weights/osnet_x1_0_msmt17.onnx", providers=providers); print("[INFO] Using GPU.")
except Exception:
    providers = ["CPUExecutionProvider"]; sess = ort.InferenceSession("weights/osnet_x0_25_msmt17.onnx", providers=providers); print("[WARN] Using CPU.")
inp_name = sess.get_inputs()[0].name
print("[INFO] Models loaded.")

id_bank = {}
next_sid = 1
target_id = None
current_detections_per_cam = {}
tracker_window_owner = None
bank_lock = threading.Lock()

def pan_and_scan_16_9(box, frame_w, frame_h, headroom_ratio=0.20):
    x1, y1, x2, y2 = box; box_w, box_h = x2 - x1, y2 - y1
    if box_w <= 0 or box_h <= 0: return None
    center_x, center_y = x1 + box_w / 2, y1 + box_h / 2
    aspect_ratio_16_9 = 16 / 9
    if (box_w / box_h) > aspect_ratio_16_9: new_w, new_h = box_w, int(box_w / aspect_ratio_16_9)
    else: new_h, new_w = box_h, int(box_h * aspect_ratio_16_9)
    vertical_offset = int(new_h * headroom_ratio); adjusted_center_y = center_y - (vertical_offset / 2)
    ideal_x1, ideal_y1 = int(center_x - new_w / 2), int(adjusted_center_y - new_h / 2)
    ideal_x2, ideal_y2 = int(center_x + new_w / 2), int(adjusted_center_y + new_h / 2)
    dx = 0;
    if ideal_x1 < 0: dx = -ideal_x1
    if ideal_x2 >= frame_w: dx = frame_w - 1 - ideal_x2
    dy = 0;
    if ideal_y1 < 0: dy = -ideal_y1
    if ideal_y2 >= frame_h: dy = frame_h - 1 - ideal_y2
    final_x1, final_y1, final_x2, final_y2 = ideal_x1 + dx, ideal_y1 + dy, ideal_x2 + dx, ideal_y2 + dy
    return max(0, final_x1), max(0, final_y1), min(frame_w - 1, final_x2), min(frame_h - 1, final_y2)

def create_letterboxed_image(image, target_size):
    target_w, target_h = target_size; h, w, _ = image.shape
    if w == 0 or h == 0: return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    scale = min(target_w / w, target_h / h); new_w, new_h = int(w * scale), int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h)); canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_offset, y_offset = (target_w - new_w) // 2, (target_h - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image
    return canvas
def extract_feat(bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(cv2.resize(bgr, (128, 256)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0; rgb -= (0.485, 0.456, 0.406); rgb /= (0.229, 0.224, 0.225)
    blob = np.repeat(rgb.transpose(2, 0, 1)[None], 16, axis=0); feat = sess.run(None, {inp_name: blob})[0][0]; feat /= (np.linalg.norm(feat) + 1e-6)
    return feat.astype(np.float32)
def assign_id(feat: np.ndarray) -> int:
    global next_sid;
    if not id_bank: id_bank[next_sid] = [feat]; new_id = next_sid; next_sid += 1; return new_id
    best_id, best_d = None, float("inf")
    for sid, feats in id_bank.items():
        valid = [f for f in feats if isinstance(f, np.ndarray) and f.ndim == 1];
        if not valid: continue
        d = min(np.linalg.norm(feat - f) for f in valid)
        if d < best_d: best_d, best_id = d, sid
    if best_id is not None and best_d < DIST_TH:
        id_bank[best_id].append(feat)
        if len(id_bank[best_id]) > MAX_FEATS: id_bank[best_id].pop(0)
        return best_id
    new_id = next_sid; id_bank[new_id] = [feat]; next_sid += 1; return new_id

def select_target_id(event, x, y, flags, cam_idx):
    global target_id
    if event == cv2.EVENT_LBUTTONDOWN:
        with bank_lock:
            target_id = None
            if cam_idx in current_detections_per_cam:
                for det in current_detections_per_cam[cam_idx]:
                    x1, y1, x2, y2, sid = det
                    if x1 < x < x2 and y1 < y < y2:
                        target_id = sid; print(f"[INFO] CAM {cam_idx} | Target locked on ID: {sid}"); break
    elif event == cv2.EVENT_RBUTTONDOWN:
        with bank_lock:
            if target_id is not None:
                print(f"[INFO] CAM {cam_idx} | Target unlocked.")
                target_id = None

def process_camera(camera_index: int):
    global target_id, tracker_window_owner
    
    main_window_name = f"Camera {camera_index}"
    cv2.namedWindow(main_window_name); cv2.setMouseCallback(main_window_name, select_target_id, param=camera_index)
    
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened(): print(f"[ERROR] Cannot open camera {camera_index}."); return
    
    frame_h, frame_w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    track_id_to_sid = {}

    while True:
        if cv2.getWindowProperty(main_window_name, cv2.WND_PROP_VISIBLE) < 1: break # 窗口被关闭则退出
        
        ret, frame = cap.read()
        if not ret: break
        
        local_detections = []
        target_found_this_frame = False

        res = yolo.predict(frame, conf=0.25, classes=[0], verbose=False)[0]
        dets_to_sort = np.array([box.xyxy.cpu().numpy()[0].tolist() + [box.conf.cpu().numpy()[0]] for box in res.boxes], dtype=np.float32)
        tracks = tracker.update(dets_to_sort) if dets_to_sort.size > 0 else tracker.update(np.empty((0, 5)))
        
        active_track_ids = {int(t[4]) for t in tracks}
        for tid in list(track_id_to_sid.keys()):
            if tid not in active_track_ids: del track_id_to_sid[tid]

        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            adj_box = pan_and_scan_16_9((x1, y1, x2, y2), frame_w, frame_h, headroom_ratio=HEADROOM_RATIO)
            if adj_box is None: continue
            adj_x1, adj_y1, adj_x2, adj_y2 = adj_box
            
            sid = track_id_to_sid.get(track_id, -1)
            if sid == -1:
                with bank_lock:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0: sid = assign_id(extract_feat(crop)); track_id_to_sid[track_id] = sid
            if sid == -1: continue

            local_detections.append((adj_x1, adj_y1, adj_x2, adj_y2, sid))
            
            with bank_lock:
                is_target = (sid == target_id)
            
            if is_target:
                target_found_this_frame = True
                with bank_lock:
                    if tracker_window_owner is None or tracker_window_owner == camera_index:
                        tracker_window_owner = camera_index
                        should_update_tracker_view = True
                    else:
                        should_update_tracker_view = False
                
                if should_update_tracker_view:
                    target_crop_16_9 = frame[adj_y1:adj_y2, adj_x1:adj_x2]
                    if target_crop_16_9.shape[0] > 0 and target_crop_16_9.shape[1] > 0:
                        letterboxed_crop = create_letterboxed_image(target_crop_16_9, TRACKER_WINDOW_SIZE)
                        cv2.imshow(TRACKER_WINDOW_NAME, letterboxed_crop)
            
            label_color = (0, 0, 255) if is_target else (0, 255, 0)
            cv2.rectangle(frame, (adj_x1, adj_y1), (adj_x2, adj_y2), label_color, 2)
            cv2.putText(frame, f"ID:{sid}", (adj_x1, adj_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)
                
        with bank_lock:
            current_detections_per_cam[camera_index] = local_detections
            if target_id is not None and not target_found_this_frame:
                if tracker_window_owner == camera_index:
                    tracker_window_owner = None
                    try: cv2.destroyWindow(TRACKER_WINDOW_NAME)
                    except: pass
            
        cv2.imshow(main_window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release(); cv2.destroyWindow(main_window_name); print(f"[INFO] Thread for Camera {camera_index} finished.")

if __name__ == "__main__":
    if os.path.exists(BANK_FILE) and not CLEAR_BANK_ON_EXIT:
        try:
            with open(BANK_FILE, "rb") as f: id_bank = pickle.load(f); next_sid = max(id_bank.keys()) + 1 if id_bank else 1
            print(f"[INFO] Loaded {len(id_bank)} face IDs from {BANK_FILE}")
        except: id_bank, next_sid = {}, 1
    
    camera_indices = [0, 1]
    threads = []
    print(f"[INFO] Starting {len(camera_indices)} camera threads...")
    for index in camera_indices:
        thread = threading.Thread(target=process_camera, args=(index,)); threads.append(thread); thread.start()
    for thread in threads: thread.join()
    print("[INFO] All threads have been closed.")
    
    if CLEAR_BANK_ON_EXIT:
        if os.path.exists(BANK_FILE): os.remove(BANK_FILE)
    else:
        with open(BANK_FILE, "wb") as f: pickle.dump(id_bank, f)
    print("[INFO] Program terminated.")