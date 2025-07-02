import os
import time
import pickle
import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
from sort import Sort

DIST_TH    = 0.80
BANK_FILE  = "idbank.pkl"
TARGET_FPS = 60
MAX_FEATS  = 10
CLEAR_BANK_ON_EXIT = True
TRACKER_WINDOW_NAME = "Tracker View"
TRACKER_WINDOW_SIZE = (800, 450)
HEADROOM_RATIO = 0.15
TRACKER_GRACE_PERIOD = 1.5

target_id = None
current_detections = []
target_last_seen_time = 0
last_good_box = {}

print("[INFO] Loading models and tracker...")
yolo = YOLO("weights/yolo11s.pt")
try:
    providers = ["CUDAExecutionProvider"]; sess = ort.InferenceSession("weights/osnet_x1_0_msmt17.onnx", providers=providers); print("[INFO] Using GPU.")
except Exception:
    providers = ["CPUExecutionProvider"]; sess = ort.InferenceSession("weights/osnet_x0_25_msmt17.onnx", providers=providers); print("[WARN] Using CPU.")
inp_name = sess.get_inputs()[0].name
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3); track_id_to_sid = {}; print("[INFO] Models loaded.")

def create_smart_16_9_box(box, frame_w, frame_h, headroom_ratio=0.15):
    x1, y1, x2, y2 = box
    box_w, box_h = x2 - x1, y2 - y1
    if box_w <= 0 or box_h <= 0: return None, False
    center_x, center_y = x1 + box_w / 2, y1 + box_h / 2
    aspect_ratio_16_9 = 16 / 9

    if (box_w / box_h) > aspect_ratio_16_9:
        exp_w, exp_h = box_w, int(box_w / aspect_ratio_16_9)
    else:
        exp_h, exp_w = box_h, int(box_h * aspect_ratio_16_9)
    
    if exp_w > frame_w or exp_h > frame_h:
        if (frame_w / frame_h) > aspect_ratio_16_9: new_h, new_w = frame_h, int(frame_h * aspect_ratio_16_9)
        else: new_w, new_h = frame_w, int(frame_w / aspect_ratio_16_9)
        final_box = int(center_x-new_w/2), int(center_y-new_h/2), int(center_x+new_w/2), int(center_y+new_h/2)
        return final_box, True
    else:
        vertical_offset = int(exp_h * headroom_ratio); adjusted_center_y = center_y - (vertical_offset / 2)
        ideal_x1, ideal_y1 = int(center_x - exp_w / 2), int(adjusted_center_y - exp_h / 2)
        ideal_x2, ideal_y2 = int(center_x + exp_w / 2), int(adjusted_center_y + exp_h / 2)
        dx = 0;
        if ideal_x1 < 0: dx = -ideal_x1
        if ideal_x2 >= frame_w: dx = frame_w - 1 - ideal_x2
        dy = 0;
        if ideal_y1 < 0: dy = -ideal_y1
        if ideal_y2 >= frame_h: dy = frame_h - 1 - ideal_y2
        final_box = max(0, ideal_x1 + dx), max(0, ideal_y1 + dy), min(frame_w - 1, ideal_x2 + dx), min(frame_h - 1, ideal_y2 + dy)
        is_clamped = (dx != 0 or dy != 0)
        return final_box, is_clamped

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
if os.path.exists(BANK_FILE) and not CLEAR_BANK_ON_EXIT:
    try:
        with open(BANK_FILE, "rb") as f: id_bank = pickle.load(f); next_sid = max(id_bank.keys()) + 1 if id_bank else 1
    except: id_bank, next_sid = {}, 1
else: id_bank, next_sid = {}, 1
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
def select_target_id(event, x, y, flags, param):
    global target_id, target_last_seen_time, last_good_box
    if event == cv2.EVENT_LBUTTONDOWN:
        target_id = None
        if target_id in last_good_box: del last_good_box[target_id]
        for det in current_detections:
            x1, y1, x2, y2, sid = det
            if x1 < x < x2 and y1 < y < y2: target_id = sid; print(f"[INFO] Target locked on ID: {sid}"); target_last_seen_time = time.time(); break
    elif event == cv2.EVENT_RBUTTONDOWN:
        print("[INFO] Target unlocked."); target_id = None;
        if target_id in last_good_box: del last_good_box[target_id]
        try: cv2.destroyWindow(TRACKER_WINDOW_NAME)
        except: pass

MAIN_WINDOW_NAME = "YOLO Tracking"
cv2.namedWindow(MAIN_WINDOW_NAME); cv2.setMouseCallback(MAIN_WINDOW_NAME, select_target_id)
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW); interval = 1.0 / TARGET_FPS; last_t = 0.0
frame_h, frame_w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

try:
    while True:
        ret, frame = cap.read();
        if not ret: break
        now = time.time();
        if now - last_t < interval: time.sleep(max(0, interval-(now-last_t))); continue
        last_t = now
        current_detections.clear(); target_found_this_frame = False
        
        res = yolo.predict(frame, conf=0.25, classes=[0], verbose=False)[0]
        dets_to_sort = np.array([box.xyxy.cpu().numpy()[0].tolist() + [box.conf.cpu().numpy()[0]] for box in res.boxes], dtype=np.float32)
        tracks = tracker.update(dets_to_sort) if dets_to_sort.size > 0 else tracker.update(np.empty((0, 5)))
        active_track_ids = {int(t[4]) for t in tracks};
        for tid in list(track_id_to_sid.keys()):
            if tid not in active_track_ids: del track_id_to_sid[tid]
        
        final_boxes_to_draw = []

        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            
            sid = track_id_to_sid.get(track_id, -1)
            if sid == -1:
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0: sid = assign_id(extract_feat(crop)); track_id_to_sid[track_id] = sid
            if sid == -1: continue
            display_box = None
            if sid == target_id:
                target_found_this_frame = True
                target_last_seen_time = time.time()
                
                candidate_box, is_clamped = create_smart_16_9_box((x1, y1, x2, y2), frame_w, frame_h, headroom_ratio=HEADROOM_RATIO)
                
                if not is_clamped:
                    last_good_box[target_id] = candidate_box
                    display_box = candidate_box
                else:
                    display_box = last_good_box.get(target_id, candidate_box)

                if display_box:
                    dx1, dy1, dx2, dy2 = display_box
                    target_crop = frame[dy1:dy2, dx1:dx2]
                    if target_crop.size > 0:
                        cv2.imshow(TRACKER_WINDOW_NAME, create_letterboxed_image(target_crop, TRACKER_WINDOW_SIZE))
            else:
                display_box = create_smart_16_9_box((x1, y1, x2, y2), frame_w, frame_h, headroom_ratio=HEADROOM_RATIO)[0]

            if display_box:
                final_boxes_to_draw.append((display_box, sid))

        for box_to_draw, sid in final_boxes_to_draw:
            dx1, dy1, dx2, dy2 = box_to_draw
            current_detections.append((dx1, dy1, dx2, dy2, sid))
            label_color = (0, 0, 255) if sid == target_id else (0, 255, 0)
            cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), label_color, 2)
            cv2.putText(frame, f"ID:{sid}", (dx1, dy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)

        if target_id is not None and not target_found_this_frame:
            time_since_lost = time.time() - target_last_seen_time
            if time_since_lost > TRACKER_GRACE_PERIOD:
                try: 
                    cv2.destroyWindow(TRACKER_WINDOW_NAME)
                except: 
                    pass
            else:
                last_box = last_good_box.get(target_id)
                if last_box:
                    lx1, ly1, lx2, ly2 = last_box
                    live_crop = frame[ly1:ly2, lx1:lx2]
                    if live_crop.size > 0:
                        lost_view = create_letterboxed_image(live_crop, TRACKER_WINDOW_SIZE)
                        cv2.putText(lost_view, "Re-acquiring...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                        cv2.imshow(TRACKER_WINDOW_NAME, lost_view)

        cv2.imshow(MAIN_WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27): break
finally:
    cap.release(); cv2.destroyAllWindows()
    if CLEAR_BANK_ON_EXIT:
        if os.path.exists(BANK_FILE): os.remove(BANK_FILE)
    else:
        with open(BANK_FILE, "wb") as f: pickle.dump(id_bank, f)