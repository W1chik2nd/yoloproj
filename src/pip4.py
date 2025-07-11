# & 'C:\Users\ASUS\yolovs\Scripts\Activate.ps1'
import os
import time
import pickle
import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
from sort import Sort

DIST_TH    = 0.85
BANK_FILE  = "idbank.pkl"
TARGET_FPS = 60
MAX_FEATS  = 10
CLEAR_BANK_ON_EXIT = True
TRACKER_WINDOW_NAME = "Tracker View"
TRACKER_WINDOW_SIZE = (800, 450)
HEADROOM_RATIO = 0.20
TRACKER_GRACE_PERIOD = 1.5
JITTER_THRESHOLD = 1.5
FILTER_MIN_CUTOFF = 0.4
FILTER_BETA = 1.0

target_id = None
current_detections = []
target_last_seen_time = 0
last_good_box = {}
stabilized_boxes = {}
filter_sets = {}

def alpha(cutoff, freq):
    te = 1.0 / freq; tau = 1.0 / (2 * np.pi * cutoff); return 1.0 / (1.0 + tau / te)
class OneEuroFilter:
    def __init__(self, freq, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.freq, self.min_cutoff, self.beta, self.d_cutoff = freq, min_cutoff, beta, d_cutoff
        self.x_prev, self.dx_prev, self.t_prev = None, None, None
    def __call__(self, x, t=None):
        if t is None: t = time.time()
        if self.t_prev is None: self.t_prev = t
        t_e = t - self.t_prev
        if t_e < 1e-6: return self.x_prev if self.x_prev is not None else x
        freq = 1.0 / t_e
        if self.x_prev is None: self.x_prev, self.dx_prev = x, 0.0; self.t_prev = t; return x
        dx = (x - self.x_prev) / t_e; a_d = alpha(self.d_cutoff, freq)
        if self.dx_prev is None: self.dx_prev = dx
        self.dx_prev = (1.0 - a_d) * self.dx_prev + a_d * dx
        cutoff = self.min_cutoff + self.beta * abs(self.dx_prev); a = alpha(cutoff, freq)
        x_filtered = (1.0 - a) * self.x_prev + a * x
        self.x_prev, self.t_prev = x_filtered, t
        return x_filtered

print("[INFO] Loading models and tracker...")
yolo = YOLO("weights/yolo11s.pt")
try:
    providers = ["CUDAExecutionProvider"]; sess = ort.InferenceSession("weights/osnet_x1_0_msmt17.onnx", providers=providers); print("[INFO] Using GPU.")
except Exception:
    providers = ["CPUExecutionProvider"]; sess = ort.InferenceSession("weights/osnet_x0_25_msmt17.onnx", providers=providers); print("[WARN] Using CPU.")
inp_name = sess.get_inputs()[0].name
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3); track_id_to_sid = {}; print("[INFO] Models loaded.")

def get_hybrid_stabilized_box(sid, new_box, now):
    if sid not in filter_sets:
        freq = TARGET_FPS
        filter_sets[sid] = {
            'cx': OneEuroFilter(freq, FILTER_MIN_CUTOFF, FILTER_BETA), 'cy': OneEuroFilter(freq, FILTER_MIN_CUTOFF, FILTER_BETA),
            'w': OneEuroFilter(freq, FILTER_MIN_CUTOFF, FILTER_BETA), 'h': OneEuroFilter(freq, FILTER_MIN_CUTOFF, FILTER_BETA)
        }
        stabilized_boxes[sid] = new_box; return new_box
    last_box = stabilized_boxes.get(sid, new_box)
    last_cx, last_cy = (last_box[0] + last_box[2]) / 2, (last_box[1] + last_box[3]) / 2
    new_cx, new_cy = (new_box[0] + new_box[2]) / 2, (new_box[1] + new_box[3]) / 2
    dist = np.sqrt((last_cx - new_cx)**2 + (last_cy - new_cy)**2)
    if dist < JITTER_THRESHOLD: return last_box
    else:
        box_w, box_h = new_box[2] - new_box[0], new_box[3] - new_box[1]
        smooth_cx = filter_sets[sid]['cx'](new_cx, now); smooth_cy = filter_sets[sid]['cy'](new_cy, now)
        smooth_w = filter_sets[sid]['w'](box_w, now); smooth_h = filter_sets[sid]['h'](box_h, now)
        sx1, sy1 = int(smooth_cx - smooth_w/2), int(smooth_cy - smooth_h/2)
        sx2, sy2 = int(smooth_cx + smooth_w/2), int(smooth_cy + smooth_h/2)
        stabilized_box = (sx1, sy1, sx2, sy2); stabilized_boxes[sid] = stabilized_box
        return stabilized_box
def pan_and_scan_16_9(box, frame_w, frame_h, headroom_ratio=0.15):
    x1, y1, x2, y2 = box; box_w, box_h = x2 - x1, y2 - y1
    if box_w <= 0 or box_h <= 0: return None
    center_x, center_y = x1 + box_w / 2, y1 + box_h / 2; aspect_ratio_16_9 = 16 / 9
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
def extract_feat(bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(cv2.resize(bgr, (128, 256)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0; rgb -= (0.485, 0.456, 0.406); rgb /= (0.229, 0.224, 0.225)
    blob = np.repeat(rgb.transpose(2, 0, 1)[None], 16, axis=0); feat = sess.run(None, {inp_name: blob})[0][0]; feat /= (np.linalg.norm(feat) + 1e-6)
    return feat.astype(np.float32)

if os.path.exists(BANK_FILE) and not CLEAR_BANK_ON_EXIT:
    try:
        with open(BANK_FILE, "rb") as f: id_bank = pickle.load(f); next_sid = max(id_bank.keys()) + 1 if id_bank else 1
    except: id_bank, next_sid = {}, 1
else: id_bank, next_sid = {}, 1

def assign_id(feat: np.ndarray, sids_to_exclude=set()) -> int:
    global next_sid;
    if not id_bank:
        id_bank[next_sid] = [feat]; new_id = next_sid; next_sid += 1; return new_id
    
    best_id, best_d = None, float("inf")
    for sid, feats in id_bank.items():
        if sid in sids_to_exclude:
            continue
            
        valid = [f for f in feats if isinstance(f, np.ndarray) and f.ndim == 1];
        if not valid: continue
        d = min(np.linalg.norm(feat - f) for f in valid)
        if d < best_d: best_d, best_id = d, sid
        
    if best_id is not None and best_d < DIST_TH:
        id_bank[best_id].append(feat)
        if len(id_bank[best_id]) > MAX_FEATS: id_bank[best_id].pop(0)
        return best_id
    
    new_id = next_sid;
    while new_id in id_bank: new_id += 1
    next_sid = new_id + 1
    id_bank[new_id] = [feat];
    return new_id

def select_target_id(event, x, y, flags, param):
    global target_id, target_last_seen_time, last_good_box, filter_sets
    if event == cv2.EVENT_LBUTTONDOWN:
        if target_id is not None:
            if target_id in last_good_box: del last_good_box[target_id]
            if target_id in filter_sets: del filter_sets[target_id]
        target_id = None
        for det in current_detections:
            x1, y1, x2, y2, sid = det
            if x1 < x < x2 and y1 < y < y2:
                target_id = sid; print(f"[INFO] Target locked on ID: {sid}"); target_last_seen_time = time.time(); break
    elif event == cv2.EVENT_RBUTTONDOWN:
        print("[INFO] Target unlocked.")
        if target_id is not None:
            if target_id in last_good_box: del last_good_box[target_id]
            if target_id in filter_sets: del filter_sets[target_id]
        target_id = None
        try: cv2.destroyWindow(TRACKER_WINDOW_NAME)
        except: pass

MAIN_WINDOW_NAME = "YOLO Tracking"
cv2.namedWindow(MAIN_WINDOW_NAME); cv2.setMouseCallback(MAIN_WINDOW_NAME, select_target_id)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW); interval = 1.0 / TARGET_FPS; last_t = 0.0
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
        active_sids = set()
        for tid in list(track_id_to_sid.keys()):
            if tid not in active_track_ids: del track_id_to_sid[tid]
        sids_in_current_frame = set()
        
        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            sid = track_id_to_sid.get(track_id, -1)
            
            if sid == -1:
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    feature = extract_feat(crop)
                    sid = assign_id(feature, sids_to_exclude=sids_in_current_frame)
                    track_id_to_sid[track_id] = sid
            
            if sid == -1: continue
            
            sids_in_current_frame.add(sid)
            active_sids.add(sid)
            
            stabilized_box = get_hybrid_stabilized_box(sid, (x1, y1, x2, y2), now)
            adj_box = pan_and_scan_16_9(stabilized_box, frame_w, frame_h, headroom_ratio=HEADROOM_RATIO)
            if adj_box is None: continue
            
            adj_x1, adj_y1, adj_x2, adj_y2 = adj_box
            current_detections.append((adj_x1, adj_y1, adj_x2, adj_y2, sid))

            if sid == target_id:
                target_found_this_frame = True; target_last_seen_time = time.time()
                last_good_box[target_id] = adj_box
                target_crop = frame[adj_y1:adj_y2, adj_x1:adj_x2]
                if target_crop.size > 0: cv2.imshow(TRACKER_WINDOW_NAME, cv2.resize(target_crop, TRACKER_WINDOW_SIZE))
            label_color = (0, 0, 255) if sid == target_id else (0, 255, 0)
            cv2.rectangle(frame, (adj_x1, adj_y1), (adj_x2, adj_y2), label_color, 2)
            cv2.putText(frame, f"ID:{sid}", (adj_x1, adj_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)

        for sid_to_clean in list(filter_sets.keys()):
            if sid_to_clean not in active_sids: del filter_sets[sid_to_clean]
        for sid_to_clean in list(stabilized_boxes.keys()):
            if sid_to_clean not in active_sids: del stabilized_boxes[sid_to_clean]
        
        if target_id is not None and not target_found_this_frame:
            time_since_lost = time.time() - target_last_seen_time
            if time_since_lost > TRACKER_GRACE_PERIOD:
                try: cv2.destroyWindow(TRACKER_WINDOW_NAME)
                except: pass
            else:
                last_box = last_good_box.get(target_id)
                if last_box:
                    lx1, ly1, lx2, ly2 = last_box
                    if lx2 > lx1 and ly2 > ly1:
                       live_crop = frame[ly1:ly2, lx1:lx2]
                    if live_crop.size > 0:
                        lost_view = cv2.resize(live_crop, TRACKER_WINDOW_SIZE)
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