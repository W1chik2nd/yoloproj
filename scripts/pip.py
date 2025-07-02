import os
import time
import pickle
import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
from sort import Sort

DIST_TH    = 0.90
BANK_FILE  = "idbank.pkl"
TARGET_FPS = 60
MAX_FEATS  = 10
CLEAR_BANK_ON_EXIT = True
TRACKER_WINDOW_NAME = "Tracker View"
TRACKER_WINDOW_SIZE = (800, 450)
HEADROOM_RATIO = 0.20
TRACKER_GRACE_PERIOD = 1.5
FILTER_MIN_CUTOFF = 0.05
FILTER_BETA = 1.0

target_id = None
current_detections = []
target_last_seen_time = 0
last_good_box = {}
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
    if not id_bank: id_bank[next_sid] = [feat]; new_id = next_sid; next_sid += 1; return new_id
    best_id, best_d = None, float("inf")
    for sid, feats in id_bank.items():
        if sid in sids_to_exclude: continue
        valid = [f for f in feats if isinstance(f, np.ndarray) and f.ndim == 1];
        if not valid: continue
        d = min(np.linalg.norm(feat - f) for f in valid)
        if d < best_d: best_d, best_id = d, sid
    if best_id is not None and best_d < DIST_TH:
        id_bank[best_id].append(feat)
        if len(id_bank[best_id]) > MAX_FEATS: id_bank[best_id].pop(0)
        return best_id
    new_id = next_sid;
    while new_id in id_bank or new_id in sids_to_exclude: new_id += 1
    next_sid = new_id + 1; id_bank[new_id] = [feat]; return new_id

def select_target_id(event, x, y, flags, param):
    global target_id, target_last_seen_time, last_good_box, filter_sets
    if event == cv2.EVENT_LBUTTONDOWN:
        if target_id is not None and target_id in filter_sets: del filter_sets[target_id]
        target_id = None
        for det in current_detections:
            x1, y1, x2, y2, sid = det
            if x1 < x < x2 and y1 < y < y2:
                target_id = sid; print(f"[INFO] Target locked on ID: {sid}"); target_last_seen_time = time.time(); break
    elif event == cv2.EVENT_RBUTTONDOWN:
        print("[INFO] Target unlocked.")
        if target_id is not None and target_id in filter_sets: del filter_sets[target_id]
        target_id = None
        try: cv2.destroyWindow(TRACKER_WINDOW_NAME)
        except: pass

MAIN_WINDOW_NAME = "YOLO Tracking"
cv2.namedWindow(MAIN_WINDOW_NAME); cv2.setMouseCallback(MAIN_WINDOW_NAME, select_target_id)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW); interval = 1.0 / TARGET_FPS; last_t = 0.0
frame_h, frame_w = None, None
fixed_16_9_w, fixed_16_9_h = None, None

try:
    while True:
        ret, frame = cap.read();
        if not ret: break
        
        if frame_h is None:
            frame_h, frame_w, _ = frame.shape
            if (frame_w / frame_h) > (16/9):
                fixed_16_9_h, fixed_16_9_w = frame_h, int(frame_h * (16/9))
            else:
                fixed_16_9_w, fixed_16_9_h = frame_w, int(frame_w / (16/9))
            print(f"[INFO] Frame: {frame_w}x{frame_h}, Fixed 16:9 Viewport: {fixed_16_9_w}x{fixed_16_9_h}")

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
                    exclusion_list = sids_in_current_frame.copy()
                    if target_id is not None: exclusion_list.add(target_id)
                    sid = assign_id(extract_feat(crop), sids_to_exclude=exclusion_list)
                    track_id_to_sid[track_id] = sid
            
            if sid == -1: continue
            
            sids_in_current_frame.add(sid)
            active_sids.add(sid)
            
            if sid not in filter_sets:
                freq = TARGET_FPS
                filter_sets[sid] = {
                    'cx': OneEuroFilter(freq, FILTER_MIN_CUTOFF, FILTER_BETA),
                    'cy': OneEuroFilter(freq, FILTER_MIN_CUTOFF, FILTER_BETA)
                }
            
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            smooth_cx = filter_sets[sid]['cx'](cx, now)
            smooth_cy = filter_sets[sid]['cy'](cy, now)

            adj_x1 = int(smooth_cx - fixed_16_9_w / 2)
            adj_y1 = int(smooth_cy - (fixed_16_9_h / 2) * (1 - HEADROOM_RATIO)) # 应用头顶空间

            adj_x1 = max(0, min(adj_x1, frame_w - fixed_16_9_w))
            adj_y1 = max(0, min(adj_y1, frame_h - fixed_16_9_h))
            
            adj_x2 = adj_x1 + fixed_16_9_w
            adj_y2 = adj_y1 + fixed_16_9_h
            
            display_box = (adj_x1, adj_y1, adj_x2, adj_y2)
            current_detections.append((adj_x1, adj_y1, adj_x2, adj_y2, sid))

            if sid == target_id:
                target_found_this_frame = True; target_last_seen_time = time.time()
                last_good_box[target_id] = display_box
                
                target_crop = frame[adj_y1:adj_y2, adj_x1:adj_x2]
                if target_crop.size > 0:
                    resized_crop = cv2.resize(target_crop, TRACKER_WINDOW_SIZE)
                    cv2.imshow(TRACKER_WINDOW_NAME, resized_crop)
            
            label_color = (0, 0, 255) if sid == target_id else (0, 255, 0)
            cv2.rectangle(frame, (adj_x1, adj_y1), (adj_x2, adj_y2), label_color, 2)
            cv2.putText(frame, f"ID:{sid}", (adj_x1, adj_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)

        for sid_to_clean in list(filter_sets.keys()):
            if sid_to_clean not in active_sids: del filter_sets[sid_to_clean]
        
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