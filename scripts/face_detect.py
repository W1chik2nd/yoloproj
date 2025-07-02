import cv2, time, os, pickle
import numpy as np
from insightface.app import FaceAnalysis

BANK_FILE = "face_idbank.pkl"
DIST_TH_EMB = 0.6
IOU_TH = 0.5
TRACK_TIMEOUT = 2.0
TARGET_FPS = 15
CLEAR_ON_EXIT = False
MAX_FEATS = 8

fa = FaceAnalysis(allowed_modules=['detection','recognition'], providers=['CUDAExecutionProvider'])
fa.prepare(ctx_id=0, det_size=(640,640))

if os.path.exists(BANK_FILE):
    raw = pickle.load(open(BANK_FILE, 'rb'))
    id_bank = {int(fid): (vals if isinstance(vals, list) else [vals]) for fid, vals in raw.items()}
    next_id = max(id_bank.keys()) + 1
else:
    id_bank = {}
    next_id = 1


def assign_id(emb):
    global next_id
    best_d, best_id = float('inf'), None
    for fid, vecs in id_bank.items():
        d = min(np.linalg.norm(emb - v) for v in vecs)
        if d < best_d:
            best_d, best_id = d, fid
    if best_id is not None and best_d < DIST_TH_EMB:
        bucket = id_bank[best_id]
        bucket.append(emb)
        if len(bucket) > MAX_FEATS:
            bucket.pop(0)
        return best_id
    id_bank[next_id] = [emb]
    next_id += 1
    return next_id - 1


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (areaA + areaB - inter)

active_tracks = []

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
interval = 1.0 / TARGET_FPS
last_t = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    now = time.time()
    if now - last_t < interval:
        continue
    last_t = now

    faces = fa.get(frame)
    active_tracks = [tr for tr in active_tracks if now - tr['last_seen'] < TRACK_TIMEOUT]
    new_tracks = []

    for f in faces:
        x1, y1, x2, y2 = map(int, f.bbox)
        emb = f.embedding
        matched = None
        for tr in active_tracks:
            if iou((x1, y1, x2, y2), tr['bbox']) > IOU_TH:
                matched = tr
                break
        if matched:
            face_id = matched['id']
            matched['bbox'] = (x1, y1, x2, y2)
            matched['last_seen'] = now
            new_tracks.append(matched)
        else:
            face_id = assign_id(emb)
            new_tracks.append({'id': face_id, 'bbox': (x1, y1, x2, y2), 'last_seen': now})
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID:{face_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    active_tracks = new_tracks
    cv2.imshow("FaceID with IoU+Emb", frame)
    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
        break

cap.release()
cv2.destroyAllWindows()
if CLEAR_ON_EXIT:
    if os.path.exists(BANK_FILE):
        os.remove(BANK_FILE)
else:
    with open(BANK_FILE, 'wb') as f:
        pickle.dump(id_bank, f)
