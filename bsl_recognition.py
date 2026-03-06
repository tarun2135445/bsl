"""
BSL (British Sign Language) Recognition System
================================================
Modes:
  collect   - Capture hand landmark data for a sign label
  train     - Train the classifier on collected data
  recognize - Live recognition from webcam
  list      - Show sample counts per label
  delete    - Remove all samples for a label

Usage:
  python bsl_recognition.py collect --label A --samples 200
  python bsl_recognition.py train
  python bsl_recognition.py recognize
"""

import argparse
import csv
import os
import pickle
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
DATA_FILE = DATA_DIR / "landmarks.csv"
MODEL_FILE = Path("bsl_model.pkl")
LANDMARK_MODEL = Path("hand_landmarker.task")
LANDMARK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# ─── MediaPipe Tasks API shortcuts ────────────────────────────────────────────
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode
BaseOptions = mp.tasks.BaseOptions

NUM_LANDMARKS = 21
FEATURES_PER_HAND = NUM_LANDMARKS * 2  # normalised (x-wrist, y-wrist)

# Hand skeleton connections for manual drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (0, 9), (9, 10), (10, 11), (11, 12),      # middle
    (0, 13), (13, 14), (14, 15), (15, 16),    # ring
    (0, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (5, 9), (9, 13), (13, 17),                # palm knuckles
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def ensure_landmark_model():
    """Download the MediaPipe hand landmark model if not already present."""
    if LANDMARK_MODEL.exists():
        return
    print(f"[INFO] Downloading hand landmark model from Google MediaPipe storage...")
    urllib.request.urlretrieve(LANDMARK_MODEL_URL, LANDMARK_MODEL)
    print(f"[INFO] Saved to {LANDMARK_MODEL}")


def extract_features(hand_landmarks):
    """Return a flat list of (x-wrist, y-wrist) normalised coordinates."""
    wrist_x = hand_landmarks[0].x
    wrist_y = hand_landmarks[0].y
    coords = []
    for lm in hand_landmarks:
        coords.append(lm.x - wrist_x)
        coords.append(lm.y - wrist_y)
    return coords


def draw_landmarks(frame, hand_landmarks, h, w):
    """Draw hand skeleton on the frame."""
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 220, 130), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)
        cv2.circle(frame, (x, y), 4, (0, 180, 100), 1)


def draw_info(frame, text, pos=(10, 30), color=(0, 255, 0), size=0.9, thickness=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), thickness + 2)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)


def open_camera(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {camera_index}. Try --camera with a different index.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


# ─── Mode: collect ────────────────────────────────────────────────────────────

def collect(label: str, n_samples: int, camera_index: int, delay: float):
    """Capture hand landmarks and append them to the CSV dataset."""
    ensure_landmark_model()
    DATA_DIR.mkdir(exist_ok=True)

    write_header = not DATA_FILE.exists()
    csv_fp = open(DATA_FILE, "a", newline="")
    writer = csv.writer(csv_fp)
    if write_header:
        header = [f"f{i}" for i in range(FEATURES_PER_HAND)] + ["label"]
        writer.writerow(header)

    cap = open_camera(camera_index)
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(LANDMARK_MODEL)),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    collected = 0
    collecting = False
    frame_ts = 0

    print(f"\n[COLLECT] Label: '{label}'  Target samples: {n_samples}")
    print("  Press SPACE to start/pause collection, Q to quit early.\n")

    with HandLandmarker.create_from_options(options) as landmarker:
        while collected < n_samples:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            frame_ts += 33  # ~30 fps in ms

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = landmarker.detect_for_video(mp_image, frame_ts)

            hand_detected = bool(result.hand_landmarks)

            if hand_detected:
                hl = result.hand_landmarks[0]
                draw_landmarks(frame, hl, h, w)

                if collecting:
                    features = extract_features(hl)
                    writer.writerow(features + [label])
                    collected += 1
                    time.sleep(delay)

            status = "COLLECTING" if collecting else "PAUSED"
            color = (0, 200, 0) if collecting else (0, 100, 255)
            draw_info(frame, f"Label: {label}  [{status}]", (10, 30), color)
            draw_info(frame, f"Samples: {collected}/{n_samples}", (10, 65))
            if not hand_detected:
                draw_info(frame, "No hand detected", (10, 100), (0, 0, 255))
            draw_info(frame, "SPACE=start/pause  Q=quit", (10, h - 15), (200, 200, 200), 0.55, 1)

            bar_w = int(w * collected / n_samples)
            cv2.rectangle(frame, (0, h - 8), (bar_w, h), (0, 200, 0), -1)

            cv2.imshow("BSL - Collect", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                collecting = not collecting

    csv_fp.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"[COLLECT] Done. {collected} samples saved for '{label}' → {DATA_FILE}")


# ─── Mode: train ──────────────────────────────────────────────────────────────

def train():
    """Train a Random Forest classifier on the collected landmark CSV."""
    if not DATA_FILE.exists():
        print(f"[ERROR] No data file found at {DATA_FILE}. Run 'collect' first.")
        sys.exit(1)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report

    print(f"[TRAIN] Loading data from {DATA_FILE} ...")
    raw = np.genfromtxt(DATA_FILE, delimiter=",", dtype=None, encoding="utf-8", names=True)

    feature_cols = [f"f{i}" for i in range(FEATURES_PER_HAND)]
    X = np.column_stack([raw[c].astype(float) for c in feature_cols])
    y_raw = raw["label"].astype(str)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    classes = le.classes_
    print(f"[TRAIN] Classes ({len(classes)}): {list(classes)}")
    print(f"[TRAIN] Samples:  {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\n[TRAIN] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))

    scores = cross_val_score(clf, X, y, cv=5)
    print(f"[TRAIN] 5-fold CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    payload = {"model": clf, "label_encoder": le}
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(payload, f)
    print(f"\n[TRAIN] Model saved → {MODEL_FILE}")


# ─── Mode: recognize ──────────────────────────────────────────────────────────

def recognize(camera_index: int, confidence_threshold: float):
    """Run live BSL sign recognition from webcam."""
    ensure_landmark_model()

    if not MODEL_FILE.exists():
        print(f"[ERROR] No model file found at {MODEL_FILE}. Run 'train' first.")
        sys.exit(1)

    with open(MODEL_FILE, "rb") as f:
        payload = pickle.load(f)

    clf = payload["model"]
    le = payload["label_encoder"]

    cap = open_camera(camera_index)
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(LANDMARK_MODEL)),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    SMOOTH = 7
    pred_history = []
    last_label = ""
    frame_ts = 0

    print("\n[RECOGNIZE] Press Q to quit.\n")

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            frame_ts += 33

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = landmarker.detect_for_video(mp_image, frame_ts)

            if result.hand_landmarks:
                hl = result.hand_landmarks[0]
                draw_landmarks(frame, hl, h, w)

                features = extract_features(hl)
                proba = clf.predict_proba([features])[0]
                top_idx = int(np.argmax(proba))
                top_conf = proba[top_idx]
                top_label = le.inverse_transform([top_idx])[0]

                pred_history.append(top_label)
                if len(pred_history) > SMOOTH:
                    pred_history.pop(0)

                voted = Counter(pred_history).most_common(1)[0][0]
                last_label = voted if top_conf >= confidence_threshold else "?"

                # Big sign box (top-right)
                cv2.rectangle(frame, (w - 170, 0), (w, 110), (0, 0, 0), -1)
                cv2.putText(frame, last_label, (w - 155, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0, 255, 100), 5)
                draw_info(frame, f"{top_conf*100:.1f}%", (w - 155, 128),
                          (200, 200, 200), 0.65, 1)

                # Top-3 bar chart
                sorted_idx = np.argsort(proba)[::-1][:3]
                for rank, idx in enumerate(sorted_idx):
                    lbl = le.inverse_transform([idx])[0]
                    bar_len = int(220 * proba[idx])
                    y_off = h - 95 + rank * 30
                    cv2.rectangle(frame, (10, y_off), (10 + bar_len, y_off + 22),
                                  (0, 160, 255), -1)
                    draw_info(frame, f"{lbl}: {proba[idx]*100:.1f}%",
                              (14, y_off + 17), (255, 255, 255), 0.56, 1)
            else:
                draw_info(frame, "Show your hand to the camera", (10, 55), (0, 150, 255))
                pred_history.clear()

            draw_info(frame, "BSL Sign Recognition", (10, 30))
            draw_info(frame, "Q = quit", (10, h - 10), (180, 180, 180), 0.5, 1)

            cv2.imshow("BSL - Recognize", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


# ─── Mode: list ───────────────────────────────────────────────────────────────

def list_labels():
    if not DATA_FILE.exists():
        print(f"No dataset found at {DATA_FILE}.")
        return
    with open(DATA_FILE, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        labels = [row[-1] for row in reader]
    counts = Counter(labels)
    print(f"\nDataset: {DATA_FILE}  ({sum(counts.values())} total samples)\n")
    print(f"{'Label':<20} {'Samples':>8}")
    print("-" * 30)
    for lbl, cnt in sorted(counts.items()):
        print(f"{lbl:<20} {cnt:>8}")
    print()


# ─── Mode: delete ─────────────────────────────────────────────────────────────

def delete_label(label: str):
    if not DATA_FILE.exists():
        print("No dataset found.")
        return
    tmp = DATA_DIR / "_tmp.csv"
    kept = removed = 0
    with open(DATA_FILE, "r") as fin, open(tmp, "w", newline="") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        writer.writerow(next(reader))  # header
        for row in reader:
            if row[-1] == label:
                removed += 1
            else:
                writer.writerow(row)
                kept += 1
    tmp.replace(DATA_FILE)
    print(f"[DELETE] Removed {removed} samples for '{label}'. {kept} samples remain.")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BSL Sign Recognition — collect data, train, and recognise signs",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    p_col = sub.add_parser("collect", help="Record hand landmark samples for a sign")
    p_col.add_argument("--label", required=True, help="Sign label (e.g. A, HELLO, THANK_YOU)")
    p_col.add_argument("--samples", type=int, default=300)
    p_col.add_argument("--camera", type=int, default=0)
    p_col.add_argument("--delay", type=float, default=0.05,
                       help="Delay between frames in seconds (default: 0.05)")

    sub.add_parser("train", help="Train the classifier on collected data")

    p_rec = sub.add_parser("recognize", help="Live webcam sign recognition")
    p_rec.add_argument("--camera", type=int, default=0)
    p_rec.add_argument("--confidence", type=float, default=0.5,
                       help="Min confidence to display a prediction (default: 0.5)")

    sub.add_parser("list", help="List labels and sample counts in the dataset")

    p_del = sub.add_parser("delete", help="Remove all samples for a label")
    p_del.add_argument("--label", required=True)

    args = parser.parse_args()

    if args.mode == "collect":
        collect(args.label, args.samples, args.camera, args.delay)
    elif args.mode == "train":
        train()
    elif args.mode == "recognize":
        recognize(args.camera, args.confidence)
    elif args.mode == "list":
        list_labels()
    elif args.mode == "delete":
        delete_label(args.label)


if __name__ == "__main__":
    main()
