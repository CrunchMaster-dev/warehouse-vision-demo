"""
Stage 0 desk demo — proves the full inventory-vision loop on existing hardware.
Detects a chosen COCO class, smooths the count, fires a 'reorder' alert when
stock stays below threshold for the dwell period.

Controls:
  Q       quit
  R       reset (clears smoothing buffer + alert state)
  + / -   raise / lower threshold on the fly (great for demo)
"""

import time
from collections import deque
from statistics import median

import cv2
import torch
from ultralytics import YOLO

# -------------------- CONFIG (tweak these for the demo) --------------------
VIDEO_SOURCE   = 0              # iVCam virtual camera (phone via iVCam app)
TARGET_CLASS   = "bottle"       # COCO class to count. Try: cup, bottle, book, cell phone.
THRESHOLD      = 4              # alert if smoothed count stays below this
DWELL_SECONDS  = 5              # ...for this many seconds (set low for demo; 1800 in prod)
SMOOTH_WINDOW  = 15             # rolling median over N frames to kill jitter
CONFIDENCE     = 0.35           # detection confidence (0.0 - 1.0)
MODEL_FILE     = "yolo11n.pt"   # nano model. Auto-downloads on first run (~5 MB).
# --------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = YOLO(MODEL_FILE)
    class_names = model.names  # idx -> name

    # Find the class index for our target. Bail clearly if it doesn't exist.
    target_idx = next((i for i, n in class_names.items() if n == TARGET_CLASS), None)
    if target_idx is None:
        raise SystemExit(f"'{TARGET_CLASS}' is not a COCO class. Try: cup, bottle, book, cell phone.")

    backend = cv2.CAP_DSHOW if isinstance(VIDEO_SOURCE, int) else cv2.CAP_ANY
    cap = cv2.VideoCapture(VIDEO_SOURCE, backend)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {VIDEO_SOURCE}")

    history = deque(maxlen=SMOOTH_WINDOW)
    below_since = None
    alert_active = False
    threshold = THRESHOLD

    fps_t0, fps_n, fps = time.time(), 0, 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Lost video frame, retrying...")
            time.sleep(0.5)
            continue

        # Run detection
        results = model(frame, conf=CONFIDENCE, device=device, verbose=False)[0]

        # Filter to target class + draw boxes
        raw_count = 0
        for box in results.boxes:
            cls_idx = int(box.cls[0])
            if cls_idx != target_idx:
                continue
            raw_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(frame, f"{TARGET_CLASS} {conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

        # Smoothed count
        history.append(raw_count)
        smoothed = int(round(median(history)))

        # Alert logic: must be below threshold continuously for DWELL_SECONDS
        now = time.time()
        if smoothed < threshold:
            if below_since is None:
                below_since = now
            if now - below_since >= DWELL_SECONDS:
                alert_active = True
        else:
            below_since = None
            alert_active = False

        # FPS
        fps_n += 1
        if now - fps_t0 >= 1.0:
            fps = fps_n / (now - fps_t0)
            fps_t0, fps_n = now, 0

        # -------------------- HUD overlay --------------------
        h, w = frame.shape[:2]

        # Top bar (status)
        bar_color = (0, 0, 200) if alert_active else (0, 140, 0)
        cv2.rectangle(frame, (0, 0), (w, 60), bar_color, -1)
        status_text = "STATUS: REORDER TRIGGERED" if alert_active else "STATUS: OK"
        cv2.putText(frame, status_text, (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Bottom bar (stats)
        cv2.rectangle(frame, (0, h - 70), (w, h), (40, 40, 40), -1)
        line = f"{TARGET_CLASS}: raw={raw_count}  smoothed={smoothed}  threshold={threshold}  fps={fps:.1f}"
        cv2.putText(frame, line, (15, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "Q quit  |  R reset  |  + raise threshold  |  - lower threshold",
                    (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("Warehouse Vision - Stage 0 Demo", frame)

        # -------------------- key handling --------------------
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if cv2.getWindowProperty("Warehouse Vision - Stage 0 Demo", cv2.WND_PROP_VISIBLE) < 1:
            break
        elif key == ord("r"):
            history.clear()
            below_since = None
            alert_active = False
            print("Reset.")
        elif key in (ord("+"), ord("=")):
            threshold += 1
            print(f"Threshold -> {threshold}")
        elif key == ord("-"):
            threshold = max(0, threshold - 1)
            print(f"Threshold -> {threshold}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
