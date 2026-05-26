# Warehouse Vision — Stage 0 Demo

Real-time inventory detection loop using YOLO11. Points a camera at a shelf, counts a target item, and fires a reorder alert when stock stays below threshold for a configurable dwell period.

Built as a proof-of-concept for Groundwork.ai facility automation.

---

## How it works

1. Each frame is run through YOLO11n (COCO model, ~5 MB)
2. Detections are filtered to the target class (default: `bottle`)
3. A rolling median over the last N frames smooths out jitter
4. If the smoothed count stays below `THRESHOLD` for `DWELL_SECONDS`, a **REORDER TRIGGERED** alert fires
5. Stock returning above threshold clears the alert

---

## Requirements

- Python 3.10+
- NVIDIA GPU recommended (CUDA); falls back to CPU automatically
- [iVCam](https://www.e2esoft.com/ivcam/) (phone as webcam) or any OpenCV-compatible camera

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install ultralytics opencv-python torch torchvision
```

The YOLO11n model (`yolo11n.pt`) downloads automatically on first run.

---

## Run

1. Open iVCam on your PC and connect your phone (or plug in a webcam)
2. Activate the venv, then:

```bash
python demo.py
```

---

## Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `R` | Reset smoothing buffer and alert state |
| `+` / `=` | Raise threshold |
| `-` | Lower threshold |

The window's **X button** also closes the demo cleanly.

---

## Config (top of `demo.py`)

| Variable | Default | Description |
|----------|---------|-------------|
| `VIDEO_SOURCE` | `0` | Camera index (iVCam = 0) |
| `TARGET_CLASS` | `"bottle"` | Any COCO class name |
| `THRESHOLD` | `4` | Alert when count drops below this |
| `DWELL_SECONDS` | `5` | Seconds below threshold before alert fires (use 1800 for production) |
| `SMOOTH_WINDOW` | `15` | Rolling median window (frames) |
| `CONFIDENCE` | `0.35` | Detection confidence cutoff |
| `MODEL_FILE` | `"yolo11n.pt"` | Swap in a custom-trained model here |

---

## iVCam setup notes (Windows)

- CAP_DSHOW backend is required — MSMF drops frames with virtual cameras
- Firewall rules needed for ports **5895** and **5896** (all profiles, including Public)
- PC client: `C:\Program Files\e2eSoft\iVCam\iVCam.exe`

---

## Next steps

- Collect training images of actual inventory items (pallets, boxes, etc.)
- Label with [Roboflow](https://roboflow.com), train a custom YOLO11n model, swap `best.pt` into `MODEL_FILE`
- Entry/exit pallet counting at aisle entrances (solves stacking/occlusion problem)
