"""Model loading, inference and drawing helpers."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

from utils.analysis import estimate_diameter_mm, dominant_color_name

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# BGR palette used for boxes (one per class, cycles if more classes)
PALETTE = [
    (107, 107, 15),   # deep teal
    (35, 185, 232),   # signal yellow
    (58, 100, 184),   # copper
    (120, 160, 60),
    (160, 90, 120),
    (200, 140, 40),
]


def list_local_models() -> list[str]:
    """Return .pt files found in the models/ folder."""
    MODELS_DIR.mkdir(exist_ok=True)
    return sorted(p.name for p in MODELS_DIR.glob("*.pt"))


@st.cache_resource(show_spinner="Loading model...")
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def run_detection(
    model: YOLO,
    image_bgr: np.ndarray,
    conf: float,
    iou: float,
    imgsz: int,
    px_per_mm: float | None,
):
    """Run YOLO on one BGR image.

    Returns (annotated_bgr, detections) where detections is a list of dicts.
    """
    result = model.predict(image_bgr, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
    names = result.names
    annotated = image_bgr.copy()
    detections: list[dict] = []

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return annotated, detections

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)

    # scale line thickness / font to image size
    scale = max(image_bgr.shape[:2]) / 1000
    thickness = max(1, int(round(2 * scale)))
    font_scale = max(0.4, 0.5 * scale)

    for i, (box, c, k) in enumerate(zip(xyxy, confs, clss), start=1):
        x1, y1, x2, y2 = [int(v) for v in box]
        w_px, h_px = x2 - x1, y2 - y1
        diameter_mm = estimate_diameter_mm(w_px, h_px, px_per_mm)
        color_name = dominant_color_name(image_bgr[max(y1, 0):y2, max(x1, 0):x2])

        detections.append(
            {
                "ID": i,
                "Class": names.get(k, str(k)),
                "Confidence": round(float(c), 3),
                "Color": color_name,
                "Width (px)": w_px,
                "Height (px)": h_px,
                "Diameter (mm)": diameter_mm,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            }
        )

        color = PALETTE[k % len(PALETTE)]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        label = str(i)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            annotated, label, (x1 + 3, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return annotated, detections


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def decode_image(file_bytes: bytes) -> np.ndarray | None:
    arr = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def encode_png(image_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image_bgr)
    return buf.tobytes() if ok else b""


def detect_boxes(model: YOLO, image_bgr: np.ndarray, conf: float, iou: float, imgsz: int) -> list[dict]:
    """Run YOLO and return plain boxes (no drawing), numbered top-to-bottom, left-to-right."""
    result = model.predict(image_bgr, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    items = []
    for b, c in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = [int(v) for v in b]
        items.append({"Confidence": round(float(c), 3), "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    items.sort(key=lambda d: ((d["y1"] + d["y2"]) // 80, d["x1"]))
    for i, d in enumerate(items, start=1):
        d["ID"] = i
    return items
