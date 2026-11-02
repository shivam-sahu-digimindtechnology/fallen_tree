from typing import List, Dict, Any
import cv2
import numpy as np


COLOR_BOX = (0, 255, 0)       # green
COLOR_TEXT = (255, 255, 255)
COLOR_ALERT_BG = (0, 120, 0)  # dark green
COLOR_ALERT_TEXT = (255, 255, 255)


def draw_bounding_boxes(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    thickness: int = 2,
    font_scale: float = 0.65,
) -> np.ndarray:
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # ========== FULL-WIDTH ALERT BANNER (only when tree detected) ==========
    if detections:
        banner_height = 36
        cv2.rectangle(annotated, (0, 0), (w, banner_height), COLOR_ALERT_BG, -1)

        alert_text = f"⚠  FALLEN TREE DETECTED  |  Count: {len(detections)}"

        (tw, th), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
        text_x = (w - tw) // 2
        text_y = (banner_height + th) // 2

        cv2.putText(
            annotated,
            alert_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            COLOR_ALERT_TEXT,
            2,
            cv2.LINE_AA,
        )

    # ========== DRAW BOUNDING BOXES ==========
    for det in detections:
        xmin, ymin, xmax, ymax = det["bbox"]
        conf = det.get("confidence", 0.0)
        label = det.get("label", "Fallen Tree")

        text = f"{label} {conf:.2f}"

        # Box
        cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), COLOR_BOX, thickness + 1)

        # Text background
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(
            annotated,
            (xmin, ymin - th - baseline - 6),
            (xmin + tw + 6, ymin),
            COLOR_BOX,
            -1,
        )

        cv2.putText(
            annotated,
            text,
            (xmin + 3, ymin - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            COLOR_TEXT,
            1,
            cv2.LINE_AA,
        )

    return annotated


def print_detections(detections: List[Dict[str, Any]], frame_idx: int = -1) -> None:
    if not detections:
        return

    prefix = f"[Frame {frame_idx}] " if frame_idx >= 0 else ""
    print(f"{prefix}=== Fallen Tree Detections ({len(detections)}) ===")
    for i, det in enumerate(detections, 1):
        print(
            f"  {i}. {det.get('label', 'Fallen Tree'):15s} "
            f"conf={det.get('confidence', 0):.3f}  "
            f"bbox={det.get('bbox')}"
        )
    print()