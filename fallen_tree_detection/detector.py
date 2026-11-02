from typing import List, Dict, Any
import numpy as np
from ultralytics import YOLO
from config import settings


class FallenTreeDetector:
    def __init__(
        self,
        model_path: str = settings.MODEL_PATH,
        conf: float = settings.CONFIDENCE_THRESHOLD,
    ):
        self.model = YOLO(model_path)
        self.conf = conf
        print(f"[OK] Fallen Tree model loaded: {model_path}")

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect fallen trees in the frame.
        Returns list of detections.
        """
        results = self.model(frame, conf=self.conf, verbose=False)[0]

        detections = []
        if results.boxes is None:
            return detections

        for box in results.boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0].item())
            class_id = int(box.cls[0].item())

            xmin, ymin, xmax, ymax = map(int, xyxy)

            detections.append({
                "bbox": [xmin, ymin, xmax, ymax],
                "confidence": conf,
                "class_id": class_id,
                "label": "Fallen Tree",
            })

        return detections