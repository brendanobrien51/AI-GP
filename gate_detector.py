"""
YOLOv8 Gate Detector
====================
Inference wrapper for trained gate_detector.pt.
Returns same dict format as score_contour() so it drops in
as a replacement for find_best_gate_contour() with zero downstream changes.
"""

import numpy as np
from pathlib import Path


class GateDetector:
    """
    Loads a trained YOLOv8-nano gate detector and runs inference.

    Usage:
        detector = GateDetector("gate_detector.pt")
        result = detector.detect(frame)   # BGR uint8 numpy array
        # result: {"centroid_px": (cx, cy), "score": float, ...} or None
    """

    def __init__(self, model_path: str = "gate_detector.pt",
                 conf_threshold: float = 0.28):
        self._conf = conf_threshold
        self._model = None
        path = Path(model_path)

        if not path.exists():
            print(f"[GateDetector] WARNING: model not found at {path}. "
                  f"Falling back to contour detection every frame.")
            return

        try:
            from ultralytics import YOLO
            self._model = YOLO(str(path))
            # Warm up the model (first inference is slow due to CUDA compilation)
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            self._model(dummy, verbose=False)
            print(f"[GateDetector] Loaded {path} (conf_threshold={conf_threshold})")
        except Exception as e:
            print(f"[GateDetector] WARNING: Failed to load model: {e}. "
                  f"Falling back to contour detection.")
            self._model = None

    @property
    def available(self) -> bool:
        return self._model is not None

    def detect(self, frame: np.ndarray) -> dict | None:
        """
        Run YOLOv8 gate detection on a single BGR frame.

        Returns dict compatible with score_contour() output:
            {"centroid_px": (cx, cy), "score": conf, "rect": None,
             "area_frac": 0.0, "aspect": 1.0, "contour": None}
        or None if no confident detection.
        """
        if self._model is None:
            return None

        try:
            results = self._model(frame, verbose=False, conf=self._conf)
        except Exception:
            return None

        best = None
        best_conf = 0.0

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < self._conf or conf <= best_conf:
                    continue
                # xyxy bounding box
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                fh, fw = frame.shape[:2]
                area_frac = ((x2 - x1) * (y2 - y1)) / (fw * fh)
                w = x2 - x1
                h = y2 - y1
                aspect = max(w, h) / (min(w, h) + 1e-6)

                best_conf = conf
                best = {
                    "centroid_px": (cx, cy),
                    "score": conf,
                    "rect": None,
                    "area_frac": area_frac,
                    "aspect": aspect,
                    "contour": None,
                    "bbox_xyxy": (int(x1), int(y1), int(x2), int(y2)),
                }

        return best
