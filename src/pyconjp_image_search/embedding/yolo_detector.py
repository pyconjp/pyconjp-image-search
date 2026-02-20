"""YOLO11 wrapper for object detection."""

import uuid
from pathlib import Path

from ultralytics import YOLO

from pyconjp_image_search.config import YOLO_MODEL_NAME, YOLO_MODEL_PATH
from pyconjp_image_search.models import ObjectDetection


class YOLODetector:
    """Detect objects using YOLO11 (COCO 80 classes)."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        imgsz: int = 1280,
        conf: float = 0.25,
    ) -> None:
        path = str(model_path or YOLO_MODEL_PATH)
        self.model = YOLO(path)
        self.imgsz = imgsz
        self.conf = conf
        self.model_name = YOLO_MODEL_NAME

    def detect(self, image_path: str | Path, image_id: int) -> list[ObjectDetection]:
        """Detect objects in an image and return ObjectDetection objects.

        Args:
            image_path: Path to the image file.
            image_id: Database ID of the image (FK to images.id).

        Returns:
            List of ObjectDetection objects, one per detected object.
        """
        results = self.model(str(image_path), imgsz=self.imgsz, conf=self.conf, verbose=False)
        detections: list[ObjectDetection] = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                detections.append(
                    ObjectDetection(
                        detection_id=str(uuid.uuid4()),
                        image_id=image_id,
                        model_name=self.model_name,
                        label=self.model.names[int(box.cls)],
                        confidence=float(box.conf),
                        bbox_x1=x1,
                        bbox_y1=y1,
                        bbox_x2=x2,
                        bbox_y2=y2,
                    )
                )

        return detections
