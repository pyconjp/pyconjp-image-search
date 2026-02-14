"""InsightFace wrapper for face detection and embedding extraction."""

import uuid
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from pyconjp_image_search.config import INSIGHTFACE_MODEL_NAME
from pyconjp_image_search.models import FaceDetection


class InsightFaceEmbedder:
    """Detect faces and extract ArcFace embeddings using InsightFace."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        device: str = "cuda",
    ) -> None:
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        self.app = FaceAnalysis(name=model_name, providers=providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.model_full_name = INSIGHTFACE_MODEL_NAME

    def detect_faces(self, image_path: Path, image_id: int) -> list[FaceDetection]:
        """Detect faces in an image and return FaceDetection objects.

        Args:
            image_path: Path to the image file.
            image_id: Database ID of the image (FK to images.id).

        Returns:
            List of FaceDetection objects, one per detected face.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            return []

        faces = self.app.get(img)
        detections = []

        for face in faces:
            bbox = face.bbox.astype(float)
            gender = "M" if face.gender == 1 else "F"

            detection = FaceDetection(
                face_id=str(uuid.uuid4()),
                image_id=image_id,
                model_name=self.model_full_name,
                bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                det_score=float(face.det_score),
                landmark=face.kps.tolist() if face.kps is not None else None,
                age=int(face.age),
                gender=gender,
                embedding=face.normed_embedding.astype(np.float32),
                person_label=None,
                cluster_id=None,
            )
            detections.append(detection)

        return detections
