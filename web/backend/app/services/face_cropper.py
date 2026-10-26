from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from ..config import settings
from ..utils.logger import logger
from .media_validator import MediaValidationError


NO_FACE_DETECTED_CODE = "NO_FACE_DETECTED"
NO_FACE_DETECTED_MESSAGE = (
    "Không tìm thấy khuôn mặt trong ảnh. "
    "Hãy dùng ảnh chụp rõ mặt, chính diện."
)


class FaceCropper:
    """
    Face-crop preprocessing aligned with the training data.

    Router4, the experts and X2DFD were trained on DeepfakeBench
    face crops, so full frames are reduced to the largest detected
    face before inference:
      - OpenCV Haar cascade (frontal face)
      - square crop centred on the face,
        side = max(w, h) * FACE_CROP_SCALE
      - square shifted (not cut) to stay inside the image
      - LANCZOS resize to FACE_CROP_SIZE x FACE_CROP_SIZE
      - lossless PNG output

    This is web-demo preprocessing and does not alter model
    thresholds, calibration, WFS, Router4, or final-score logic.
    """

    CASCADE_FILE = "haarcascade_frontalface_default.xml"

    _cascade = None


    @classmethod
    def _get_cascade(cls):

        if cls._cascade is None:

            import cv2  # type: ignore

            cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades
                + cls.CASCADE_FILE
            )

            if cascade.empty():
                raise RuntimeError(
                    "Haar cascade could not be loaded: "
                    f"{cls.CASCADE_FILE}"
                )

            cls._cascade = cascade

        return cls._cascade


    @staticmethod
    def provenance_tag() -> str:

        if not settings.FACE_CROP_ENABLED:
            return "disabled"

        return (
            f"haar_{settings.FACE_CROP_SCALE:g}x_"
            f"{settings.FACE_CROP_SIZE}"
        )


    @classmethod
    def detect_largest_face(
        cls,
        img: Image.Image,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Returns (x, y, w, h) of the largest face, or None."""

        import cv2  # type: ignore

        gray = cv2.cvtColor(
            np.asarray(
                img.convert("RGB")
            ),
            cv2.COLOR_RGB2GRAY,
        )

        min_size = int(
            settings.FACE_MIN_SIZE
        )

        faces = cls._get_cascade().detectMultiScale(
            gray,
            1.1,
            5,
            minSize=(min_size, min_size),
        )

        if len(faces) == 0:
            return None

        x, y, w, h = max(
            faces,
            key=lambda f: int(f[2]) * int(f[3]),
        )

        return int(x), int(y), int(w), int(h)


    @staticmethod
    def _square_box(
        face: Tuple[int, int, int, int],
        img_w: int,
        img_h: int,
    ) -> Tuple[int, int, int, int]:

        x, y, w, h = face

        side = int(
            round(
                max(w, h)
                * settings.FACE_CROP_SCALE
            )
        )

        side = max(
            1,
            min(side, img_w, img_h),
        )

        cx = x + w / 2.0
        cy = y + h / 2.0

        left = int(round(cx - side / 2.0))
        top = int(round(cy - side / 2.0))

        left = min(max(left, 0), img_w - side)
        top = min(max(top, 0), img_h - side)

        return left, top, left + side, top + side


    @classmethod
    def crop_image(
        cls,
        img: Image.Image,
    ) -> Optional[Image.Image]:
        """Returns the resized square face crop, or None if no face is found."""

        img = img.convert("RGB")

        face = cls.detect_largest_face(img)

        if face is None:
            return None

        box = cls._square_box(
            face,
            img.width,
            img.height,
        )

        size = int(settings.FACE_CROP_SIZE)

        return img.crop(box).resize(
            (size, size),
            Image.Resampling.LANCZOS,
        )


    @classmethod
    def crop_face_file(
        cls,
        src_path: Path,
        dst_path: Path,
    ) -> Path:
        """
        Crops the largest face of src_path into dst_path (PNG).

        Raises MediaValidationError(NO_FACE_DETECTED) when no face is found.
        """

        with Image.open(src_path) as src:
            cropped = cls.crop_image(src)

        if cropped is None:
            raise MediaValidationError(
                NO_FACE_DETECTED_CODE,
                NO_FACE_DETECTED_MESSAGE,
            )

        cropped.save(
            dst_path,
            format="PNG",
        )

        logger.info(
            "Face crop %s -> %s (%dx%d)",
            src_path.name,
            dst_path.name,
            cropped.width,
            cropped.height,
        )

        return dst_path
