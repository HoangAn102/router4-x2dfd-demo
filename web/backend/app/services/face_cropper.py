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

    Detection order (first hit wins, so images the strict Haar pass
    already handled are cropped exactly as before):
      1. "haar"          strict Haar on the full image (1.1, 5)
      2. "yunet"         OpenCV YuNet CNN detector (FACE_YUNET_MODEL, score >= FACE_YUNET_SCORE)
                         on a copy downscaled to YUNET_LONG_SIDE; catches close-up selfies Haar misses
      3. "dlib"          dlib HOG frontal detector, upsample 1 (only if dlib is installed)

    A relaxed Haar pass (1280px, 1.05, 3) was tried and rejected on 2026-10-26: it
    returned glasses/wall/shelf boxes on phone selfies and fired on 13/29 face-free images.
    """

    CASCADE_FILE = "haarcascade_frontalface_default.xml"
    YUNET_LONG_SIDE = 1280

    _cascade = None
    _yunet = None


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
    def provenance_tag(detector: str = "haar") -> str:

        if not settings.FACE_CROP_ENABLED:
            return "disabled"

        return (
            f"{detector}_{settings.FACE_CROP_SCALE:g}x_"
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


    @classmethod
    def _detect_yunet(
        cls,
        img: Image.Image,
    ) -> Optional[Tuple[int, int, int, int]]:
        """YuNet on a downscaled copy; box mapped back to img coordinates. None if model missing."""

        import cv2  # type: ignore

        model = Path(settings.FACE_YUNET_MODEL)
        if not model.is_file():
            return None

        rgb = img.convert("RGB")
        scale = min(1.0, cls.YUNET_LONG_SIDE / max(rgb.size))
        if scale < 1.0:
            rgb = rgb.resize(
                (round(rgb.width * scale), round(rgb.height * scale)),
                Image.Resampling.LANCZOS,
            )

        if cls._yunet is None:
            cls._yunet = cv2.FaceDetectorYN.create(
                str(model), "", rgb.size, float(settings.FACE_YUNET_SCORE), 0.3, 5000
            )
        cls._yunet.setInputSize(rgb.size)
        cls._yunet.setScoreThreshold(float(settings.FACE_YUNET_SCORE))

        _, faces = cls._yunet.detect(cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR))
        if faces is None or len(faces) == 0:
            return None

        f = max(faces, key=lambda r: float(r[2]) * float(r[3]))
        x, y = max(0.0, float(f[0])), max(0.0, float(f[1]))
        w = min(float(f[2]), rgb.width - x)
        h = min(float(f[3]), rgb.height - y)
        x, y, w, h = (int(round(v / scale)) for v in (x, y, w, h))
        if w < settings.FACE_MIN_SIZE or h < settings.FACE_MIN_SIZE:
            return None
        return x, y, w, h


    @staticmethod
    def _detect_dlib(
        img: Image.Image,
    ) -> Optional[Tuple[int, int, int, int]]:
        """dlib HOG frontal detector (upsample 1); None when dlib is not installed."""

        try:
            import dlib  # type: ignore
        except ImportError:
            return None

        rects = dlib.get_frontal_face_detector()(np.asarray(img.convert("RGB")), 1)
        if len(rects) == 0:
            return None

        r = max(rects, key=lambda r: r.width() * r.height())
        x, y = max(0, r.left()), max(0, r.top())
        w = min(img.width, r.right()) - x
        h = min(img.height, r.bottom()) - y
        if w < settings.FACE_MIN_SIZE or h < settings.FACE_MIN_SIZE:
            return None
        return x, y, w, h


    @classmethod
    def detect_face_with_fallback(
        cls,
        img: Image.Image,
    ) -> Optional[Tuple[Tuple[int, int, int, int], str]]:
        """Returns ((x, y, w, h), detector) using haar -> yunet -> dlib, or None."""

        haar = cls.detect_largest_face(img)
        if haar is not None:
            # Sanity check: a Haar "face" under FACE_HAAR_MIN_FRACTION of the short side is
            # often background texture (a 70px shelf box on a 1920x2560 selfie). Prefer a
            # larger YuNet face in that case; changed 0 of 832 regression images.
            if max(haar[2], haar[3]) < settings.FACE_HAAR_MIN_FRACTION * min(img.size):
                yunet = cls._detect_yunet(img)
                if yunet is not None and max(yunet[2], yunet[3]) > max(haar[2], haar[3]):
                    return yunet, "yunet"
            return haar, "haar"

        for name, detect in (
            ("yunet", cls._detect_yunet),
            ("dlib", cls._detect_dlib),
        ):
            face = detect(img)
            if face is not None:
                return face, name

        return None


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
    def crop_image_with_detector(
        cls,
        img: Image.Image,
        fallback: bool = True,
    ) -> Optional[Tuple[Image.Image, str]]:
        """Returns (resized square face crop, detector name), or None if no face is found."""

        img = img.convert("RGB")

        if fallback:
            found = cls.detect_face_with_fallback(img)
        else:
            face = cls.detect_largest_face(img)
            found = None if face is None else (face, "haar")

        if found is None:
            return None

        face, detector = found

        box = cls._square_box(
            face,
            img.width,
            img.height,
        )

        size = int(settings.FACE_CROP_SIZE)

        crop = img.crop(box).resize(
            (size, size),
            Image.Resampling.LANCZOS,
        )
        return crop, detector


    @classmethod
    def crop_image(
        cls,
        img: Image.Image,
    ) -> Optional[Image.Image]:
        """Strict-Haar face crop (video frames keep the original detector), or None."""

        found = cls.crop_image_with_detector(img, fallback=False)
        return None if found is None else found[0]


    @classmethod
    def crop_face_file_with_detector(
        cls,
        src_path: Path,
        dst_path: Path,
    ) -> Tuple[Path, str]:
        """
        Crops the largest face of src_path into dst_path (PNG); returns (dst_path, detector).

        Raises MediaValidationError(NO_FACE_DETECTED) when no face is found.
        """

        with Image.open(src_path) as src:
            found = cls.crop_image_with_detector(src)

        if found is None:
            raise MediaValidationError(
                NO_FACE_DETECTED_CODE,
                NO_FACE_DETECTED_MESSAGE,
            )

        cropped, detector = found

        cropped.save(
            dst_path,
            format="PNG",
        )

        logger.info(
            "Face crop %s -> %s (%dx%d, detector=%s)",
            src_path.name,
            dst_path.name,
            cropped.width,
            cropped.height,
            detector,
        )

        return dst_path, detector


    @classmethod
    def crop_face_file(
        cls,
        src_path: Path,
        dst_path: Path,
    ) -> Path:
        """Crops the largest face of src_path into dst_path (PNG); raises NO_FACE_DETECTED."""

        return cls.crop_face_file_with_detector(src_path, dst_path)[0]
