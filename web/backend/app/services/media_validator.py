from pathlib import Path
from typing import Tuple

from PIL import Image, ImageOps

from ..config import settings


class MediaValidationError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class MediaValidator:
    """
    Uploaded-media validator.

    Image policy:
      - accepted containers: JPEG-family, PNG, WEBP
      - MPO is accepted because it is JPEG-family
      - MPO is canonicalized to its primary frame as JPEG
      - oversized images are downscaled preserving aspect ratio
      - no crop, no stretching
      - animated PNG/WEBP is rejected instead of silently picking
        an arbitrary frame

    Video policy:
      - MP4/MOV ISO-BMFF, AVI, MKV
      - real decode check in LIVE mode
    """

    SIGNATURES = {
        "jpeg": b"\xFF\xD8\xFF",
        "png": b"\x89PNG\r\n\x1a\n",
        "mkv": b"\x1A\x45\xDF\xA3",
    }


    # ========================================================
    # IMAGE HELPERS
    # ========================================================

    @classmethod
    def _detect_image_container(
        cls,
        file_path: Path,
    ) -> str:

        with open(file_path, "rb") as f:
            header = f.read(16)

        if header.startswith(
            cls.SIGNATURES["jpeg"]
        ):
            return "jpeg"

        if header.startswith(
            cls.SIGNATURES["png"]
        ):
            return "png"

        if (
            header.startswith(b"RIFF")
            and header[8:12] == b"WEBP"
        ):
            return "webp"

        raise MediaValidationError(
            "INVALID_MAGIC_BYTES",
            (
                "File format signature not recognized. "
                "Supported images: JPEG/JPG/MPO, PNG, WEBP."
            ),
        )


    @classmethod
    def _inspect_image(
        cls,
        file_path: Path,
    ) -> Tuple[
        str,
        str,
        int,
        int,
        int,
    ]:
        """
        Returns:
          container,
          Pillow decoded format,
          width,
          height,
          n_frames
        """

        if (
            not file_path.exists()
            or file_path.stat().st_size == 0
        ):
            raise MediaValidationError(
                "EMPTY_FILE",
                "Uploaded file is empty (0 bytes).",
            )

        size_mb = (
            file_path.stat().st_size
            / (1024 * 1024)
        )

        if (
            size_mb
            > settings.MAX_IMAGE_SIZE_MB
        ):
            raise MediaValidationError(
                "IMAGE_SIZE_EXCEEDED",
                (
                    f"Image size ({size_mb:.2f}MB) "
                    "exceeds maximum allowed "
                    f"({settings.MAX_IMAGE_SIZE_MB}MB)."
                ),
            )

        container = (
            cls._detect_image_container(
                file_path
            )
        )

        try:

            # Integrity check.
            with Image.open(
                file_path
            ) as verify_img:
                verify_img.verify()

            # Reopen because verify() invalidates decoder state.
            with Image.open(
                file_path
            ) as img:

                width, height = img.size

                decoded_format = (
                    img.format or ""
                ).upper()

                n_frames = int(
                    getattr(
                        img,
                        "n_frames",
                        1,
                    )
                    or 1
                )

        except MediaValidationError:
            raise

        except Exception as exc:

            raise MediaValidationError(
                "CORRUPTED_IMAGE",
                (
                    "Image file is corrupted "
                    "and cannot be decoded: "
                    f"{exc}"
                ),
            )

        # MPO is explicitly allowed:
        # first image is the primary JPEG frame.
        if (
            n_frames > 1
            and decoded_format != "MPO"
        ):
            raise MediaValidationError(
                "ANIMATED_IMAGE_UNSUPPORTED",
                (
                    "Animated/multi-frame PNG or WEBP "
                    "is not supported by the single-image detector. "
                    "Please upload one static frame."
                ),
            )

        return (
            container,
            decoded_format,
            width,
            height,
            n_frames,
        )


    @classmethod
    def validate_image_file(
        cls,
        file_path: Path,
    ) -> Tuple[int, int]:

        (
            _container,
            _decoded_format,
            width,
            height,
            _n_frames,
        ) = cls._inspect_image(
            file_path
        )

        if (
            width
            > settings.MAX_IMAGE_DIMENSION
            or height
            > settings.MAX_IMAGE_DIMENSION
        ):
            raise MediaValidationError(
                "IMAGE_DIMENSION_EXCEEDED",
                (
                    f"Image resolution {width}x{height} "
                    "exceeds limit of "
                    f"{settings.MAX_IMAGE_DIMENSION}px."
                ),
            )

        return width, height


    @staticmethod
    def _pil_output_format(
        container: str,
    ) -> str:
        """
        IMPORTANT:
        Output format follows container magic bytes.

        Example:
          JPEG container + Pillow decoded format MPO
          -> output JPEG

        This prevents the previous MPO rejection bug.
        """

        mapping = {
            "jpeg": "JPEG",
            "png": "PNG",
            "webp": "WEBP",
        }

        try:
            return mapping[container]

        except KeyError:

            raise MediaValidationError(
                "INVALID_IMAGE_FORMAT",
                (
                    "Unsupported image container: "
                    f"{container}"
                ),
            )


    @classmethod
    def prepare_image_file(
        cls,
        file_path: Path,
    ) -> Tuple[int, int]:
        """
        Prepare an image for the single-image inference pipeline.

        Normalization occurs only when required:
          1. the image is MPO, or
          2. its longest side exceeds MAX_IMAGE_DIMENSION.

        MPO behavior:
          - use primary frame (frame 0)
          - canonicalize to standard JPEG

        Oversized-image behavior:
          - preserve aspect ratio
          - no crop
          - no stretching
          - deterministic LANCZOS downsampling

        This is web-demo preprocessing and does not alter model
        thresholds, calibration, WFS, Router4, or final-score logic.
        """

        (
            container,
            decoded_format,
            width,
            height,
            _n_frames,
        ) = cls._inspect_image(
            file_path
        )

        needs_resize = (
            width
            > settings.MAX_IMAGE_DIMENSION
            or height
            > settings.MAX_IMAGE_DIMENSION
        )

        needs_mpo_canonicalization = (
            decoded_format == "MPO"
        )

        if (
            not needs_resize
            and not needs_mpo_canonicalization
        ):
            # Byte-for-byte file remains untouched.
            return width, height

        try:

            with Image.open(
                file_path
            ) as src:

                # MPO primary image.
                if decoded_format == "MPO":
                    src.seek(0)

                # Detach from the multi-frame source decoder.
                img = src.copy()

            # Respect orientation when a rewrite is already required.
            img = ImageOps.exif_transpose(
                img
            )

            current_w, current_h = (
                img.size
            )

            if (
                current_w
                > settings.MAX_IMAGE_DIMENSION
                or current_h
                > settings.MAX_IMAGE_DIMENSION
            ):

                max_dim = int(
                    settings.MAX_IMAGE_DIMENSION
                )

                scale = min(
                    max_dim / current_w,
                    max_dim / current_h,
                )

                new_w = max(
                    1,
                    int(
                        round(
                            current_w
                            * scale
                        )
                    ),
                )

                new_h = max(
                    1,
                    int(
                        round(
                            current_h
                            * scale
                        )
                    ),
                )

                img = img.resize(
                    (
                        new_w,
                        new_h,
                    ),
                    Image.Resampling.LANCZOS,
                )

            fmt = cls._pil_output_format(
                container
            )

            save_kwargs = {}

            if fmt == "JPEG":

                # MPO and regular JPEG-family images become
                # a single standard JPEG image.
                if img.mode not in {
                    "RGB",
                    "L",
                }:
                    img = img.convert(
                        "RGB"
                    )

                save_kwargs.update(
                    quality=95,
                    subsampling=0,
                    optimize=True,
                )

            elif fmt == "PNG":

                save_kwargs.update(
                    optimize=True,
                )

            elif fmt == "WEBP":

                save_kwargs.update(
                    quality=95,
                    method=6,
                )

            tmp_path = (
                file_path.parent
                / (
                    file_path.name
                    + ".normalized.tmp"
                )
            )

            img.save(
                tmp_path,
                format=fmt,
                **save_kwargs,
            )

            # Atomic replacement inside the same request workspace.
            tmp_path.replace(
                file_path
            )

        except MediaValidationError:
            raise

        except Exception as exc:

            raise MediaValidationError(
                "IMAGE_NORMALIZATION_FAILED",
                (
                    "Image could not be normalized: "
                    f"{exc}"
                ),
            )

        # Full post-normalization validation.
        return cls.validate_image_file(
            file_path
        )


    # ========================================================
    # VIDEO
    # ========================================================

    @classmethod
    def validate_video_file(
        cls,
        file_path: Path,
    ) -> Tuple[
        float,
        int,
        float,
    ]:

        if (
            not file_path.exists()
            or file_path.stat().st_size == 0
        ):
            raise MediaValidationError(
                "EMPTY_FILE",
                "Uploaded video is empty (0 bytes).",
            )

        size_mb = (
            file_path.stat().st_size
            / (1024 * 1024)
        )

        if (
            size_mb
            > settings.MAX_VIDEO_SIZE_MB
        ):
            raise MediaValidationError(
                "VIDEO_SIZE_EXCEEDED",
                (
                    f"Video size ({size_mb:.2f}MB) "
                    "exceeds maximum allowed "
                    f"({settings.MAX_VIDEO_SIZE_MB}MB)."
                ),
            )

        with open(
            file_path,
            "rb",
        ) as f:
            header = f.read(32)

        # MP4 and QuickTime MOV are both ISO-BMFF and normally
        # contain an ftyp box near the beginning.
        is_iso_bmff = (
            b"ftyp"
            in header[:16]
        )

        is_avi = (
            header.startswith(b"RIFF")
            and header[8:12] == b"AVI "
        )

        is_mkv = header.startswith(
            cls.SIGNATURES["mkv"]
        )

        if not (
            is_iso_bmff
            or is_avi
            or is_mkv
        ):
            raise MediaValidationError(
                "INVALID_VIDEO_FORMAT",
                (
                    "Video container format signature "
                    "not recognized. Supported formats: "
                    "MP4, MOV, AVI, MKV."
                ),
            )

        duration = 0.0
        frame_count = 0
        fps = 25.0

        try:

            import cv2  # type: ignore

            cap = cv2.VideoCapture(
                str(file_path)
            )

            if not cap.isOpened():
                raise MediaValidationError(
                    "CORRUPTED_VIDEO",
                    "Video stream could not be opened.",
                )

            fps = (
                cap.get(
                    cv2.CAP_PROP_FPS
                )
                or 25.0
            )

            frame_count = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )

            duration = (
                frame_count / fps
                if fps > 0
                else 0.0
            )

            ret, _ = cap.read()
            cap.release()

            if (
                not ret
                or frame_count <= 0
            ):
                raise MediaValidationError(
                    "CORRUPTED_VIDEO",
                    (
                        "Failed to decode any valid "
                        "frames from video."
                    ),
                )

        except ImportError:

            if (
                settings.RUN_MODE
                == "live"
            ):
                raise MediaValidationError(
                    "VIDEO_RUNTIME_UNAVAILABLE",
                    (
                        "OpenCV is unavailable in LIVE mode; "
                        "refusing simulated video metadata."
                    ),
                )

            duration = 10.0
            frame_count = 250
            fps = 25.0

        if (
            duration
            > settings.MAX_VIDEO_DURATION_SEC
        ):
            raise MediaValidationError(
                "VIDEO_DURATION_EXCEEDED",
                (
                    f"Video duration ({duration:.1f}s) "
                    "exceeds maximum allowed "
                    f"({settings.MAX_VIDEO_DURATION_SEC}s)."
                ),
            )

        return (
            duration,
            frame_count,
            fps,
        )
