import base64
import io
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image
from ..utils.logger import logger
from .face_cropper import FaceCropper


class VideoExtractor:
    """Extracts video frames adhering to DeepfakeBench uniform sampling protocol."""

    @staticmethod
    def encode_micro_thumbnail(img: Image.Image, size=(128, 128)) -> str:
        """Convert an in-memory image to a compact 128x128 JPEG Base64 string."""
        thumb = img.convert("RGB")
        thumb.thumbnail(size)
        buffer = io.BytesIO()
        thumb.save(buffer, format="JPEG", quality=70)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @classmethod
    def generate_micro_thumbnail(cls, image_path: Path, size=(128, 128)) -> Optional[str]:
        """Convert an image to a compact 128x128 JPEG Base64 string for zero-disk leak delivery."""
        try:
            with Image.open(image_path) as img:
                return cls.encode_micro_thumbnail(img, size)
        except Exception as e:
            logger.warning(f"Could not generate micro-thumbnail for {image_path}: {e}")
            return None

    @classmethod
    def extract_uniform_frames(
        cls,
        video_path: Path,
        output_dir: Path,
        target_num_frames: int = 32,
        crop_faces: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Extract up to target_num_frames evenly spaced across the duration of the video.

        Frames are stored as lossless PNG. With crop_faces=True, frame_path points to
        the face crop (see FaceCropper) and sampled frames without a detectable face
        are skipped; the thumbnail always shows the original frame.

        Returns: List of dicts with frame_index, timestamp_seconds, frame_path, thumbnail_b64.
        frame_index is the original sampled-frame index (0..N-1); skipped frames leave gaps.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []
        skipped_no_face = 0

        try:
            import cv2  # type: ignore
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video stream for {video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

            if total_frames <= 0:
                raise RuntimeError("Video has 0 readable frames.")

            num_samples = min(total_frames, target_num_frames)
            step = max(1, total_frames // num_samples)
            sampled_indices = [i * step for i in range(num_samples)]

            curr_frame = 0
            sample_ptr = 0

            while cap.isOpened() and sample_ptr < len(sampled_indices):
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                if curr_frame == sampled_indices[sample_ptr]:
                    timestamp = round(curr_frame / fps, 2)

                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)

                    if crop_faces:
                        model_img = FaceCropper.crop_image(pil_img)
                        frame_name = f"frame_{sample_ptr:03d}_face.png"
                    else:
                        model_img = pil_img
                        frame_name = f"frame_{sample_ptr:03d}.png"

                    if model_img is None:
                        skipped_no_face += 1
                        logger.info(
                            f"No face detected in sampled frame {sample_ptr} "
                            f"(t={timestamp}s) of {video_path.name}; skipping."
                        )
                    else:
                        frame_path = output_dir / frame_name
                        model_img.save(frame_path, format="PNG")

                        try:
                            thumb_b64 = cls.encode_micro_thumbnail(pil_img)
                        except Exception as e:
                            logger.warning(f"Could not generate micro-thumbnail for frame {sample_ptr}: {e}")
                            thumb_b64 = None

                        results.append({
                            "frame_index": sample_ptr,
                            "timestamp_seconds": timestamp,
                            "frame_path": frame_path,
                            "thumbnail_b64": thumb_b64,
                        })

                    sample_ptr += 1

                curr_frame += 1

            cap.release()

        except ImportError:
            # Fallback for mock environments without OpenCV installed
            logger.warning("cv2 is not installed. Generating simulated frame placeholders.")
            for i in range(min(16, target_num_frames)):
                frame_path = output_dir / f"sim_frame_{i:03d}.png"
                img = Image.new("RGB", (256, 256), color=(i * 15, 100, 200))
                img.save(frame_path, format="PNG")
                thumb_b64 = cls.generate_micro_thumbnail(frame_path)
                results.append({
                    "frame_index": i,
                    "timestamp_seconds": round(i * 0.4, 2),
                    "frame_path": frame_path,
                    "thumbnail_b64": thumb_b64,
                })

        logger.info(
            f"Extracted {len(results)} uniform frames from {video_path.name}"
            + (f" ({skipped_no_face} skipped: no face)" if crop_faces else "")
        )
        return results
