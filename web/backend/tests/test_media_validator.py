import hashlib
import tempfile
import unittest

from pathlib import Path

from PIL import Image

from web.backend.app.config import settings
from web.backend.app.services.media_validator import (
    MediaValidationError,
    MediaValidator,
)


class MediaValidatorRegressionTests(
    unittest.TestCase
):

    def setUp(self):
        self.tmp = (
            tempfile.TemporaryDirectory()
        )

        self.root = Path(
            self.tmp.name
        )

        self.old_max_dimension = (
            settings.MAX_IMAGE_DIMENSION
        )


    def tearDown(self):
        settings.MAX_IMAGE_DIMENSION = (
            self.old_max_dimension
        )

        self.tmp.cleanup()


    def test_small_jpeg_is_not_reencoded(self):

        path = (
            self.root
            / "normal.jpg"
        )

        Image.new(
            "RGB",
            (320, 240),
            (80, 120, 160),
        ).save(
            path,
            format="JPEG",
            quality=90,
        )

        before = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()

        size = (
            MediaValidator
            .prepare_image_file(
                path
            )
        )

        after = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()

        self.assertEqual(
            size,
            (320, 240),
        )

        self.assertEqual(
            before,
            after,
            "Normal JPEG should remain byte-identical",
        )


    def test_mpo_is_canonicalized_to_jpeg(self):

        path = (
            self.root
            / "phone_photo.mpo"
        )

        primary = Image.new(
            "RGB",
            (320, 240),
            (200, 30, 30),
        )

        secondary = Image.new(
            "RGB",
            (320, 240),
            (30, 30, 200),
        )

        primary.save(
            path,
            format="MPO",
            save_all=True,
            append_images=[
                secondary
            ],
        )

        with Image.open(
            path
        ) as before:
            self.assertEqual(
                before.format,
                "MPO",
            )

            self.assertGreaterEqual(
                before.n_frames,
                2,
            )

        result = (
            MediaValidator
            .prepare_image_file(
                path
            )
        )

        self.assertEqual(
            result,
            (320, 240),
        )

        with Image.open(
            path
        ) as after:
            self.assertEqual(
                after.format,
                "JPEG",
            )

            self.assertEqual(
                getattr(
                    after,
                    "n_frames",
                    1,
                ),
                1,
            )


    def test_oversized_mpo_resizes_and_canonicalizes(self):

        # Small test dimensions for speed,
        # while exercising the same logic as
        # 4284x5712 -> 3072x4096.
        settings.MAX_IMAGE_DIMENSION = 256

        path = (
            self.root
            / "large_phone_photo.jpg"
        )

        primary = Image.new(
            "RGB",
            (428, 571),
            (100, 100, 100),
        )

        secondary = Image.new(
            "RGB",
            (64, 64),
            (10, 10, 10),
        )

        primary.save(
            path,
            format="MPO",
            save_all=True,
            append_images=[
                secondary
            ],
        )

        result = (
            MediaValidator
            .prepare_image_file(
                path
            )
        )

        self.assertEqual(
            result,
            (192, 256),
        )

        with Image.open(
            path
        ) as img:
            self.assertEqual(
                img.format,
                "JPEG",
            )

            self.assertEqual(
                img.size,
                (192, 256),
            )


    def test_large_regular_jpeg_preserves_ratio(self):

        settings.MAX_IMAGE_DIMENSION = 256

        path = (
            self.root
            / "large.jpg"
        )

        Image.new(
            "RGB",
            (428, 571),
            (125, 125, 125),
        ).save(
            path,
            format="JPEG",
        )

        result = (
            MediaValidator
            .prepare_image_file(
                path
            )
        )

        self.assertEqual(
            result,
            (192, 256),
        )


    def test_unknown_magic_is_rejected(self):

        path = (
            self.root
            / "fake.jpg"
        )

        path.write_bytes(
            b"this-is-not-an-image"
        )

        with self.assertRaises(
            MediaValidationError
        ) as ctx:

            MediaValidator.prepare_image_file(
                path
            )

        self.assertEqual(
            ctx.exception.code,
            "INVALID_MAGIC_BYTES",
        )


if __name__ == "__main__":
    unittest.main()
