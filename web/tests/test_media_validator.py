import io
import pytest
from PIL import Image
from web.backend.app.services.media_validator import (
    MediaValidator,
    MediaValidationError,
)


def test_media_validator_valid_image(tmp_path):
    img_path = tmp_path / "valid.jpg"
    img = Image.new("RGB", (200, 200), color="blue")
    img.save(img_path, format="JPEG")

    width, height = MediaValidator.validate_image_file(img_path)
    assert width == 200
    assert height == 200


def test_media_validator_rejects_empty_file(tmp_path):
    empty_path = tmp_path / "empty.png"
    empty_path.write_bytes(b"")

    with pytest.raises(MediaValidationError) as exc:
        MediaValidator.validate_image_file(empty_path)
    assert "EMPTY" in exc.value.code


def test_media_validator_rejects_invalid_magic_bytes(tmp_path):
    fake_img = tmp_path / "fake.jpg"
    fake_img.write_bytes(b"NOT_A_REAL_IMAGE_HEADER_1234567890")

    with pytest.raises(MediaValidationError) as exc:
        MediaValidator.validate_image_file(fake_img)
    assert "MAGIC_BYTES" in exc.value.code or "INVALID" in exc.value.code


def test_media_validator_rejects_excessive_dimensions(tmp_path):
    huge_path = tmp_path / "huge.png"
    # Create an image exceeding max dimensions 4096
    huge_img = Image.new("RGB", (4097, 100), color="red")
    huge_img.save(huge_path, format="PNG")

    with pytest.raises(MediaValidationError) as exc:
        MediaValidator.validate_image_file(huge_path)
    assert "DIMENSION" in exc.value.code
