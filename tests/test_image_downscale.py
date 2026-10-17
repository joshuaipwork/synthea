"""Tests for the automatic downscaling of oversized image attachments.

The API provider rejects requests containing images above its size limit,
which forced users to resize photos by hand. The bot now rescales anything
over the configured ``max_image_input_pixels`` limit (2MP by default) on its
way into the request, while images already inside the limit must reach the
model byte for byte unchanged.
"""

# pylint: disable=missing-function-docstring

import base64
from io import BytesIO
from pathlib import Path

import pytest
import yaml
from PIL import Image
from PIL.ExifTags import Base as ExifBase

from synthea.config import Config
from synthea.context_manager import ContextManager
from synthea.image_downscale import MAX_UPLOAD_IMAGE_PIXELS, downscale_image

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def context_manager() -> ContextManager:
    return ContextManager(bot_user_id=0)


def _image_bytes(width, height, image_format="JPEG", mode="RGB") -> bytes:
    buffer = BytesIO()
    Image.new(mode, (width, height)).save(buffer, format=image_format)
    return buffer.getvalue()


def _dimensions(data: bytes) -> tuple[int, int]:
    with Image.open(BytesIO(data)) as image:
        return image.size


def _format(data: bytes) -> str:
    with Image.open(BytesIO(data)) as image:
        return image.format


def _write_config(tmp_path, monkeypatch, **overrides) -> None:
    """Points Config() at a copy of the repository config, with overrides."""
    with (REPO_ROOT / "config.yaml").open(encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    for key, value in overrides.items():
        if value is None:
            settings.pop(key, None)
        else:
            settings[key] = value
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(settings), encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)


@pytest.fixture()
def oversized_image() -> bytes:
    # 12MP, three times over the limit
    return _image_bytes(4000, 3000)


class TestDownscaleImage:
    def test_threshold_is_two_megapixels(self):
        assert MAX_UPLOAD_IMAGE_PIXELS == 2_000_000

    def test_oversized_image_is_downscaled_to_the_limit(self, oversized_image):
        width, height = _dimensions(downscale_image(oversized_image))

        assert width * height <= MAX_UPLOAD_IMAGE_PIXELS
        # it was an oversized image, so it really should have been shrunk
        assert (width, height) != (4000, 3000)

    def test_aspect_ratio_is_preserved(self, oversized_image):
        width, height = _dimensions(downscale_image(oversized_image))

        # 4000x3000 is 4:3 -- downscaling must squish, not crop or stretch
        assert width / height == pytest.approx(4 / 3, rel=0.01)

    def test_format_is_preserved(self, oversized_image):
        assert _format(downscale_image(oversized_image)) == "JPEG"

    @pytest.mark.parametrize(
        ("width", "height"),
        # a 1600x1200 photo and a 800x600 photo are both well under 2MP
        [(1600, 1200), (800, 600), (64, 64)],
    )
    def test_images_within_the_limit_are_byte_identical(self, width, height):
        original = _image_bytes(width, height)

        assert downscale_image(original) == original

    def test_image_exactly_at_the_limit_is_untouched(self):
        original = _image_bytes(2000, 1000)

        assert downscale_image(original) == original

    def test_just_over_the_limit_is_downscaled(self):
        width, height = _dimensions(downscale_image(_image_bytes(2000, 1001)))

        assert width * height <= MAX_UPLOAD_IMAGE_PIXELS

    def test_huge_portrait_image_is_downscaled(self):
        width, height = _dimensions(downscale_image(_image_bytes(1080, 9600)))

        assert width * height <= MAX_UPLOAD_IMAGE_PIXELS
        # the tall aspect ratio survives the rescale
        assert height > width

    def test_png_keeps_its_format_and_alpha_channel(self):
        downscaled = downscale_image(_image_bytes(3000, 3000, mode="RGBA", image_format="PNG"))

        assert _format(downscaled) == "PNG"
        with Image.open(BytesIO(downscaled)) as image:
            assert image.mode == "RGBA"
            assert image.size[0] * image.size[1] <= MAX_UPLOAD_IMAGE_PIXELS

    def test_gif_is_downscaled(self):
        original = _image_bytes(3000, 2000, image_format="GIF", mode="P")

        downscaled = downscale_image(original)

        assert _format(downscaled) == "GIF"
        assert _dimensions(downscaled) != (3000, 2000)

    @pytest.mark.parametrize("data", [b"", b"not an image", b"\x89PNG broken"])
    def test_unreadable_data_is_returned_unchanged(self, data):
        # a bad image must never break the message it was attached to
        assert downscale_image(data) == data

    def test_exif_rotation_is_applied_before_rescaling(self):
        image = Image.new("RGB", (4000, 2000))
        exif = image.getexif()
        # orientation 6: stored sideways, displayed portrait
        exif[ExifBase.Orientation] = 6
        buffer = BytesIO()
        image.save(buffer, format="JPEG", exif=exif)

        width, height = _dimensions(downscale_image(buffer.getvalue()))

        assert width * height <= MAX_UPLOAD_IMAGE_PIXELS
        assert height > width

    def test_custom_pixel_limit_is_honoured(self, oversized_image):
        width, height = _dimensions(
            downscale_image(oversized_image, max_pixels=500_000),
        )

        assert width * height <= 500_000


class TestImageToBase64:
    def test_oversized_image_is_downscaled_in_the_payload(
        self, context_manager, tmp_path,
    ):
        path = tmp_path / "big.jpg"
        Image.new("RGB", (4000, 3000)).save(path, format="JPEG")

        header, payload = context_manager.image_to_base64(str(path)).split(",", 1)

        assert header == "data:image/jpeg;base64"
        width, height = _dimensions(base64.b64decode(payload))
        assert width * height <= MAX_UPLOAD_IMAGE_PIXELS

    def test_small_image_payload_keeps_the_original_bytes(
        self, context_manager, tmp_path,
    ):
        path = tmp_path / "small.jpg"
        Image.new("RGB", (640, 480)).save(path, format="JPEG")
        original = path.read_bytes()

        url = context_manager.image_to_base64(str(path))

        assert url == (
            f"data:image/jpeg;base64,{base64.b64encode(original).decode('utf-8')}"
        )


class TestConfiguredLimit:
    def test_repository_config_loads_the_default(self, config):
        assert config.max_image_input_pixels == 2_000_000

    def test_missing_key_falls_back_to_two_megapixels(self, tmp_path, monkeypatch):
        _write_config(tmp_path, monkeypatch, max_image_input_pixels=None)

        assert Config().max_image_input_pixels == MAX_UPLOAD_IMAGE_PIXELS

    @pytest.mark.parametrize("limit", [500_000, 16_000_000, 0])
    def test_configured_value_is_read(self, tmp_path, monkeypatch, limit):
        _write_config(tmp_path, monkeypatch, max_image_input_pixels=limit)

        assert Config().max_image_input_pixels == limit

    def test_a_larger_limit_sends_the_image_untouched(
        self, context_manager, tmp_path, monkeypatch,
    ):
        # 12MP is only rejected because 2MP is too small a limit for this
        # provider; a configured 16MP limit means no resizing at all
        _write_config(tmp_path, monkeypatch, max_image_input_pixels=16_000_000)
        path = tmp_path / "photo.jpg"
        Image.new("RGB", (4000, 3000)).save(path, format="JPEG")

        _, payload = context_manager.image_to_base64(str(path)).split(",", 1)

        assert base64.b64decode(payload) == path.read_bytes()

    def test_a_smaller_limit_rescales_further(
        self, context_manager, tmp_path, monkeypatch,
    ):
        _write_config(tmp_path, monkeypatch, max_image_input_pixels=500_000)
        path = tmp_path / "photo.jpg"
        Image.new("RGB", (4000, 3000)).save(path, format="JPEG")

        _, payload = context_manager.image_to_base64(str(path)).split(",", 1)
        width, height = _dimensions(base64.b64decode(payload))

        assert width * height <= 500_000

    def test_a_zero_limit_never_resizes(
        self, context_manager, tmp_path, monkeypatch,
    ):
        _write_config(tmp_path, monkeypatch, max_image_input_pixels=0)
        path = tmp_path / "photo.jpg"
        Image.new("RGB", (4000, 3000)).save(path, format="JPEG")

        _, payload = context_manager.image_to_base64(str(path)).split(",", 1)

        assert base64.b64decode(payload) == path.read_bytes()