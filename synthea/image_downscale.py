"""Automatic downscaling of oversized image attachments.

Users send whatever their camera produced - a modern phone photo is 12MP or
more - and the API provider rejects those requests outright, which forces
people to resize photos by hand before they can chat. Instead, any image
covering more pixels than we allow is scaled down (keeping its aspect ratio)
before it is encoded into the request, so the model still sees the whole
image and the user doesn't have to do anything.

Images already within the limit are handed back byte for byte unchanged, and
anything that can't be read or re-encoded is returned untouched: a damaged
image must never break the message it was attached to.
"""

import math
from io import BytesIO

from PIL import Image, ImageOps

from synthea.constants import DEFAULT_MAX_IMAGE_INPUT_PIXELS
from synthea.utilities import inference_logger

# the limit applied when there is no config in hand (2MP); the running bot
# uses ``max_image_input_pixels`` from config.yaml, which may be larger, or 0
# to never resize anything
MAX_UPLOAD_IMAGE_PIXELS = DEFAULT_MAX_IMAGE_INPUT_PIXELS

# LANCZOS, whichever alias the installed Pillow exposes it under
_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS

# formats that accept an embedded colour profile when writing
_ICC_FORMATS = {"JPEG", "PNG", "WEBP"}


def downscale_image(
    image_data: bytes, max_pixels: int = MAX_UPLOAD_IMAGE_PIXELS,
) -> bytes:
    """Scales ``image_data`` down so that it covers at most ``max_pixels`` pixels.

    If nothing else is passed, images larger than
    ``DEFAULT_MAX_IMAGE_INPUT_PIXELS`` - 2MP - are the ones considered
    oversized; the bot passes the configured ``max_image_input_pixels`` here
    instead, and a limit of 0 (or less) disables the resizing entirely.

    The aspect ratio is preserved, so a 4000x3000 photo arrives at about
    1632x1224 rather than being cropped or stretched, and the image keeps its
    original format, so the mime type of the payload doesn't change.

    Args:
        image_data: The raw bytes of the image.
        max_pixels: Width * height that may be sent onwards. Images larger
            than this (and only those) are resampled.

    Returns:
        The downscaled image bytes, or the input unchanged when the image is
        already within the limit or cannot be processed.

    """
    if not image_data or max_pixels <= 0:
        return image_data

    try:
        return _downscale(image_data, max_pixels)
    except Exception as error:  # never let a bad image break the conversation
        inference_logger.warning(
            f"downscale_image: couldn't downscale the image, sending it as is: {error}",
        )
        return image_data


def _downscale(image_data: bytes, max_pixels: int) -> bytes:
    with Image.open(BytesIO(image_data)) as source:
        image_format = source.format
        if not image_format:
            return image_data

        # a camera stores the rotation in EXIF rather than in the pixels, and
        # providers strip EXIF when they decode the payload, so bake the
        # orientation into the rescaled image
        image = ImageOps.exif_transpose(source)
        width, height = image.size

        if width * height <= max_pixels:
            # already small enough: give back the untouched original
            return image_data

        scale = math.sqrt(max_pixels / (width * height))
        # truncating instead of rounding keeps the scaled width * height at or
        # below max_pixels for every aspect ratio
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))

        # animated images are reduced to their current (first) frame; models
        # only ever look at a single frame of these anyway
        resized = image.resize((new_width, new_height), resample=_LANCZOS)
        resized = _saveable_mode(resized, image_format)

        save_args = {}
        profile = source.info.get("icc_profile")
        if profile and image_format in _ICC_FORMATS:
            save_args["icc_profile"] = profile

        buffer = BytesIO()
        resized.save(buffer, format=image_format, **save_args)

    inference_logger.info(
        f"downscaled an oversized attachment from {width}x{height} "
        f"to {new_width}x{new_height} ({image_format})",
    )
    return buffer.getvalue()


def _saveable_mode(image: Image.Image, image_format: str) -> Image.Image:
    """Adjusts the pixel mode so the target format accepts it.

    JPEG only writes 8 bit grayscale, RGB and CMYK, so anything carrying an
    alpha channel or a palette has to be flattened first. Modes that don't
    convert are returned as they are: the save then fails and the caller
    falls back to the original image.
    """
    if image_format != "JPEG" or image.mode in ("RGB", "L", "CMYK"):
        return image
    for mode in ("RGB", "L"):
        try:
            # visible colours are kept; any transparency composites onto black
            return image.convert(mode)
        except (ValueError, OSError, TypeError):
            continue
    return image
