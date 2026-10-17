"""Constants shared across the application."""

SYSTEM_TAG = "System"

# how much text the read_url tool will return from a single page
DEFAULT_URL_READER_MAX_CHARS: int = 20_000

# images sent as input are downscaled to at most this many pixels
# (width * height), unless config.yaml says otherwise: 2MP
DEFAULT_MAX_IMAGE_INPUT_PIXELS: int = 2_000_000
