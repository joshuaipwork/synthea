"""Utilities for embedding and retrieving image-generation metadata in PNG files.

ComfyUI already embeds the full workflow as JSON in the "prompt" key of the PNG
metadata, but that payload is bulky and hard to reason about. We add a compact
"prompt_simple" key holding the positive prompt (and optionally the negative
prompt) so that, when the responding LLM has no vision, the bot can read the
image back and tell the model what it previously generated. This lets the model
refine its own images across multi-turn conversations.

PNG text chunks (tEXt/iTXt) survive Discord's CDN, so the metadata round-trips
through messages: the bot embeds the prompt when generating, and reads it back
when the image reappears in the chat history.
"""

import json
from io import BytesIO

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from synthea.utilities import inference_logger

# key we write for the compact positive prompt
PROMPT_SIMPLE_KEY = "prompt_simple"
# key we write for the compact negative prompt
NEGATIVE_PROMPT_KEY = "negative_prompt"
# key ComfyUI writes with the full workflow JSON
COMFYUI_PROMPT_KEY = "prompt"


def embed_prompt_in_image(
    image_data: bytes, positive_prompt: str, negative_prompt: str = "",
) -> bytes:
    """Embeds the generation prompts into a PNG's metadata, preserving any
    existing metadata (e.g. ComfyUI's "prompt" and "workflow" keys).

    Returns the original bytes unchanged if the image isn't a PNG or can't be
    processed, so a metadata failure never breaks image delivery.
    """
    try:
        image = Image.open(BytesIO(image_data))
    except Exception:
        inference_logger.warning(
            "embed_prompt_in_image: couldn't open image, returning it unchanged",
        )
        return image_data

    if image.format != "PNG":
        inference_logger.warning(
            f"embed_prompt_in_image: image format is {image.format}, not PNG; "
            "returning it unchanged",
        )
        return image_data

    png_info = PngInfo()
    for key, value in image.text.items():
        try:
            png_info.add_text(key, value)
        except (UnicodeEncodeError, ValueError):
            # skip metadata we can't store in a tEXt chunk rather than
            # dropping the whole image
            inference_logger.warning(
                f"embed_prompt_in_image: skipping existing metadata key {key!r}",
            )

    _add_text_fallback(png_info, PROMPT_SIMPLE_KEY, positive_prompt)
    if negative_prompt:
        _add_text_fallback(png_info, NEGATIVE_PROMPT_KEY, negative_prompt)

    buffer = BytesIO()
    try:
        image.save(buffer, format="PNG", pnginfo=png_info)
    except Exception:
        inference_logger.warning(
            "embed_prompt_in_image: failed to re-save image, returning it unchanged",
        )
        return image_data
    return buffer.getvalue()


def _add_text_fallback(png_info: PngInfo, key: str, value: str) -> None:
    """Adds text to a PngInfo, preferring tEXt (proven to survive Discord's CDN)
    and falling back to iTXt for values with characters outside latin-1
    (e.g. emoji). Both are readable back via Image.text.
    """
    try:
        png_info.add_text(key, value)
    except (UnicodeEncodeError, ValueError):
        png_info.add_itxt(key, value)


def extract_prompt_from_image(image_data: bytes) -> str | None:
    """Extracts the positive generation prompt from a PNG's metadata.

    Checks the compact "prompt_simple" key first, then falls back to parsing
    ComfyUI's full workflow JSON in the "prompt" key (so images generated
    before prompt embedding existed can still be understood).

    Returns None if no prompt can be found.
    """
    try:
        image = Image.open(BytesIO(image_data))
    except Exception:
        return None

    if image.format != "PNG":
        return None

    text = image.text or {}
    simple_prompt = text.get(PROMPT_SIMPLE_KEY)
    if simple_prompt:
        return simple_prompt

    comfy_prompt = text.get(COMFYUI_PROMPT_KEY)
    if comfy_prompt:
        try:
            workflow = json.loads(comfy_prompt)
        except (json.JSONDecodeError, TypeError):
            return None
        return _extract_prompt_from_workflow(workflow)

    return None


def _extract_prompt_from_workflow(workflow: dict) -> str | None:
    """Finds the positive prompt in a ComfyUI workflow dict.

    The positive prompt is the text input of the node referenced by a sampler's
    "positive" input. If no such reference exists, falls back to any
    CLIPTextEncode node that isn't referenced as a negative.
    """
    positive_ids: set[str] = set()
    negative_ids: set[str] = set()
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {})
        for key, ids in (("positive", positive_ids), ("negative", negative_ids)):
            ref = inputs.get(key)
            if isinstance(ref, list) and isinstance(ref[0], str):
                ids.add(ref[0])

    for node_id in positive_ids:
        node = workflow.get(node_id)
        text = node.get("inputs", {}).get("text") if isinstance(node, dict) else None
        if isinstance(text, str) and text.strip():
            return text

    for node_id, node in workflow.items():
        if (
            isinstance(node, dict)
            and node.get("class_type") == "CLIPTextEncode"
            and node_id not in negative_ids
        ):
            text = node.get("inputs", {}).get("text")
            if isinstance(text, str) and text.strip():
                return text

    return None
