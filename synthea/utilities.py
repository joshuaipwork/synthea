import ast
import os
import re
import json
import logging
import datetime
import xml.etree.ElementTree as ET

from logging.handlers import RotatingFileHandler

from synthea.exceptions import InvalidImageDimensionsException

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
script_dir = os.path.dirname(os.path.abspath(__file__))
now = datetime.datetime.now()
log_folder = os.path.join(script_dir, "inference_logs")
os.makedirs(log_folder, exist_ok=True)
log_file_path = os.path.join(
    log_folder, f"function-calling-inference_{now.strftime('%Y-%m-%d_%H-%M-%S')}.log"
)
# Use RotatingFileHandler from the logging.handlers module
file_handler = RotatingFileHandler(log_file_path, maxBytes=0, backupCount=0)
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s", datefmt="%Y-%m-%d:%H:%M:%S")
file_handler.setFormatter(formatter)

inference_logger = logging.getLogger("function-calling-inference")
inference_logger.addHandler(file_handler)

def parse_dimensions(self, dimensions: str) -> tuple[int, int]:
    """
    Parses a dimension string of the form '1000x1000' into a width and height,
    applying the limitations of the ComfyUI empty latent image node.
    """
    values = dimensions.split('x')
    if len(values) != 2:
        raise InvalidImageDimensionsException(f"Couldn't parse '{dimensions}' into a width and height")

    width, height = int(values[0]), int(values[1])

    if width < 16 or height < 16:
        raise InvalidImageDimensionsException(f"Invalid dimensions {dimensions} - minimum size is 16x16")
    if width > 16384 or height > 16384:
        raise InvalidImageDimensionsException(f"Invalid dimensions {dimensions} - maximum size is 16384x16384")
    if width * height > self.config.image_maximum_pixels:
        raise InvalidImageDimensionsException(
            f"Dimensions {dimensions} exceed the maximum pixel count of {self.config.image_maximum_pixels}")

    return width, height

def split_text(text, max_length=1800) -> list[str]:
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]


def split_text_smartly(text, max_length=2000) -> list[str]:
    """
    Split the text into pieces of at most max_length characters. 
    The function prioritizes splitting at paragraph breaks, then periods, and finally spaces.
    It tries to keep the pieces about equally sized.
    
    Args:
    - text (str): The input text.
    - max_length (int): The maximum length of each piece. Default is 2000.

    Returns:
    - List[str]: List of text pieces.
    """
    
    # Split by paragraph first
    paragraphs = text.split('\n')
    pieces = []
    current_piece = ""

    for paragraph in paragraphs:
        # If the current piece + the new paragraph is too long
        if len(current_piece + paragraph) > max_length:
            # If the current piece is not empty, add it to the pieces
            if current_piece:
                pieces.append(current_piece)
                current_piece = ""
            
            # If the paragraph itself is longer than max_length, split it further
            while len(paragraph) > max_length:
                # Find the last period within max_length
                split_point = paragraph.rfind('.', 0, max_length)
                
                # If there's no period, find the last space within max_length
                if split_point == -1:
                    split_point = paragraph.rfind(' ', 0, max_length)
                
                # If there's neither a period nor a space, just split at max_length
                if split_point == -1:
                    split_point = max_length
                
                # Add the split part to the pieces and remove it from the paragraph
                pieces.append(paragraph[:split_point + 1].strip())
                paragraph = paragraph[split_point + 1:].strip()
            
            # Add the remainder of the paragraph to the current piece
            current_piece = paragraph
        else:
            # If the paragraph can be added to the current piece without exceeding max_length
            if current_piece:
                current_piece += "\n" + paragraph
            else:
                current_piece = paragraph

    # If there's any remaining text in the current piece, add it to the pieces
    if current_piece:
        pieces.append(current_piece)

    return pieces
    