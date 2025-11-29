from dataclasses import dataclass


@dataclass
class ModelFormatDefinition:
    """
    A simple class for storing and loading model definitions
    """
    reasoning_start_tag: str
    reasoning_end_tag: str
    solution_start_tag: str
    solution_end_tag: str