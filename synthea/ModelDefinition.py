from dataclasses import dataclass

from synthea import ModelFormatDefinition


@dataclass
class ModelDefinition:
    """
    A simple class for storing and loading model definitions
    """
    description: str
    vision: bool
    reasoning: bool
    enforce_reasoning: bool
    template: ModelFormatDefinition