from dataclasses import dataclass

@dataclass
class ModelDefinition:
    """
    A simple class for storing and loading model definitions
    """
    description: str
    vision: bool
    reasoning: bool
    enforce_reasoning: bool