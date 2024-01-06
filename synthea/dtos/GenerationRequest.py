import discord

class GenerationRequest:
    """
    
    """
    def __init__(self, response_index: int, context: str = "") -> None:
        # the response to update
        self.response_index: int = response_index
        self.context: str = context
