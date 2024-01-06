import discord

class ResponseUpdate:
    """
    
    """
    def __init__(self, response_index: str, message_is_completed: bool, new_message: str = "") -> None:
        # the response to update
        self.response_index: int = response_index
        self.message_is_completed: bool = message_is_completed
        self.new_message: str = new_message
