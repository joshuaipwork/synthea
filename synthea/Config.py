import yaml


class Config:  
    """
    A simple class for storing and loading config.yaml
    """
    def __init__(self):
        """
        Load config.yaml and parse it into the class fields
        """
        with open("config.yaml", "r", encoding="utf-8") as file:
            loaded_file: dict[str, str] = yaml.safe_load(file)
        self.context_length: int = loaded_file["context_length"]
        self.max_new_tokens: int = loaded_file["max_new_tokens"]
        self.command_start_str: str = loaded_file["command_start_str"]
        self.system_prompt: str = loaded_file["system_prompt"]
        self.default_model: str = loaded_file["default_model"]

        # generation parameters
        self.temperature: float = loaded_file["temperature"]
        self.presence_penalty: float = loaded_file["presence_penalty"]
        self.frequency_penalty: float = loaded_file["frequency_penalty"]
        self.top_p: float = loaded_file["top_p"]
