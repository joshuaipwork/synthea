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
        self.bot_name: str = loaded_file["bot_name"]
        self.chat_template: str = loaded_file["chat_template"]

        # generation parameters
        self.temperature: float = loaded_file["temperature"]
        self.presence_penalty: float = loaded_file["presence_penalty"]
        self.frequency_penalty: float = loaded_file["frequency_penalty"]
        self.top_p: float = loaded_file["top_p"]
        self.stop_words: list[str] = loaded_file["stop_words"]

        # server parameters
        self.api_key: str = loaded_file["api_key"]
        self.api_base_url: str = loaded_file["api_base_url"]

        self.image_api_key: str = loaded_file["image_api_key"]
        self.image_api_base_url: str = loaded_file["image_api_base_url"]
        self.image_system_prompt: str = loaded_file["image_system_prompt"]
        self.image_question_prompt: str = loaded_file["image_question_prompt"]
        self.image_processing_enabled: bool = loaded_file["image_processing_enabled"]
