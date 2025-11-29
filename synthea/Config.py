import yaml

from synthea.ModelFormatDefinition import ModelFormatDefinition
from synthea.ModelDefinition import ModelDefinition


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
        self.bot_name: str = loaded_file["bot_name"]
        self.chat_template: str = loaded_file["chat_template"]

        # generation parameters
        self.temperature: float = loaded_file["temperature"]
        self.presence_penalty: float = loaded_file["presence_penalty"]
        self.frequency_penalty: float = loaded_file["frequency_penalty"]
        self.top_p: float = loaded_file["top_p"]
        self.stop_words: list[str] = loaded_file["stop_words"]

        # prompt parameters
        self.tool_prompt: str = loaded_file["tool_prompt"]
        self.use_tools: bool = bool(loaded_file["use_tools"])

        # main model parameters
        self.api_key: str = loaded_file["api_key"]
        self.api_base_url: str = loaded_file["api_base_url"]
        self.default_model_name: str = loaded_file["default_model_name"]
        # Convert the list of dicts into a dict of ModelDefinitions
        self.models: dict[str, ModelDefinition] = {}
        for model in loaded_file["models"]:
            # Get the model name (the only key)
            model_name = list(model.keys())[0]
            # Get the model properties (the only value)
            model_props = list(model.values())[0]

            format: ModelFormatDefinition = None
            if "format" in model_props:
                format = ModelFormatDefinition(
                    reasoning_start_tag=model_props["format"]["reasoning_start_tag"],
                    reasoning_end_tag=model_props["format"]["reasoning_end_tag"],
                    solution_start_tag=model_props["format"]["solution_start_tag"],
                    solution_end_tag=model_props["format"]["solution_end_tag"])

            # Create ModelDefinition from the properties
            self.models[model_name] = ModelDefinition(
                description=model_props["description"],
                vision=model_props["vision"],
                reasoning=model_props["reasoning"],
                template=format)


        # prompt parameters
        self.reasoning_system_prompt: str = loaded_file["reasoning_system_prompt"]
        self.reasoning_start_tag: str = loaded_file["reasoning_start_tag"]
        self.reasoning_end_tag: str = loaded_file["reasoning_end_tag"]

        self.image_generation_enabled: str = loaded_file["image_generation_enabled"]
        self.image_generation_system_prompt: str = loaded_file["image_generation_system_prompt"]

        # image 
        self.image_generation_api_base_url: str = loaded_file["image_generation_api_base_url"]
        self.image_generation_workflow_name: str = loaded_file["image_generation_workflow_name"]
        self.image_maximum_pixels: int = loaded_file["image_maximum_pixels"]
        self.image_default_dimensions: str = loaded_file["image_default_dimensions"]