import yaml

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

        # langfuse parameters (optional)
        self.langfuse_url: str = loaded_file["langfuse_url"]
        self.langfuse_public_key: str = loaded_file["langfuse_public_key"]
        self.langfuse_secret_key: str = loaded_file["langfuse_secret_key"]
        self.langfuse_enabled: str = self.langfuse_url and self.langfuse_public_key and self.langfuse_secret_key

        # main model parameters
        self.api_key: str = loaded_file["api_key"]
        self.api_base_url: str = loaded_file["api_base_url"]
        self.default_model_name: str = loaded_file["default_model_name"]
        # Convert the list of dicts into a dict of ModelDefinitions
        self.models: dict[str, ModelDefinition] = {}
        for model in loaded_file["models"]:
            # Get the model name (the only key)
            model_name = list(model.keys())[0].lower()
            # Get the model properties (the only value)
            model_props = list(model.values())[0]

            # Create ModelDefinition from the properties
            self.models[model_name] = ModelDefinition(
                description=model_props.get("description", ""),
                vision=model_props.get("vision", False),
                reasoning=model_props.get("reasoning", False))

        # tool APIs
        self.tavily_api_key: str = loaded_file["tavily_api_key"]

        # image 
        self.image_generation_api_base_url: str = loaded_file["image_generation_api_base_url"]
        self.image_generation_enabled: str = loaded_file["image_generation_enabled"]

        self.image_maximum_pixels: int = loaded_file["image_maximum_pixels"]
        self.image_default_height: str = loaded_file["image_default_height"]
        self.image_default_width: str = loaded_file["image_default_width"]

        self.image_generation_api_headers: dict[str, str] = loaded_file.get("image_generation_api_headers", None)

        self.comfyui_workflow_path: str = loaded_file["comfyui_workflow_path"]
        self.comfyui_prompt_node_id: int = loaded_file["comfyui_prompt_node_id"]
        self.comfyui_prompt_input_name: str = loaded_file["comfyui_prompt_input_name"]
        self.comfyui_height_node_id: int = loaded_file["comfyui_height_node_id"]
        self.comfyui_height_input_name: str = loaded_file["comfyui_height_input_name"]
        self.comfyui_width_node_id: int = loaded_file["comfyui_width_node_id"]
        self.comfyui_width_input_name: str = loaded_file["comfyui_width_input_name"]
        self.comfyui_seed_node_id: int = loaded_file["comfyui_seed_node_id"]
        self.comfyui_seed_input_name: str = loaded_file["comfyui_seed_input_name"]        