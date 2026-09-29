import hashlib
import re

import yaml

from synthea.constants import DEFAULT_URL_READER_MAX_CHARS
from synthea.model_definition import ModelDefinition


def _slug(text: str, max_length: int) -> str:
    """Roughs text down to the characters chromaDB allows in a collection name
    (lowercase alphanumerics plus ``._-``).
    """
    slug = re.sub(r"[^a-z0-9._-]+", "-", text.lower()).strip("-._")
    return slug[:max_length].rstrip("-._")


class Config:
    """A simple class for storing and loading config.yaml
    """

    def __init__(self):
        """Load config.yaml and parse it into the class fields
        """
        with open("config.yaml", encoding="utf-8") as file:
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
        self.langfuse_enabled: str = (
            self.langfuse_url and self.langfuse_public_key and self.langfuse_secret_key
        )

        # main model parameters
        self.api_key: str = loaded_file["api_key"]
        self.api_base_url: str = loaded_file["api_base_url"]

        # embeddings url
        self.enable_memory: bool = loaded_file.get("enable_memory", False)
        self.enable_rag_lookup: bool = loaded_file.get("enable_rag_lookup", False)

        # url reading (the read_url tool)
        self.enable_url_reader: bool = loaded_file.get("enable_url_reader", True)
        self.url_reader_max_chars: int = loaded_file.get(
            "url_reader_max_chars", DEFAULT_URL_READER_MAX_CHARS,
        )
        self.embeddings_base_url: str = loaded_file.get(
            "embeddings_base_url", self.api_base_url,
        )
        self.embeddings_model: str = loaded_file.get(
            "embeddings_model", "text-embedding-3-small",
        )

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
                reasoning=model_props.get("reasoning", False),
                reasoning_effort=model_props.get("reasoning_effort", "medium"),
            )

        # tool APIs
        self.tavily_api_key: str = loaded_file["tavily_api_key"]

        # image
        self.image_generation_api_base_url: str = loaded_file[
            "image_generation_api_base_url"
        ]
        self.image_generation_enabled: str = loaded_file["image_generation_enabled"]

        self.image_maximum_pixels: int = loaded_file["image_maximum_pixels"]
        self.image_default_height: str = loaded_file["image_default_height"]
        self.image_default_width: str = loaded_file["image_default_width"]

        self.image_generation_api_headers: dict[str, str] = loaded_file.get("image_generation_api_headers")

        self.comfyui_workflow_path: str = loaded_file["comfyui_workflow_path"]
        self.comfyui_prompt_node_id: int = loaded_file["comfyui_prompt_node_id"]
        self.comfyui_prompt_input_name: str = loaded_file["comfyui_prompt_input_name"]
        self.comfyui_height_node_id: int = loaded_file["comfyui_height_node_id"]
        self.comfyui_height_input_name: str = loaded_file["comfyui_height_input_name"]
        self.comfyui_width_node_id: int = loaded_file["comfyui_width_node_id"]
        self.comfyui_width_input_name: str = loaded_file["comfyui_width_input_name"]
        self.comfyui_seed_node_id: int = loaded_file["comfyui_seed_node_id"]
        self.comfyui_seed_input_name: str = loaded_file["comfyui_seed_input_name"]

    @property
    def embeddings_scope(self) -> str:
        """A short id for the embedding space this configuration produces.

        Every vector collection the bot keeps (memories, saved documents) is
        suffixed with it, because embeddings from a different provider or model
        have a different shape and are not comparable - and chromaDB refuses to
        keep vectors of mixed shapes in one collection. So repointing the config
        at another embeddings service starts fresh collections instead of
        tripping over the ones the previous service filled.

        Only the embeddings settings are hashed: changing unrelated config
        keeps the same collections, and a trailing slash on the url does not
        count as a new provider.
        """
        identity = (
            f"{self.embeddings_base_url.rstrip('/')}\n{self.embeddings_model}"
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:6]
        # keep the whole collection name inside chromaDB's 63 character limit,
        # even with the 20 digit discord ids the document collections carry
        return f"{_slug(self.embeddings_model, 20) or 'model'}-{digest}"
