from exllamav2 import *
from exllamav2.generator import *
from ruamel.yaml import YAML
import sys, torch

class ExLlamaV2Model:
    """
    A wrapper around Huggingface models designed for chatbot functionality.
    """

    def __init__(self):
        """
        Initializes the chatbot model, loading configurations and settings from specified YAML files.
        """
        yaml = YAML()
        with open("config.yaml", "r", encoding="utf-8") as file:
            new_config = yaml.load(file)

        # import basic settings from huggingface repo
        self.config = ExLlamaV2Config()
        self.config.model_dir = f"./models/{new_config['model_name_or_path']}"
        self.config.prepare()

        # override with settings from config.yaml
        
        if "context_length" in new_config:
            print(f"Set context length to {new_config['context_length']}")
            self.config.max_seq_len = new_config["context_length"]
        else:
            print("No config.yaml context length found. Defaulting to context length of " + self.config.max_seq_len)

        self.model = ExLlamaV2(self.config)
        self.cache = ExLlamaV2Cache(self.model, lazy = True)

        print("Loading model...")
        self.model.load_autosplit(self.cache)

        self.tokenizer = ExLlamaV2Tokenizer(self.config)
        self.generator = ExLlamaV2StreamingGenerator(self.model, self.cache, self.tokenizer)

        stop_on = new_config["stop_on"]
        stop_on.append(self.tokenizer.eos_token_id)
        self.generator.set_stop_conditions(stop_on)
        print("Model loaded!")


    def begin_stream(self, context: str) -> None:
        """
        Start generating text from the following context.

        Args:
            context (str): The full context for the model to conduct inference on.
        """
        yaml = YAML()
        with open("config.yaml", "r", encoding="utf-8") as file:
            new_config = yaml.load(file)

        self.gen_settings = ExLlamaV2Sampler.Settings()

        self.gen_settings.mirostat = new_config["mirostat"] if "mirostat" in new_config else False
        self.gen_settings.mirostat_eta = new_config["mirostat_eta"] if "mirostat_eta" in new_config else 1.5
        self.gen_settings.mirostat_tau = new_config["mirostat_tau"] if "mirostat_tau" in new_config else 0.1

        self.gen_settings.top_k = new_config["top_k"] if "top_k" in new_config else 50
        self.gen_settings.top_p = new_config["top_p"] if "top_p" in new_config else 0.8
        self.gen_settings.min_p = new_config["min_p"] if "min_p" in new_config else 0

        self.gen_settings.temperature = new_config["temperature"] if "temperature" in new_config else 2.0
        self.gen_settings.token_repetition_penalty = new_config["repetition_penalty"] if "repetition_penalty" in new_config else 1.15

        context_ids = self.tokenizer.encode(context, add_bos = True)
        self.generator.begin_stream(context_ids, self.gen_settings)

        return self.generator

if __name__ == "__main__":
    chat_model: ExLlamaV2Model = ExLlamaV2Model()
    while True:

        print()
        instruction = input("User: ")
        print()
        print("Assistant:", end = "")

        chat_model.begin_stream(instruction)

        while True:
            chunk, eos, _ = chat_model.generator.stream()
            if eos: break
            print(chunk, end = "")
            sys.stdout.flush()

        print()


