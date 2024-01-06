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

        if "context_length" in new_config:
            print(f"Set context length to {new_config['context_length']}")
            self.config. = new_config["context_length"]
        else:
            print("No config.yaml context length found. Defaulting to context length of " + self.config.max_seq_len)

        self.model = ExLlamaV2(self.config)
        self.cache = ExLlamaV2Cache(self.model, lazy = True)

        print("Loading model...")
        self.model.load_autosplit(self.cache)

        self.tokenizer = ExLlamaV2Tokenizer(self.config)
        self.generator = ExLlamaV2StreamingGenerator(self.model, self.cache, self.tokenizer)
        self.generator.set_stop_conditions([self.tokenizer.eos_token_id])
        self.gen_settings = ExLlamaV2Sampler.Settings()


    def begin_stream(self, context: str) -> None:
        """
        Start generating text from the following context.

        Args:
            context (str): The full context for the model to conduct inference on.
        """

        instruction_ids = self.tokenizer.encode(context, add_bos = True)
        context_ids = instruction_ids if self.generator.sequence_ids is None \
            else torch.cat([self.generator.sequence_ids, instruction_ids], dim = -1)
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


