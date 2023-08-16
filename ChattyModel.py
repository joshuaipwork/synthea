from pathlib import Path
import yaml
from typing import Optional
import yaml
from transformers import AutoTokenizer, pipeline, logging, TextGenerationPipeline, TextStreamer
from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
from exllama.tokenizer import ExLlamaTokenizer
from exllama.generator import ExLlamaGenerator
import os, glob


from transformers import StoppingCriteria

class StopAtReply(StoppingCriteria):
    """
    Sometimes, the bot will try and predict what the user will say next,
    and add it to the output. Users don't like being told what they are about to say,
    so this stopping criteria ends the response if that happens.
    """
    def __init__(self, tokenizer: AutoTokenizer, prompt_format: dict[str, str], prompt: str):
        self.tokenizer: AutoTokenizer = tokenizer
        self.forbidden_strings = [
            prompt_format['user_message_tag'],
            prompt_format['bot_message_tag'],
        ]
        self.prompt = prompt

    def __call__(self, input_ids, scores, **kwargs):
        # Get the generated text as a string
        generated_text = self.tokenizer.decode(input_ids[0])

        # the generated text includes both the prompt and the new text,
        # so remove the prompt
        generated_text = generated_text.replace(self.prompt,'')

        # quit if any forbidden strings are in it
        return any(string in generated_text for string in self.forbidden_strings)

    def __len__(self):
        return 1

    def __iter__(self):
        yield self

class ChattyModel:
    """
    A wrapper around Huggingface models for the chatbot.
    """
    def __init__(self):
        with open('config.yaml', "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)
        with open(f"formats/{self.config['format']}.yaml", "r", encoding="utf-8") as file:
            self.format = yaml.safe_load(file)
        self.tokenizer = None
        self.model_config = None
        self.generator = None

    def load_model(self, hf_model_dir: Optional[str]=None):
        """
        Loads a model from a local directory. The local directory is in the models/
        folder, and is named according to the huggingface model repo to which the
        model belongs. For a model to be loadable, tokenizer.model, config.json,
        and the *.safetensors version of the model must be in the local directory.

        Args:
            hf_model_dir (str or None): The huggingface repo from which the model
                was originally downloaded. This is also the directory with in the /models
                folder to which the model was downloaded.
        """
        # Directory containing model, tokenizer, generator
        if not hf_model_dir:
            hf_model_dir = self.config['model_name_or_path']

        local_model_dir =  os.path.join("./models", hf_model_dir)
            

        # Locate files we need within that directory
        tokenizer_path = os.path.join(local_model_dir, "tokenizer.model")
        print(tokenizer_path)
        model_config_path = os.path.join(local_model_dir, "config.json")
        print(model_config_path)
        st_pattern = os.path.join(local_model_dir, "*.safetensors")
        print(st_pattern)
        model_path = glob.glob(st_pattern)[0]

        # Create config, model, tokenizer and generator

        self.model_config = ExLlamaConfig(model_config_path)               # create config from config.json
        self.model_config.model_path = model_path                          # supply path to model weights file

        hf_model_dir = ExLlama(self.model_config)                    # create ExLlama instance and load the weights
        self.tokenizer = ExLlamaTokenizer(tokenizer_path)            # create tokenizer from tokenizer model file
        cache = ExLlamaCache(hf_model_dir)
        self.generator = ExLlamaGenerator(
            hf_model_dir,
            self.tokenizer,
            cache
        )

        # Prevent printing spurious transformers error when using pipeline with AutoGPTQ
        logging.set_verbosity(logging.CRITICAL)

    def generate_from_defaults(self, prompt: str) -> str:
        """
        Generates from a prompt with default settings.
        """
        return self.generate(prompt)


    def generate(
            self,
            prompt: str,
            temperature: Optional[float]=None,
            top_p: Optional[float]=None,
            top_k: Optional[float]=None,
            repetition_penalty: Optional[float]=None,
            max_new_tokens: Optional[int]=None,
        ) -> str:
        """
        Generates text from the model using exllama.

        Args:
            prompt (str): The prompt to feed to the AI.
        """
        # if generation parameters were not specified, use defaults from the model
        if not temperature:
            temperature = self.config['default_temperature']
        if not top_p:
            top_p = self.config['default_top_p']
        if not top_k:
            top_k = self.config['default_top_k']
        if not repetition_penalty:
            repetition_penalty = self.config['default_repetition_penalty']
        if not max_new_tokens:
            max_new_tokens = self.config['max_new_tokens']

        # Update generator with configuration from character and model config

        # self.generator.disallow_tokens([self.tokenizer.eos_token_id])
        self.generator.settings.token_repetition_penalty_max = repetition_penalty
        self.generator.settings.temperature = temperature
        self.generator.settings.top_p = top_p
        self.generator.settings.top_k = top_k

        output = self.generator.generate_simple(
            prompt,
            max_new_tokens=max_new_tokens
        )

        str_output = output[len(prompt):]

        return str_output

    def generate_from_character(self, prompt: str, character: str) -> str:
        """
        Args:
            config (str): The name of a prompt config. 
                A prompt config contains things like background, guidelines for the LLM's response, and other
                information useful for getting the result the user wants. 
                It is loaded from the corresponding file in the /prompt_configs folder,
                and is used to generate a prompt for the model using a prompt template.
            prompt (str): The prompt to feed to the AI alongside the information in the config.
        """
        # TODO: Update documentation
        # TODO: Don't load yaml twice
        try:
            with open(f'characters/{character}.yaml', "r", encoding='utf-8') as f:
                loaded_config = yaml.safe_load(f)
                temperature = loaded_config['temperature'] if 'temperature' in loaded_config else None
                top_p = loaded_config['top_p'] if 'top_p' in loaded_config else None
                repetition_penalty = loaded_config['repetition_penalty'] if 'repetition_penalty' in loaded_config else None

            return self.generate(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )
        except FileNotFoundError as err:
            raise FileNotFoundError(f"No character matching {character} was found in this server. You may need to create a new character by this name.") from err

        # TODO: If the prompt doesn't exist, tell the user something is wrong