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
    Custom stopping criteria for text generation. 
    Stops generation if the model predicts what the user might say next.

    Attributes:
        tokenizer (AutoTokenizer): The tokenizer used to convert text to tokens and vice-versa.
        forbidden_strings (list): Strings that should not be present in the generated output.
        prompt (str): The user's input.
    """
    
    def __init__(self, tokenizer: AutoTokenizer, prompt_format: dict[str, str], prompt: str):
        """
        Initializes the stopping criteria.

        Args:
            tokenizer (AutoTokenizer): The tokenizer used for the model.
            prompt_format (dict[str, str]): Format of the prompt.
            prompt (str): User's input.
        """
        super()
        self.tokenizer: AutoTokenizer = tokenizer
        self.forbidden_strings = [
            prompt_format['user_message_tag'],
            prompt_format['bot_message_tag'],
        ]
        self.prompt = prompt

    def __call__(self, input_ids, scores, **kwargs):
        """
        Determines whether generation should stop based on the presence of forbidden strings.

        Args:
            input_ids (list): Generated tokens' IDs.
            scores (list): Probabilities for each generated token.
            kwargs: Additional parameters (not used in this method).

        Returns:
            bool: Whether the generation should stop or not.
        """
        # Get the generated text as a string
        generated_text = self.tokenizer.decode(input_ids[0])
        # Remove the prompt from the generated text
        generated_text = generated_text.replace(self.prompt,'')
        # Check if any forbidden strings are present
        return any(string in generated_text for string in self.forbidden_strings)

    def __len__(self):
        return 1

    def __iter__(self):
        yield self

class ChattyModel:
    """
    A wrapper around Huggingface models designed for chatbot functionality.

    Attributes:
        config (dict): Configuration for the chatbot, loaded from 'config.yaml'.
        format (dict): Format specifications for the chatbot's responses.
        tokenizer (ExLlamaTokenizer): Tokenizer for the model.
        model_config (ExLlamaConfig): Configuration for the ExLlama model.
        generator (ExLlamaGenerator): Generator for producing model responses.
    """
    
    def __init__(self):
        """
        Initializes the chatbot model, loading configurations and settings from specified YAML files.
        """
        with open('config.yaml', "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)
        with open(f"formats/{self.config['format']}.yaml", "r", encoding="utf-8") as file:
            self.format = yaml.safe_load(file)
        self.tokenizer = None
        self.model_config = None
        self.generator = None

    def load_model(self, hf_model_dir: Optional[str]=None):
        """
        Loads a model from a local directory. 

        Args:
            hf_model_dir (str, optional): The Huggingface repository from which the model was downloaded. 
                This is also the directory within the /models folder where the model was saved.
        """
        # If a model directory isn't specified, default to the config setting
        if not hf_model_dir:
            hf_model_dir = self.config['model_name_or_path']

        local_model_dir = os.path.join("./models", hf_model_dir)
        
        # Locate necessary files within the model directory
        tokenizer_path = os.path.join(local_model_dir, "tokenizer.model")
        model_config_path = os.path.join(local_model_dir, "config.json")
        st_pattern = os.path.join(local_model_dir, "*.safetensors")
        model_path = glob.glob(st_pattern)[0]

        # Initialize ExLlama components
        self.model_config = ExLlamaConfig(model_config_path)
        self.model_config.model_path = model_path
        hf_model_dir = ExLlama(self.model_config)
        self.tokenizer = ExLlamaTokenizer(tokenizer_path)
        cache = ExLlamaCache(hf_model_dir)
        self.generator = ExLlamaGenerator(hf_model_dir, self.tokenizer, cache)

        # Mute unnecessary logging
        logging.set_verbosity(logging.CRITICAL)

    def generate_from_defaults(self, prompt: str) -> str:
        """
        Generates a response using default settings.

        Args:
            prompt (str): User's input.

        Returns:
            str: Model's response.
        """
        return self.generate(prompt)

    def generate(self, prompt: str, temperature: Optional[float]=None, top_p: Optional[float]=None, top_k: Optional[float]=None, repetition_penalty: Optional[float]=None, max_new_tokens: Optional[int]=None) -> str:
        """
        Generates a response from the model using specified settings or defaults.

        Args:
            prompt (str): User's input.
            temperature (float, optional): Temperature setting for randomness of response.
            top_p (float, optional): Nucleus sampling parameter.
            top_k (float, optional): Top-K sampling parameter.
            repetition_penalty (float, optional): Penalty for repeated tokens.
            max_new_tokens (int, optional): Maximum number of tokens to generate.

        Returns:
            str: Model's response.
        """
        # If parameters aren't specified, default to the model's configuration
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

        # Update generator settings
        self.generator.settings.token_repetition_penalty_max = repetition_penalty
        self.generator.settings.temperature = temperature
        self.generator.settings.top_p = top_p
        self.generator.settings.top_k = top_k

        output = self.generator.generate_simple(prompt, max_new_tokens=max_new_tokens)

        # Remove the initial prompt from the generated output
        str_output = output[len(prompt):]

        return str_output

    def generate_from_character(self, prompt: str, character: str) -> str:
        """
        Generates a response based on a specified character's configuration.

        Args:
            prompt (str): User's input.
            character (str): Name of the character profile.

        Returns:
            str: Model's response.

        Raises:
            FileNotFoundError: If the character configuration file is not found.
        """
        try:
            # Load character-specific configuration
            with open(f'characters/{character}.yaml', "r", encoding='utf-8') as f:
                loaded_config = yaml.safe_load(f)
                temperature = loaded_config['temperature'] if 'temperature' in loaded_config else None
                top_p = loaded_config['top_p'] if 'top_p' in loaded_config else None
                repetition_penalty = loaded_config['repetition_penalty'] if 'repetition_penalty' in loaded_config else None
                max_new_tokens = loaded_config['max_new_tokens'] if 'max_new_tokens' in loaded_config else None

            return self.generate(prompt, temperature, top_p, None, repetition_penalty, max_new_tokens)

        except FileNotFoundError:
            raise FileNotFoundError(f"The character configuration file for {character} was not found.")

if __name__ == "__main__":
    chatbot = ChattyModel()
    chatbot.load_model()
    prompt = input("Enter a message: ")
    response = chatbot.generate_from_defaults(prompt)
    print(f"Response: {response}")
