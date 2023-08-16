from pathlib import Path
import yaml
from typing import Optional
import yaml
from transformers import AutoTokenizer, pipeline, logging, TextGenerationPipeline, TextStreamer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

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
        self.model = None
    
    def load_model(self):
        """
        Loads the model 
        """
        # TODO: Add functionality to select the model
        use_triton = False
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model_name_or_path'],
            use_fast=True,
            add_special_tokens=True,
        )
        self.model = AutoGPTQForCausalLM.from_quantized(
            self.config['model_name_or_path'],
            model_basename=self.config['model_basename'],
            use_safetensors=True,
            trust_remote_code=True,
            device_map='auto',
            use_triton=use_triton,
            quantize_config=None,
        )

        self.model.seqlen = self.config["seqlen"]

        # Inference can also be done using transformers' pipeline

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
            repetition_penalty: Optional[float]=None,
            max_new_tokens: Optional[int]=None,
        ) -> str:
        """
        Args:
            prompt (str): The prompt to feed to the AI.
        """
        if not temperature:
            temperature = self.config['default_temperature']
        if not top_p:
            top_p = self.config['default_top_p']
        if not repetition_penalty:
            repetition_penalty = self.config['default_repetition_penalty']
        if not max_new_tokens:
            max_new_tokens = self.config['max_new_tokens']

        pipe: TextGenerationPipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            min_new_tokens=1,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stopping_criteria=StopAtReply(self.tokenizer, self.format, prompt)
        )

        output = pipe(prompt, return_full_text=False)
        str_output = output[0]['generated_text']

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