from pathlib import Path
import yaml
from typing import Optional
import yaml
from transformers import AutoTokenizer, pipeline, logging, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

class ChattyModel:
    def __init__(self):
        with open('config.yaml', "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
            self.tokenizer = None
            self.model = None
    
    def load_model(self):
        """
        Loads the model 
        """
        # TODO: Add functionality to select the model
        use_triton = False
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name_or_path'], use_fast=True)
        self.model = AutoGPTQForCausalLM.from_quantized(self.config['model_name_or_path'],
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
            max_new_tokens: Optional[int]=None
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
            min_new_tokens=2,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        output = pipe(prompt, return_full_text=False)

        # truncate for discord limits
        # TODO: add logic to extend the character limit by posting multiple messages in a thread
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