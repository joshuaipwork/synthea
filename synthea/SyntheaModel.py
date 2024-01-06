from torch import Tensor
from ruamel.yaml import YAML
from typing import Optional
import torch
from transformers import AutoTokenizer, logging
import os, glob
from transformers import StoppingCriteria
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, Pipeline, PreTrainedTokenizer, PreTrainedModel, BitsAndBytesConfig

class StopOnTokens(StoppingCriteria):
    """
    Custom stopping criteria for text generation.
    Stops generation if the model predicts what the user might say next.

    Attributes:
        tokenizer (AutoTokenizer): The tokenizer used to convert text to tokens and vice-versa.
        forbidden_strings (list): Strings that should not be present in the generated output.
        prompt (str): The user's input.
    """

    def __init__(
        self, tokenizer: AutoTokenizer, stop_on: list[str], prompt: str
    ):
        """
        Initializes the stopping criteria.

        Args:
            tokenizer (AutoTokenizer): The tokenizer used for the model.
            prompt_format (dict[str, str]): Format of the prompt.
            prompt (str): The original prompt given to the model
        """
        super()
        self.tokenizer: AutoTokenizer = tokenizer
        self.stop_on = stop_on
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
        generated_text = generated_text.replace(self.prompt, "")
        # Check if any forbidden strings are present
        return any(string in generated_text for string in self.stop_on)

    def __len__(self):
        return 1

    def __iter__(self):
        yield self


class SyntheaModel:
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
        yaml = YAML()
        with open("config.yaml", "r", encoding="utf-8") as file:
            self.config = yaml.load(file)
        with open(
            f"formats/{self.config['format']}.yaml", "r", encoding="utf-8"
        ) as file:
            self.format = yaml.load(file)
        self.tokenizer: PreTrainedTokenizer = None
        self.model: PreTrainedModel = None
        self.pipe: Pipeline = None

    def load_model(self, model_name_or_path: Optional[str] = None):
        """
        Loads a model from 
        If the model doesn't exist, it will be downloaded from huggingface.

        Args:
            model_name_or_path (str, optional): The Huggingface repository to get
                the model and tokenizer from. You can also use a local path.
        """
        # To use a different branch, change revision
        # For example: revision="gptq-4bit-32g-actorder_True"
        if not model_name_or_path:
            model_name_or_path = self.config["model_name_or_path"]

        # bnb_config = BitsAndBytesConfig(
        #     load_in_8bit=True,
        #     # bnb_4bit_quant_type="nf4",
        #     # bnb_4bit_use_double_quant=True,
        # )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                # load_in_4bit=True,
                # quantization_config=bnb_config,
                # torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

    def generate(self, chat_history:  list[dict[str, str]]) -> str:
        """
        Generates a response from the model. Settings are based on the
        config.

        Args:
            chat_history (list of dict of str to str): The chat history to
                generate based on. Refer to huggingface documents on chat templates
                for more information.

        Returns:
            (str): the model's response
        """
        temperature = self.config["default_temperature"]
        top_p = self.config["default_top_p"]
        top_k = self.config["default_top_k"]
        repetition_penalty = self.config["default_repetition_penalty"]
        max_new_tokens = self.config["max_new_tokens"]
        context_length = self.config["context_length"]

        # TODO: Move chat template to config
        self.tokenizer.chat_template = self.config["chat_template"]
        gen_input: Tensor = self.tokenizer.apply_chat_template(
            chat_history,
            truncation=True,
            return_tensors="pt",
            max_length=context_length,
        )
        gen_input = gen_input.to("cuda")

        prompt_str = self.tokenizer.decode(gen_input[0])

        # create stopping criteria
        stopping_criteria = None
        if 'stop_on' in self.config:
            stopping_criteria=[StopOnTokens(
                tokenizer=self.tokenizer,
                stop_on=self.config['stop_on'],
                prompt=prompt_str)]

        print(f"Prompt ({len(gen_input[0])} tokens):\n" + self.tokenizer.decode(gen_input[0]))

        tensor_output: Tensor = self.model.generate(
            inputs=gen_input,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stopping_criteria=stopping_criteria
        )

        str_output: str = self.tokenizer.batch_decode(
            tensor_output[:, gen_input.shape[1]:],
            skip_special_tokens=True,
        )[0]

        str_output = self._clean_stopping_tokens(str_output)
        
        print(f"Prompt ({len(tensor_output[0])} tokens):\n" + str_output)
        return str_output

    def _clean_stopping_tokens(self, s: str):
        """
        """
        if "stop_on" in self.config and self.config["stop_on"]:
            for stop_token in self.config["stop_on"]:
                if s.endswith(stop_token):
                    return s[:-len(stop_token)]
        return s


if __name__ == "__main__":
    chat_model = SyntheaModel()
    chat_model.load_model()
    user_prompt = input("Enter a message: ")
    response = chat_model.generate([
            {"role": "system", "content": "You are as a sexy, slutty college girl with omnipotence."},
            {"role": "user", "content": user_prompt}
        ])
    print(f"Response: {response}")
