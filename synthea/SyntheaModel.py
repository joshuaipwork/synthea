import yaml
from typing import Optional
import yaml
from transformers import AutoTokenizer, logging
import os, glob
from transformers import StoppingCriteria
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


class StopAtReply(StoppingCriteria):
    """
    Custom stopping criteria for text generation.
    Stops generation if the model predicts what the user might say next.

    Attributes:
        tokenizer (AutoTokenizer): The tokenizer used to convert text to tokens and vice-versa.
        forbidden_strings (list): Strings that should not be present in the generated output.
        prompt (str): The user's input.
    """

    def __init__(
        self, tokenizer: AutoTokenizer, prompt_format: dict[str, str], prompt: str
    ):
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
            prompt_format["user_message_tag"],
            prompt_format["bot_message_tag"],
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
        generated_text = generated_text.replace(self.prompt, "")
        # Check if any forbidden strings are present
        return any(string in generated_text for string in self.forbidden_strings)

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
        with open("config.yaml", "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)
        with open(
            f"formats/{self.config['format']}.yaml", "r", encoding="utf-8"
        ) as file:
            self.format = yaml.safe_load(file)
        self.tokenizer = None
        self.model_config = None
        self.generator = None
        self.llm = None

    def load_model(
        self,
        hf_model_dir: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        context_length: Optional[int] = None,
    ):
        """
        Loads a model from a local directory.

        Args:
            hf_model_dir (str, optional): The Huggingface repository from which the model was downloaded.
                This is also the directory within the /models folder where the model was saved.
        """
        # If a model directory isn't specified, default to the config setting
        if not hf_model_dir:
            hf_model_dir = self.config["model_name_or_path"]
        local_model_dir = os.path.join("./models", hf_model_dir)

        # Locate necessary files within the model directory
        st_pattern_bin = os.path.join(local_model_dir, "*.bin")
        st_pattern_gguf = os.path.join(local_model_dir, "*.gguf")

        model_paths = glob.glob(st_pattern_bin) + glob.glob(st_pattern_gguf)
        if model_paths:
            model_path = model_paths[0]

        # load the format file for the model, containing strings to stop on
        with open(
            f"formats/{self.config['format']}.yaml", "r", encoding="utf-8"
        ) as file:
            self.format = yaml.safe_load(file)
        if "stop_on" in self.format:
            stop = self.format["stop_on"]
        else:
            stop = []

        # Callbacks support token-wise streaming
        # Verbose is required to pass to the callback manager
        # callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        # TODO: Move this to config instead of hardcoded.
        # Change this value based on your model and your GPU VRAM pool.
        n_gpu_layers = 500
        # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
        n_batch = 512

        # If parameters aren't specified, default to the model's configuration
        if not temperature:
            temperature = self.config["default_temperature"]
        if not top_p:
            top_p = self.config["default_top_p"]
        if not top_k:
            top_k = self.config["default_top_k"]
        if not repetition_penalty:
            repetition_penalty = self.config["default_repetition_penalty"]
        if not max_new_tokens:
            max_new_tokens = self.config["max_new_tokens"]
        if not context_length:
            context_length = self.config["context_length"]

        # Make sure the model path is correct for your system!
        self.llm = LlamaCpp(
            model_path=model_path,
            temperature=temperature,
            max_tokens=max_new_tokens,
            n_ctx=context_length,
            top_p=top_p,
            top_k=top_k,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            verbose=True,
            stop=stop,
        )

    def generate(self, prompt: str) -> str:
        """
        Generates a response from the model using specified settings or defaults.

        Args:
            prompt (str): The prompt to give the LLM.

        Returns:
            (str): Model's response.
        """

        # Update generator settings
        prompt_template = PromptTemplate(
            template="""{prompt}""", input_variables=["prompt"]
        )

        llm_chain = LLMChain(prompt=prompt_template, llm=self.llm)

        str_output = llm_chain.run(prompt)

        return str_output


if __name__ == "__main__":
    chat_model = SyntheaModel()
    chat_model.load_model()
    user_prompt = input("Enter a message: ")
    response = chat_model.generate(user_prompt)
    print(f"Response: {response}")
