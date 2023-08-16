import json
import yaml
from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

class ChattyModel:
    def __init__(self):
        with open('config.json', "r") as f:
            self.config = json.load(f)
    
    def load_model(self):
        """
        Loads the model 
        """
        use_triton = False
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name_or_path'], use_fast=True)
        self.model = AutoGPTQForCausalLM.from_quantized(self.config['model_name_or_path'],
                model_basename=self.config['model_basename'],
                use_safetensors=True,
                trust_remote_code=True,
                device_map='auto',
                use_triton=use_triton,
                quantize_config=None)

        self.model.seqlen = 8192

        # Inference can also be done using transformers' pipeline

        # Prevent printing spurious transformers error when using pipeline with AutoGPTQ
        logging.set_verbosity(logging.CRITICAL)

        print("*** Pipeline:")
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )

    def generate_without_character(self, prompt: str) -> str:
        prompt_template=f'''
        USER: {prompt}
        ASSISTANT:'''
        return self.generate_from_prompt(prompt_template)


    def generate_from_prompt(self, prompt: str) -> str:
        """
        Args:
            prompt (str): The prompt to feed to the AI.
        """
        input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.cuda()
        output = self.model.generate(inputs=input_ids, temperature=0.7, max_new_tokens=512)

        # convert the output back to a string
        str_output = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # truncate for discord limits
        # TODO: add logic to extend the character limit by posting multiple messages in a thread
        str_output = str_output[len(prompt):len(prompt) + 2000]
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
            with open(f'characters/{character}.yaml', "r") as f:
                loaded_config = yaml.safe_load(f)

                prompt_template=f'''
                USER: {loaded_config['intro']}
                {f"BACKGROUND: {loaded_config['background']}" if 'background' in loaded_config else ""}
                {f"GUIDELINES: {loaded_config['guidelines']}" if 'guidelines' in loaded_config else ""}
                Respond to this prompt: {prompt}
                ASSISTANT:'''

            print(prompt_template)
            return self.generate_from_prompt(prompt_template)
        except FileNotFoundError as err:
            raise FileNotFoundError(f"No character matching {character} was found in this server. You may need to create a new character by this name.")


        # TODO: If the prompt doesn't exist, tell the user something is wrong