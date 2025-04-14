
import asyncio
import json
import re
from typing import Dict, override
import aiohttp
import openai
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.completion import Completion
import requests

from synthea.Config import Config
from synthea.Model import Model

from jinja2 import Template

import Tools
from ToolUtilities import (
    inference_logger,
    get_assistant_message,
    get_chat_template,
    validate_and_extract_tool_calls
)
from synthea.dtos.GenerationResponse import GenerationResponse

TOOL_CALL_PATTERN = re.compile(r'<tool_call>(.*?)<\/tool_call>', re.DOTALL)

class LanguageModel(Model):
    """
    Makes requests to an openAI-compatible API that only
    takes in text. Use ImageModel for models with vision.
    """

    def __init__(self):
        self.config: Config = Config()
        self.openai: openai.AsyncOpenAI = openai.AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.api_base_url
        )

    @override
    async def queue_for_generation(self, chat_history: list[dict[str, dict[str, str]]]) -> GenerationResponse:
        """
        Sends a prompt to the server for generation. When the server is available,
        it will take up the prompt and generate a response.
        """
        # load config again in case system prompts change
        config: Config = Config()

        # strip out the content, since the server isn't set up for multimodal
        # and combine it to a simple string 
        for chat_message in chat_history:
            text = ""
            for content_part in chat_message["content"]:
                if content_part["type"] == "text":
                    text += content_part["text"]
                if content_part["type"] == "image_url":
                    caption: str = await self.image_model.get_caption_for_image(content_part["image_url"]["url"])
                    text += f"\n\n```SYSTEM: An image was attached to this message. Here is a description of the image: {caption}```"
            chat_message["content"] = text

        template = Template(config.chat_template)
        prompt = template.render(messages=chat_history, add_generation_prompt=True)
        inference_logger.info(prompt)
        generation_count = 0
        needs_call = True
        last_completion: str = ""
        while needs_call and generation_count < 5:
            prompt = template.render(messages=chat_history, add_generation_prompt=True)

            data: dict[str, str] = None
            data = await self.generate_with_llama(prompt)

            generation_count += 1
            # TODO: Create a type for this
            last_completion = data["content"]
            inference_logger.info(f"Completion:\n{last_completion}")
            
            # check for tool call in the response
            if config.use_tools and "<tool_call>" in last_completion:
                if not last_completion.endswith("</tool_call>"):
                    last_completion += "</tool_call>"

                needs_call = True # need another call to have bot see results
                # add the tool call message to the message history
                tool_call_message: dict[str, dict[str, str]] = dict()
                tool_call_message["content"] = last_completion
                tool_call_message["role"] = "assistant"
                chat_history.append(tool_call_message)

                # capture the JSON 
                matches: str = TOOL_CALL_PATTERN.findall(last_completion)
                tool_response_text = ""
                if matches:
                    try:
                        tool_call = json.loads(matches[0])
                        function_response = await self.execute_function_call(tool_call["name"], tool_call["arguments"])
                        tool_response_text += f"<tool_response>\n{function_response}\n</tool_response>\n"
                        inference_logger.info(f"Here's the response from the function call: {tool_call.get('name')}\n{function_response}")
                    except Exception as e:
                        inference_logger.info(f"Could not execute function: {e}")
                        tool_response_text += f"<tool_response>\nThere was an error when executing the function: {tool_call.get('name')}\nHere's the error traceback: {e}\nPlease call this function again with correct arguments within XML tags <tool_call></tool_call>\n</tool_response>\n"
                else:
                    tool_response_text += f"<tool_response>\nThere was an error when parsing the request.\nPlease call this function again with correct arguments within XML tags <tool_call></tool_call>\n</tool_response>\n"
                    inference_logger.info(f"Tried to make a tool call, but no ending tag was found.")

                tool_response_message: dict[str, dict[str, str]] = dict()
                tool_response_message["content"] = tool_response_text
                tool_response_message["role"] = "tool"         
                chat_history.append(tool_response_message)
                # inference_logger.info(f"Responded to model:\n {tool_response_message}")
            else:
                # no tool call, so we can just quit out
                needs_call = False

        # parse the response into constituent parts if reasoning was present
        generation_response: GenerationResponse = GenerationResponse()
        if (last_completion.startswith(config.reasoning_start_tag)):
            think_split = last_completion.split(config.reasoning_end_tag, 1)
            if len(think_split) == 1:
                generation_response.reasoning = think_split[0]
                generation_response.final_output = "[The model didn't finish thinking within the time limit.]"
            else:
                generation_response.reasoning = think_split[0]
                generation_response.final_output = think_split[1]
        else:
            generation_response.final_output = last_completion

        # return the final result
        return generation_response

    async def execute_function_call(self, function_name: str, function_args: dict[str]):
        function_to_call = getattr(Tools, function_name, None)
        inference_logger.info(f"Invoking function call {function_name} ...")
        function_response = await function_to_call(*function_args.values())
        results_dict = f'{{"name": "{function_name}", "content": {function_response}}}'
        return results_dict

    @override
    async def queue_for_chat_generation(self, chat_history: list[dict[str, dict[str, str]]]) -> GenerationResponse:
        """
        Sends a prompt to the server for generation with the openAI chat API. When the server is available,
        it will take up the prompt and generate a response.
        """
        # load config again in case system prompts change
        config: Config = Config()

        if not config.can_process_images:
            # strip out the content, if the server isn't set up for multimodal
            # and combine it to a simple string 
            for chat_message in chat_history:
                text = ""
                for content_part in chat_message["content"]:
                    if content_part["type"] == "text":
                        text += content_part["text"]
                chat_message["content"] = text

        print(chat_history)
        chat_completion: ChatCompletion = await self.openai.chat.completions.create(
            messages=chat_history,
            model=config.model_name,
            max_tokens=config.max_new_tokens,
            stop=config.stop_words
        )
        # TODO: Add error handling

        print(chat_completion)

        # parse the response into constituent parts if reasoning was present
        generation_response: GenerationResponse = GenerationResponse()
        if (chat_completion.choices[0].message.content.startswith(config.reasoning_start_tag)):
            think_split = chat_completion.choices[0].message.content.split(config.reasoning_end_tag, 1)
            if len(think_split) == 1:
                generation_response.reasoning = think_split[0]
                generation_response.final_output = "[The model didn't finish thinking within the time limit.]"
            else:
                generation_response.reasoning = think_split[0]
                generation_response.final_output = think_split[1]
        else:
            generation_response.final_output = chat_completion.choices[0].message.content

        # return the final result
        return generation_response
    
    async def generate_with_llama(self, prompt: str) -> Dict[str, str]:
        config: Config = Config()

        # generate the response
        headers = {"Authorization": f"Bearer {config.api_key}"}

        # Define the request body
        body = {
            'prompt': prompt,
            'stop': config.stop_words,
            'cache_prompt': True,
            'n_predict': config.max_new_tokens,
        }

        # Create an aiohttp session
        async with aiohttp.ClientSession() as session:
            # Make the POST request
            async with session.post(config.api_base_url, json=body, headers=headers) as response:
                # Check if the request was successful
                if response.status == 200:
                    # Parse the JSON response
                    data = await response.json()
                    inference_logger.info("Response data:", data)
                else:
                    response_text: str = await response.text()
                    inference_logger.error(f"Error: HTTP {response.status}")
                    inference_logger.error(response_text)
                    raise requests.exceptions.HTTPError(f"{response.status} Response from inference server: {response_text}")
        return data
