
import json
import re
from typing import override
import aiohttp
import openai
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.completion import Completion
import requests

from synthea.VisionModel import VisionModel
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

TOOL_CALL_PATTERN = re.compile(r'<tool_call>(.*?)<\/tool_call>', re.DOTALL)

class LanguageModel(Model):
    """
    Makes requests to an openAI-compatible API that only
    takes in text. Use ImageModel for models with vision.
    """

    def __init__(self):
        self.config: Config = Config()
        self.image_model: VisionModel = VisionModel()
        self.openai: openai.AsyncOpenAI = openai.AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.api_base_url
        )

    @override
    async def queue_for_generation(self, chat_history: list[dict[str, dict[str, str]]]) -> str:
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

            # generate the response
            base_url = 'http://localhost:8080'
            # params = {'param1': 'value1', 'param2': 'value2'}
            
            # Define the request body
            body = {
                'prompt': prompt,
                'stop': config.stop_words,
                'cache_prompt': True,
                'n_predict': config.max_new_tokens
            }

            # Create an aiohttp session
            async with aiohttp.ClientSession() as session:
                # Make the POST request
                async with session.post(config.api_base_url, json=body) as response:
                    # Check if the request was successful
                    if response.status == 200:
                        # Parse the JSON response
                        data = await response.json()
                        print("Response data:", data)
                    else:
                        print(f"Error: HTTP {response.status}")
                        print(await response.text())
                        raise requests.exceptions.HTTPError(f"{response.status} Response from inference server: {response.text()}")
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
                inference_logger.info(f"Responded to model:\n {tool_response_message}")
            else:
                # no tool call, so we can just quit out
                needs_call = False

        # return the final result
        return last_completion

    async def execute_function_call(self, function_name: str, function_args: dict[str]):
        function_to_call = getattr(Tools, function_name, None)
        inference_logger.info(f"Invoking function call {function_name} ...")
        function_response = await function_to_call(*function_args.values())
        results_dict = f'{{"name": "{function_name}", "content": {function_response}}}'
        return results_dict

    @override
    async def queue_for_chat_generation(self, chat_history: list[dict[str, dict[str, str]]]) -> str:
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

        print(chat_history)
        chat_completion: ChatCompletion = await self.openai.chat.completions.create(
            messages=chat_history,
            model="gpt-3.5-turbo",
            max_tokens=config.max_new_tokens,
            presence_penalty=config.presence_penalty,
            frequency_penalty=config.frequency_penalty,
            temperature=config.temperature,
            seed=-1,
            top_p=config.top_p,
            stop=config.stop_words
        )
        # TODO: Add error handling

        return chat_completion.choices[0].message.content