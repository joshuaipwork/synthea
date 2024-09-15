from typing import override
import discord
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion

from synthea.ImageDatabase import ImageDatabase
from synthea.Config import Config
from synthea.Model import Model


class VisionModel(Model):
    """
    Sends images to an external API to be captioned.
    Once captioned, the image's caption is stored into a database.
    The same image
    """

    def __init__(self):
        self.config: Config = Config()
        self.image_database: ImageDatabase = ImageDatabase()
        self.openai: AsyncOpenAI = AsyncOpenAI(
            base_url=self.config.image_api_base_url, api_key=self.config.image_api_key
        )

    @override
    async def queue_for_generation(self, chat_history: list[dict[str, dict[str, str]]]) -> str:
        """
        Sends a prompt to the server for generation. When the server is available,
        it will take up the prompt and generate a response.
        """
        config: Config = Config()

        print(chat_history)
        chat_completion: ChatCompletion = await self.openai.chat.completions.create(
            messages=chat_history,
            model="gpt-4-vision-preview",
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

    async def get_caption_for_image(self, image_url: str):
        description: str | None = self.image_database.get_image_description(image_url)
        if description:
            return description

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.config.image_system_prompt}],
            }
        ]
        content = [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": self.config.image_question_prompt},
        ]
        messages.extend([{"role": "user", "content": content}])

        response: ChatCompletion = await self.openai.chat.completions.create(
            model="gpt-4-vision-preview", messages=messages
        )
        description: str = response.choices[0].message.content
        self.image_database.add_image_description(image_url, description)

        # messages.pop([{"role": "user", "content": content}])

        # response = self.client.chat.completions.create(
        #     model="gpt-4-vision-preview", messages=messages
        # )
        # description: str = response.choices[0].message.content

        return description

    