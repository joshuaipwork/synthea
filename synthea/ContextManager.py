# -*- coding: utf-8 -*-
"""
Generate a prompt for the AI to respond to, given the
message history and persona.
"""
from typing import AsyncIterator, Optional
import discord
import pypdf
from jinja2 import Environment
import os
import yaml

from synthea.CharactersDatabase import CharactersDatabase
from synthea.CommandParser import ChatbotParser, CommandError, ParsedArgs, ParserExitedException
from synthea import SyntheaClient
from synthea.Config import Config
from synthea.VisionModel import VisionModel
from synthea.LanguageModel import LanguageModel

class ReplyChainIterator:
    """
    An async iterator which follows a chain of discord message replies until it reaches the end
    or fails to capture the last message.
    """

    def __init__(self, starting_message: discord.Message):
        self.message = starting_message
        self.message_index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.message_index == 0:
            self.message_index += 1
            return self.message            
        # go back message-by-message through the reply chain and add it to the context
        if self.message.reference:
            self.message_index += 1
            try:
                self.message = await self.message.channel.fetch_message(
                    self.message.reference.message_id
                )
                return self.message

            except (discord.NotFound, discord.HTTPException, discord.Forbidden):
                # the user may have deleted their message
                # either way, we can't follow the history anymore
                # pylint: disable-next=raise-missing-from
                raise StopAsyncIteration
        else:
            raise StopAsyncIteration


class ContextManager:
    """
    Formats prompts for the bot to generate from.
    """

    # A rough measure of how many character are in each token.
    EST_CHARS_PER_TOKEN: int = 3

    def __init__(self, bot_user_id: int):
        """
        model (str): The model that is generating the text. Used to determine the prompt format
            and other configuration options.
        bot_user_id (str): The discord user id of the bot. Used to determine if a message came from
            the bot or from a user.
        """
        self.parser: ChatbotParser = ChatbotParser()
        self.image_model: VisionModel = VisionModel()
        self.language_model: LanguageModel = LanguageModel()
        self.characters_database: CharactersDatabase = CharactersDatabase()
        self.bot_user_id: int = bot_user_id

    async def generate_chat_history_from_chat(
        self, message: discord.Message,
        system_prompt: Optional[str] = None,
    ) -> tuple[list[dict[str, str]], ParsedArgs]:
        """
        Generates a prompt which includes the context from previous messages in a reply chain.
        Messages outside of the reply chain are ignored.

        Args:
            message (discord.Message): The last message from the user.
            system_prompt (str): The system prompt to use when generating the prompt
        Returns:
            A tuple of (chat_history, args)
            chat_history: an openai compatible chat history
            args: a ParsedArgs representing the most recent command in the
                chat history 
        """
        history_iterator: ReplyChainIterator = ReplyChainIterator(message)
        chat_history, args = await self.compile_chat_history(
            message=message,
            history_iterator=history_iterator,
            default_system_prompt=system_prompt,
        )

        return chat_history, args


    async def convert_chat_history_to_prompt(self, chat_history: list[dict[str, str]], chat_template: str) -> str:
        """
        Takes a chat template and converts it to a prompt. 

        Args:
            chat_history (list of dict of str to str): A list of chat messages to convert to a prompt.
                Refer to huggingface's chat template feature for information on how this should be formatted.
            chat_template (str): The jinja2 chat template to apply to the chat history.
        """
        chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '### Instruction:\\n' + message['content'].strip()}}{% elif message['role'] == 'system' %}{{ message['content'].strip() }}{% elif message['role'] == 'assistant' %}{{ '### Response\\n'  + message['content'] }}{% endif %}{{'\\n\\n'}}{% endfor %}{{ '### Response:\\n' }}"

        # Create a Jinja2 environment and compile the template
        env = Environment()
        template = env.from_string(chat_template)

        # Render the template with your messages
        formatted_chat = template.render(messages=chat_history)

        return formatted_chat

    async def compile_chat_history(
        self,
        message: discord.Message,
        history_iterator: AsyncIterator[discord.Message],
        default_system_prompt: Optional[str] = None,
    ) -> tuple[list[dict[str, str]], ParsedArgs]:
        """
        Generates an openai completion endpoint compatible messages object
        which includes the context from previous messages from the history.
        Returns the command which applies to this chat history, which is the last command which
        was sent in the reply chain.

        Args:
            message (discord.Message): The last message to add to the prompt.
            history_iterator (ReplyChainIterator): An iterator that contains the chat history
                to be included in the prompt.
            default_system_prompt (str): The system prompt to use if the system prompt is not   
                overriden by another command within the chat history
        Returns:

        """
        config = Config()

        # pieces of the prompts are appended to the list then assembled in reverse order into the final prompt
        token_count: int = 0
        last_command_args: ParsedArgs | None = None
        system_prompt = None

        # use provided system prompt
        messages = []

        # retrieve as many tokens as can fit into the context length from history
        history_token_limit: int = config.context_length - config.max_new_tokens
        system_prompt_tokens: int = len(default_system_prompt) // self.EST_CHARS_PER_TOKEN
        token_count += system_prompt_tokens
        async for message in history_iterator:
            # some messages in the chain may be commands for the bot
            # if so, parse only the prompt in each command in order to not confuse the bot
            content, added_tokens = await self._get_content(message, history_token_limit - token_count, config)
            text: str = self._extract_text_from_content(content)
            if text.lower().startswith(config.command_start_str.lower()):
                message_args: ParsedArgs = self.parser.parse(text)
                if not last_command_args:
                    last_command_args = message_args
                if not system_prompt and last_command_args.use_as_system_prompt:
                    system_prompt = last_command_args.prompt
                    # if the command sets the system prompt, don't include it in the history
                    continue
                # clean the command to only include the prompt parameter
                # TODO: support including images in these commands
                content = [{"type": "text", "text": f"{message_args.prompt}"}]
                added_tokens = len(message_args.prompt) // self.EST_CHARS_PER_TOKEN

            # skip messages that were created by the system
            if message.author.id == self.bot_user_id and message.embeds and message.embeds[0].footer.text == SyntheaClient.SYSTEM_TAG:
                continue

            # don't include empty messages so the bot doesn't get confused.
            if not content:
                continue

            # stop retrieving context if the context would overflow
            if added_tokens + token_count > history_token_limit:
                break

            # add the username
            self._inject_username(message, content)

            # update the prompt with this message
            if message.author.id == self.bot_user_id:
                messages.insert(0, {"role": "assistant", "content": content})
            else:
                messages.insert(0, {"role": "user", "content": content})
            
            token_count += added_tokens

        # add the system prompt

        if not system_prompt:
            system_prompt = default_system_prompt
        if config.use_tools:
            system_prompt += f"\n{config.tool_prompt}"
        messages.insert(0, {"role": "system", "content": [
                    {"type": "text", "text": (system_prompt if system_prompt else default_system_prompt)}
                    ]})

        return messages, last_command_args
    
    async def _read_attachment(self, attachment: discord.Attachment) -> tuple[str, str] | None:
        """
        Args:
            attachment: The attachment to read 
        Returns:
            (openai_content_type, attachment_string)
            A tuple with the openai type of the contents of the image, and a string representing it.
            If the attachment is a text or PDF attachment, the openai compatible type is "text"
            If the attachment is an image the openai compatible type is "image_url"

            ("text", "This is the content of the PDF")
            ("image_url", "https://images.freeimages.com/images/large-previews/cd7/gingko-biloba-1058537.jpg")
        """
        config = Config()
        openai_content_type = ""
        attachment_string = ""
        attachment_bytes = await attachment.read()
        if not attachment.content_type or attachment.content_type.startswith("text/"):
            openai_content_type = "text"
            attachment_string = attachment_bytes.decode()
        elif "application/pdf" in attachment.content_type:
            print("Saving the pdf attachment")
            openai_content_type = "text"
            await attachment.save(attachment.filename)
            reader = pypdf.PdfReader(attachment.filename)

            print(f"Found {len(reader.pages)} pages in PDF. Reading them.")
            for page in reader.pages:
                page_text = page.extract_text()
                attachment_string = attachment_string + "\n" + page_text
            
            print("Removing the saved file")
            os.remove(attachment.filename)
        elif attachment.content_type.startswith("image/"):
            if not config.image_processing_enabled:
                return None
            print("Found image attachment")
            # just incldue the image url
            openai_content_type = "image_url"
            attachment_string = attachment.url

        print(f"Obtained the text from the [{attachment.content_type}] attachment as a string")
        print(f"Recorded as ({openai_content_type}, {attachment_string})")
        return (openai_content_type, attachment_string)

    async def _get_linked_content(self, message: discord.Message, remaining_tokens: int, config: Config) -> tuple[list[dict[str, str]], int]:
        """
        Gets 
        """

    async def _get_content(self, message: discord.Message, remaining_tokens: int, config: Config) -> tuple[list[dict[str, str]], int]:
        """
        Gets the text from a message and counts the tokens.
        """
        contents: list[dict[str, str]] = []
        tokens = 0
        # when the bot plays characters, it stores text in embeds rather than content
        if message.author.id == self.bot_user_id and message.embeds:
            text = message.embeds[0].description
        else:
            text = message.clean_content

        message_content = {"type": "text", "text": f"{text}"}
        contents.append(message_content)
        tokens += len(message_content["text"]) // self.EST_CHARS_PER_TOKEN

        # Iterate through any attachments associated with the message
        for attachment in message.attachments:
            attachment_contents = await self._read_attachment(attachment)
            if not attachment_contents:
                continue
            openai_content_type, attachment_content = attachment_contents
            content = {}
            if not attachment_content or attachment_content.isspace():
                content = {"type": "text", "text": "\n\nSYSTEM: A file was attached to this message, but it is either empty or is not a file type you can read."}
                tokens += len(content["text"]) // self.EST_CHARS_PER_TOKEN
            elif openai_content_type == "image_url":
                content = {"type": "image_url", "image_url": {"url": attachment_content}}
                # TODO: calculate the number of tokens associated with image
            elif (len(attachment_content) // self.EST_CHARS_PER_TOKEN) > remaining_tokens:
                content = {"type": "text", "text": "\n\nSYSTEM: A file was attached to this message, but it was too large to be read."}
                tokens += len(content["text"]) // self.EST_CHARS_PER_TOKEN
            else:
                content = {"type": "text", "text": f"\n\nSYSTEM: A file called {attachment.filename} was attached to this message. Here are the contents:\n {attachment_content}"}
                tokens += len(content["text"]) // self.EST_CHARS_PER_TOKEN
            contents.append(content)

        return contents, tokens
    
    def _extract_text_from_content(self, content: list[dict[str, str]]) -> str:
        text = ""
        for entry in content:
            if entry["type"] == "text":
                text += entry["text"]
        return text

    def _inject_username(self, message: discord.Message, content: list[dict[str, str]]):
        """
        Adds text that says "Message from [User]" to a content list
        """
        # inject the character's name if they are a character
        if message.author.id == self.bot_user_id and message.embeds and message.embeds[0].footer.text:
            char_data: dict[str, str] = self.characters_database.load_character(message.embeds[0].footer.text)                
            content.insert(0, {"type": "text", "text": f"{char_data["display_name"]}: "})
        # inject You if no character is specified
        elif message.author.id == self.bot_user_id:
            content.insert(0, {"type": "text", "text": f"You: "})
            # TODO: Figure out how to square this with -sp
        # if another user inject the user's name into the prompt so the bot knows it
        else:
            content.insert(0, {"type": "text", "text": f"{message.author.display_name}: "})
