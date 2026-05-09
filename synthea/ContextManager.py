# -*- coding: utf-8 -*-
"""
Generate a prompt for the AI to respond to, given the
message history and persona.
"""
import base64
from dataclasses import dataclass
import mimetypes
from typing import AsyncIterator
from urllib.parse import urlparse
import discord
from langchain.messages import AIMessage, HumanMessage
from langchain_core.messages import BaseMessage
import pypdf
import os
import requests

from synthea.CharactersDatabase import CharactersDatabase
from synthea.CommandParser import ChatbotParser, ParsedArgs
from synthea import SyntheaClient
from config import Config

from synthea.utilities import inference_logger

from synthea.model_definition import ModelDefinition

@dataclass
class ChatHistory:
    """
    A class representing the state of a chat log. 
    """
    # the list of cleaned messages in the chat, ordered from earliest to latest 
    messages: list[BaseMessage] = None

    # the set of arguments applied to the last message in the chat
    args: ParsedArgs = None

    # if not None, the system prompt that currently applies to this chat
    system_prompt_override: str | None = None

    # if true, the bot should create a system message rather than reply as itself
    create_system_prompt_message: bool = False

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
    REASONING_TXT_FILE_NAME: str = 'bot_thinking.txt'

    def __init__(self, bot_user_id: int):
        """
        model (str): The model that is generating the text. Used to determine the prompt format
            and other configuration options.
        bot_user_id (str): The discord user id of the bot. Used to determine if a message came from
            the bot or from a user.
        """
        self.parser: ChatbotParser = ChatbotParser()
        self.characters_database: CharactersDatabase = CharactersDatabase()
        self.bot_user_id: int = bot_user_id

    async def generate_chat_history_from_chat(
        self, message: discord.Message,
        model_definition: ModelDefinition = None,
    ) -> ChatHistory:
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
        chat_history = await self.compile_chat_history(
            message=message,
            history_iterator=history_iterator
        )

        return chat_history

    async def get_args_from_chat_history(self, message: discord.Message) -> ParsedArgs | None:
        """
        Searches a reply chain for the last command that the user sent.

        Returns:
            args: a ParsedArgs representing the most recent command in the
                chat history
        """
        history_iterator: ReplyChainIterator = ReplyChainIterator(message)
        config = Config()

        # retrieve as many tokens as can fit into the context length from history
        async for message in history_iterator:
            # some messages in the chain may be commands for the bot
            # if so, parse only the prompt in each command in order to not confuse the bot
            raw_content, _ = await self._get_content(message, None, read_attachments=False)

            # merge all the text in the message and track the number of tokens
            text: str = self._extract_text_from_content(raw_content)

            # if the message is a command, parse it
            if text.lower().startswith(config.command_start_str.lower()):
                message_args: ParsedArgs = self.parser.parse(text)
                return message_args
        
        return None

    async def compile_chat_history(
        self,
        message: discord.Message,
        history_iterator: AsyncIterator[discord.Message],
    ) -> ChatHistory:
        """
        Generates an OpenAI completion endpoint compatible messages object from a reply chain,
        and returns the command arguments that apply to the current chat context.

        Walks the reply chain from newest to oldest, accumulating messages until the context
        window is full. If a command with `use_as_system_prompt` is encountered, its prompt
        is extracted as the system prompt and the command message is excluded from the history.
        System embed messages (structural messages posted by the bot) are also excluded from
        the history, but their presence is used to set `should_reply_to_system_message` on
        the returned state.

        Args:
            message (discord.Message): The most recent message in the reply chain, which
                triggered this generation.
            history_iterator (AsyncIterator[discord.Message]): An async iterator that yields
                messages from the reply chain, starting from `message` and walking backwards.
            model_definition (ModelDefinition): The model configuration, used to determine
                capabilities such as vision support when processing attachments.

        Returns:
            ChatState: A dataclass containing:
                - messages: The compiled chat history as a list of LangChain `AIMessage` and
                `HumanMessage` objects, ordered from oldest to newest.
                - args: The `ParsedArgs` from the most recent command found in the reply chain,
                or `None` if no command was found.
                - should_generate_system_message: `True` if the user's last command creates a
                system message, and that system message has not yet been created.
        """
        config = Config()

        token_count: int = 0

        chat_history: ChatHistory = ChatHistory()
        chat_history.messages = []

        history_token_limit: int = config.context_length - config.max_new_tokens
        
        is_last_message: bool = True
        async for message in history_iterator:
            raw_content, added_tokens = await self._get_content(
                message, history_token_limit - token_count, read_attachments=True
            )
            text: str = self._extract_text_from_content(raw_content)

            # if the message is a command, parse it
            if text.lower().startswith(config.command_start_str.lower()):
                message_args: ParsedArgs = self.parser.parse(text)
                if not chat_history.args:
                    chat_history.args = message_args
                    if is_last_message and message_args.use_as_system_prompt:
                        chat_history.system_prompt_override = chat_history.args.prompt
                        chat_history.create_system_prompt_message = True
                        is_last_message = False
                        continue
                if message_args.use_as_system_prompt:
                    # if the command sets the system prompt, don't include it in the history
                    continue

                # clean the command to only include the prompt parameter
                text = message_args.prompt
                added_tokens = len(message_args.prompt) // self.EST_CHARS_PER_TOKEN

            is_last_message = False

            # skip system embed messages from chat history (they're structural, not conversational)
            if (
                message.author.id == self.bot_user_id
                and message.embeds
                and message.embeds[0].footer.text == SyntheaClient.SYSTEM_TAG
            ):
                continue

            # don't include empty messages so the bot doesn't get confused.
            if not raw_content:
                continue

            # stop retrieving context if the context would overflow
            if added_tokens + token_count > history_token_limit:
                break

            # add the username to the text for multi-user multi-turn conversations
            user_id, user_display_name = self._get_message_sender_details(message)
            text = f"{user_display_name}: {text}"

            # compile the chat history into langgraph-compatible messages
            content = []
            if text:
                content.append({"type": "text", "text": text})
            
            for entry in raw_content:
                if entry.get("type") == "text":
                    continue
                content.append({"type": entry.get("type"), entry.get("type"): entry.get(entry.get("type"))})

            if message.author.id == self.bot_user_id:
                chat_history.messages.insert(0, AIMessage(content=content, name=str(user_id)))
            else:
                chat_history.messages.insert(0, HumanMessage(content=content, name=str(user_id)))

            token_count += added_tokens

        return chat_history

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
        openai_content_type = ""
        attachment_string = ""
        attachment_bytes = await attachment.read()
        if not attachment.content_type or attachment.content_type.startswith("text/"):
            openai_content_type = "text"
            if (attachment.filename != ContextManager.REASONING_TXT_FILE_NAME):
                attachment_string = attachment_bytes.decode()
            else:
                inference_logger.info("Skipping txt file because it contains bot reasoning.")
        elif "application/pdf" in attachment.content_type:
            inference_logger.info("Saving the pdf attachment")
            openai_content_type = "text"
            await attachment.save(attachment.filename)
            reader = pypdf.PdfReader(attachment.filename)

            inference_logger.info(f"Found {len(reader.pages)} pages in PDF. Reading them.")
            for page in reader.pages:
                page_text = page.extract_text()
                attachment_string = attachment_string + "\n" + page_text
            
            inference_logger.info("Removing the saved file")
            os.remove(attachment.filename)
        elif attachment.content_type.startswith("image/"):
            # if not model_definition.vision:
            #     inference_logger.info("Skipped processing attached image since the model cannot process images.")
            #     return None
            inference_logger.info("Found image attachment")
            # just incldue the image url
            openai_content_type = "image_url"
            attachment_string = attachment.url

        inference_logger.info(f"Obtained the text from the [{attachment.content_type}] attachment as a string")
        inference_logger.info(f"Recorded as ({openai_content_type}, {attachment_string})")
        return (openai_content_type, attachment_string)

    async def _get_linked_content(self, message: discord.Message, remaining_tokens: int, config: Config) -> tuple[list[dict[str, str]], int]:
        """
        Gets 
        """

    async def _get_content(self, message: discord.Message, remaining_tokens: int, read_attachments: bool=False) -> tuple[list[dict[str, str]], int]:
        """
        Gets the text and attachments from a message and counts the tokens.
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
        if read_attachments:
            for attachment in message.attachments:
                attachment_contents = await self._read_attachment(attachment)
                if not attachment_contents:
                    continue
                openai_content_type, attachment_content = attachment_contents
                content = {}
                if attachment.filename == ContextManager.REASONING_TXT_FILE_NAME:
                    # dont attach reasoning so the bot doesn't get clogged by its own thoughts
                    continue
                elif not attachment_content or attachment_content.isspace():
                    content = {"type": "text", "text": "\n\nSYSTEM: A file was attached to this message, but it is either empty or is not a file type you can read."}
                    tokens += len(content["text"]) // self.EST_CHARS_PER_TOKEN
                elif openai_content_type == "image_url":
                    content = {"type": "image_url", "image_url": {"url": self.image_to_base64(attachment_content)}}
                    # TODO: calculate the number of tokens associated with image
                elif remaining_tokens is not None and (len(attachment_content) // self.EST_CHARS_PER_TOKEN) > remaining_tokens:
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

    def _get_message_sender_details(self, message: discord.Message) -> tuple[str, str]:
        """
        Gets the user ID and display name of the person sending the message

        Returns:
            a tuple of the id and display_name
        """
        # get the character's name if they are a character
        if message.author.id == self.bot_user_id and message.embeds and message.embeds[0].footer and message.embeds[0].footer.text != SyntheaClient.SYSTEM_TAG:
            char_id: str = message.embeds[0].footer.text
            char_data: dict[str, str] = self.characters_database.load_character(message.embeds[0].footer.text)
            return char_id, char_data["display_name"]
        # inject You if no character is specified
        elif message.author.id == self.bot_user_id:
            return self.bot_user_id, "You"
            # TODO: Figure out how to square this with -sp
        # if another user inject the user's name into the prompt so the bot knows it
        else:
            return message.author.id, message.author.display_name

    def image_to_base64(self, image_path):
        """
        Downloads an image and converts into base 64 with a mimetype declaration.
        """
        # Check if the path is a URL
        is_url = urlparse(image_path).scheme in ['http', 'https']
        
        # Get the content and mime type
        if is_url:
            response = requests.get(image_path)
            image_content = response.content
            # Try to get mime type from response headers first
            mime_type = response.headers.get('content-type')
            if not mime_type:
                # Fallback to guessing from URL
                mime_type = mimetypes.guess_type(image_path)[0]
        else:
            with open(image_path, 'rb') as img_file:
                image_content = img_file.read()
            mime_type = mimetypes.guess_type(image_path)[0]
        
        # If mime type still couldn't be determined, default to jpeg
        if mime_type is None:
            mime_type = 'image/jpeg'
        
        # Convert to base64
        b64_string = base64.b64encode(image_content).decode('utf-8')
        return f'data:{mime_type};base64,{b64_string}'