# -*- coding: utf-8 -*-
"""
Generate a prompt for the AI to respond to, given the
message history and persona.
"""
from typing import AsyncIterator, Optional
import discord
from jinja2 import Environment
import yaml

from synthea.CommandParser import ChatbotParser, CommandError, ParserExitedException


class ThreadHistoryIterator:
    """
    An async iterator which follow replies in a thread until it reaches the oldest message
    in the thread. Unlike discord.Thread.history(), this iterator will parse commands into
    a chat history.
    """

    def __init__(self, starting_message: discord.Message):
        self.message = starting_message
        self.thread: discord.Thread = self.message.channel
        self.thread_history = self.thread.history(limit=50)

    def __aiter__(self):
        return self

    async def __anext__(self):
        # if we have already reached the beginning of the thread, there are no further messages to read
        if self.message.id == self.thread.id:
            raise StopAsyncIteration

        try:
            self.message = await self.thread_history.__anext__()
            # self.message = await self.message.channel.fetch_message(self.message.reference.message_id)
            return self.message
        except (discord.NotFound, discord.HTTPException, discord.Forbidden):
            # the user may have deleted their message
            # either way, we can't follow the history anymore
            # pylint: disable-next=raise-missing-from
            raise StopAsyncIteration


class ReplyChainIterator:
    """
    An async iterator which follows a chain of discord message replies until it reaches the end
    or fails to capture the last message.
    """

    def __init__(self, starting_message: discord.Message):
        self.message = starting_message

    def __aiter__(self):
        return self

    async def __anext__(self):
        # go back message-by-message through the reply chain and add it to the context
        if self.message.reference:
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
    EST_CHARS_PER_TOKEN = 3

    def __init__(self, bot_user_id: int):
        """
        model (str): The model that is generating the text. Used to determine the prompt format
            and other configuration options.
        bot_user_id (str): The discord user id of the bot. Used to determine if a message came from
            the bot or from a user.
        """
        # TODO: Split the model config and the general bot config.
        with open("config.yaml", "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

        self.context_length: int = self.config["context_length"]
        self.bot_user_id: int = bot_user_id

    async def generate_prompt_from_chat(
        self, message: discord.Message, system_prompt: Optional[str] = None
    ) -> str:
        """
        Generates a prompt which includes the context from previous messages in a reply chain.
        Messages outside of the reply chain are ignored.

        Args:
            message (discord.Message): The last message from the user.
            system_prompt (str): The system prompt to use when generating the prompt
        """
        history_iterator: ReplyChainIterator = ReplyChainIterator(message)
        chat_history: list[dict[str, str]] = await self.compile_chat_history(
            message=message,
            history_iterator=history_iterator,
            system_prompt=system_prompt,
        )

        # load chat template from config
        if ("chat_template" in self.config):
            chat_template = self.config["chat_template"]
        else:
            print("Unable to find chat template in config, using ChatML.")
            chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}<|im_start|>assistant\n"

        prompt: str = await self.convert_chat_history_to_prompt(chat_history, chat_template)

        return prompt

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
        system_prompt: Optional[str] = None,
    ) -> list[dict[str, str]]:
        """
        Generates a prompt which includes the context from previous messages from the history.

        Args:
            message (discord.Message): The last message to add to the prompt.
            history_iterator (ReplyChainIterator): An iterator that contains the chat history
                to be included in the prompt.
            system_prompt (str): The system prompt to use when generating the prompt
        """
        # pieces of the prompts are appended to the list then assembled in reverse order into the final prompt
        token_count: int = 0

        # use provided system prompt
        if not system_prompt:
            system_prompt = self.config["system_prompt"]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self._get_text(message)[0]}
        ]

        # retrieve as many tokens as can fit into the context length from history
        history_token_limit: int = self.context_length - self.config["max_new_tokens"]
        system_prompt_tokens: int = len(system_prompt) // self.EST_CHARS_PER_TOKEN
        token_count += system_prompt_tokens
        async for message in history_iterator:
            # some messages in the chain may be commands for the bot
            # if so, parse only the prompt in each command in order to not confuse the bot
            text, added_tokens = self._get_text(message)

            # don't include empty messages so the bot doesn't get confused.
            if not text:
                continue

            # stop retrieving context if the context would overflow
            if added_tokens + token_count > history_token_limit:
                break

            # update the prompt with this message
            if message.author.id == self.bot_user_id:
                messages.insert(1, {"role": "assistant", "content": text})
            else:
                messages.insert(1, {"role": "user", "content": text})
            
            token_count += added_tokens

        return messages

    def _get_text(self, message: discord.Message):
        """
        Gets the text from a message and counts the tokens.

        Under most conditions, the text it returns will be message.content, however if it is a command
        for the bot, then only the prompt from that command will be returned.
        """
        # when the bot plays characters, it stores text in embeds rather than content
        if message.author.id == self.bot_user_id and message.embeds:
            text = message.embeds[0].description
        elif message.clean_content.startswith(self.config["command_start_str"]):
            try:
                args = ChatbotParser().parse(message.clean_content)
                text = args.prompt
            except (CommandError, ParserExitedException):
                # if the command is invalid, just append the whole thing
                text = message.clean_content
        else:
            text = message.clean_content

        tokens = len(text) // self.EST_CHARS_PER_TOKEN
        return text, tokens
