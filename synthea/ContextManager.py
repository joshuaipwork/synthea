# -*- coding: utf-8 -*-
"""
Generate a prompt for the AI to respond to, given the
message history and persona.
"""
from typing import AsyncIterator, Optional
import discord
import yaml
from synthea.SyntheaModel import SyntheaModel

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

    def __init__(self, model: SyntheaModel, bot_user_id: int):
        """
        model (str): The model that is generating the text. Used to determine the prompt format
            and other configuration options.
        bot_user_id (str): The discord user id of the bot. Used to determine if a message came from
            the bot or from a user.
        """
        self.model: SyntheaModel = model
        self.context_length: int = self.model.config["context_length"]
        self.bot_user_id: int = bot_user_id

        # TODO: Split the model config and the general bot config.
        with open("config.yaml", "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

    async def compile_prompt_from_chat(
        self, message: discord.Message, system_prompt: Optional[str] = None
    ):
        """
        Generates a prompt which includes the context from previous messages in a reply chain.
        Messages outside of the reply chain are ignored.

        Args:
            message (discord.Message): The last message from the user.
            system_prompt (str): The system prompt to use when generating the prompt
        """
        history_iterator: ReplyChainIterator = ReplyChainIterator(message)
        final_prompt = await self.compile_prompt(
            message=message,
            history_iterator=history_iterator,
            system_prompt=system_prompt,
        )
        return final_prompt

    async def compile_prompt(
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
        history_token_limit: int = self.context_length - self.model.config["max_new_tokens"]
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
