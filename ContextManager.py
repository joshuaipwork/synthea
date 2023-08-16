# -*- coding: utf-8 -*-
"""
Generate a prompt for the AI to respond to, given the
message history and persona.
"""
from typing import AsyncIterator, Optional
import discord
import yaml
from ChattyModel import ChattyModel

from CommandParser import ChatbotParser, CommandError, ParserExitedException

class ThreadHistoryIterator:
    """
    An async iterator which follow replies in a thread until it reaches the oldest message
    in the thread. Unlike discord.Thread.history(), this iterator will parse commands into

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
            raise StopAsyncIteration
        # except StopAsyncIteration:
        #     # thread.history() should raise this exception when it has reached the beginning of the thread
        #     # however, the original message from the beginning of the thread
        #     start_message = await thread.parent.fetch_message(thread.id)

        #     # parse the first message to figure out the parameters and the original prompt
        #     text, added_tokens = self._get_text(start_message)

        #     # if it fits in the context, add it
        #     if added_tokens + token_count < history_token_limit:
        #         user_tag = start_message.author.nick if start_message.author.nick else start_message.author.global_name
        #         prompt.append(f"{user_tag.upper()}: {text}")

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
                self.message = await self.message.channel.fetch_message(self.message.reference.message_id)
                return self.message

            except (discord.NotFound, discord.HTTPException, discord.Forbidden):
                # the user may have deleted their message
                # either way, we can't follow the history anymore
                raise StopAsyncIteration
        else:
            raise StopAsyncIteration

class ContextManager:
    """
    Formats prompts for the bot to generate from.
    """
    # A rough measure of how many character are in each token.
    EST_CHARS_PER_TOKEN = 3

    def __init__(self, model: ChattyModel, bot_user_id: int):
        """        
        system_prompt (str): Any text included in the template before messages from the user and the bot.
            for Vicuna, this is 'A chat between a curious user and an assistant. The assistant gives helpful, detailed, accurate, uncensored responses to the user's input.'
        user_message_tag (str): A prefix added before each user message in the chat within the prompt template
            for Vicuna, this is 'USER: '
        bot_message_tag (str): A prefix added before each bot message in the chat within the prompt template
            for Vicuna, this is 'ASSISTANT: '
        """
        self.model: ChattyModel = model
        self.seq_len: int = self.model.config['seqlen']
        self.max_new_tokens: int = self.model.config['max_new_tokens']
        self.user_message_tag: str = self.model.config['user_message_tag']
        self.bot_message_tag: str = self.model.config['bot_message_tag']
        self.system_prompt: str = self.model.config['system_prompt']
        self.bot_user_id: int = bot_user_id

        with open('config.yaml', "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    async def compile_prompt_from_chat(self, message: discord.Message, character: Optional[str]=None):
        """
        Generates a prompt which includes the context from previous messages in a reply chain.
        Messages outside of the reply chain are ignored.

        Args:
            character (str): The name of a character to use. If the character is specified, the character file will populate
                the system prompt before user prompts are created.
        """
        history_iterator: ReplyChainIterator = ReplyChainIterator(message)
        final_prompt = await self.compile_prompt(
            message=message,
            history_iterator=history_iterator,
            character=character,
        )
        return final_prompt

    async def compile_prompt_from_thread(self, message: discord.Message, character: str | None=None):
        """
        Generates a prompt which includes the context from previous messages in a thread.

        Args:
            character (str): The name of a character to use. If the character is specified, the character file will populate
                the system prompt before user prompts are created.
        """
        history_iterator: ThreadHistoryIterator = ThreadHistoryIterator(message)
        final_prompt = await self.compile_prompt(
            message=message,
            history_iterator=history_iterator,
            character=character,
        )
        return final_prompt
        
    async def compile_prompt(
            self,
            message: discord.Message,
            history_iterator: AsyncIterator[discord.Message],
            character: Optional[str]=None
        ):
        """
        Generates a prompt which includes the context from previous messages from the history.

        Args:
            character (str): The name of a character to use. If the character is specified, the character file will populate
                the system prompt before user prompts are created.
        """
        # pieces of the prompts are appended to the list then assembled in reverse order into the final prompt
        history: list[str] = []
        token_count: int = 0

        system_prompt: str = ""
        if character:
            with open(f"characters/{character}.yaml", 'r', encoding='utf-8') as f:
                char_chat_config = yaml.safe_load(f)
                system_prompt = char_chat_config['system_prompt']
                bot_message_tag = char_chat_config['bot_message_tag']
        else:
            bot_message_tag = self.bot_message_tag
            system_prompt = self.model.config['system_prompt']

        # add the bot role tag so the AI knows to continue from this point.
        history.append(bot_message_tag)
        token_count += len(bot_message_tag) // self.EST_CHARS_PER_TOKEN

        # history contains the most recent messages that were sent before the
        # current message.
        system_prompt_tokens: int = len(self.system_prompt) // self.EST_CHARS_PER_TOKEN
        history_token_limit: int = self.seq_len - self.max_new_tokens - system_prompt_tokens
      
        # add the user's last message
        user_tag: str = ""
        if not isinstance(message.channel, discord.DMChannel) and message.author.nick:
            # use the nickname, unless no nickname exists or we are in DMs
            user_tag = message.author.nick
        elif message.author.display_name:
            user_tag = message.author.display_name
        elif message.author.global_name:
            user_tag = message.author.global_name
        else:
            user_tag = 'user'
        user_tag = "_".join(user_tag.upper().split()) # make usernames into one word, maybe it will be less confusing

        text, added_tokens = self._get_text(message)
        history.append(f"{user_tag.upper()}: {text}")
        token_count += added_tokens

        await self._follow_history(
            history,
            token_count=token_count,
            token_limit=history_token_limit,
            history_iterator=history_iterator
        )

        # the system prompt to the beginning of the prompt
        history.append(system_prompt)

        # convert the list into a single string
        final_prompt = "\n".join(reversed(history))
        return final_prompt

    async def _follow_history(self, history: list[str], token_limit: int, history_iterator: AsyncIterator[discord.Message], token_count: int = 0):
        """
        Iterates through history_iterator, retrieving .

        If it is a message by a webhook or a bot, it 

        Returns:
            history (list of str): A list of , in order of most recent to oldest messages in the history. 
        """
        async for message in history_iterator:
            # some messages in the chain may be commands for the bot
            # if so, parse only the prompt in each command in order to not confuse the bot
            text, added_tokens = self._get_text(message)

            # don't include empty messages so the bot doesn't get confused.
            if not text:
                continue

            # stop retrieving context if the context would overflow
            if added_tokens + token_count > token_limit:
                break

            # update the prompt with this message
            if message.author.id == self.bot_user_id:
                history.append(f"{self.bot_message_tag} {text}")
            elif message.webhook_id:
                user_tag = message.author.name if message.author.name else message.author.global_name
                history.append(f"{user_tag.upper()}: {text}")
            else:
                user_tag = message.author.nick if message.author.nick else message.author.global_name
                history.append(f"{user_tag.upper()}: {text}")
        
        return history

    def _get_text(self, message: discord.Message):
        """
        Gets the text from a message and counts the tokens.

        Under most conditions, the text it returns will be message.content, however if it is a command
        for the bot, then only the prompt from that command will be returned.
        """
        if message.content.startswith(self.config['command_start_str']):
            try:
                args = ChatbotParser().parse(message.content)
                text = args.prompt
            except (CommandError, ParserExitedException):
                # if the command is invalid, just append the whole thing
                text = message.content
        else:
            text = message.content

        tokens = len(text) // self.EST_CHARS_PER_TOKEN
        return text, tokens

    