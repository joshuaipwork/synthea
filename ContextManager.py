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
                self.message = await self.message.channel.fetch_message(self.message.reference.message_id)
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

    def __init__(self, model: ChattyModel, bot_user_id: int):
        """        
        model (str): The model that is generating the text. Used to determine the prompt format
            and other configuration options.
        bot_user_id (str): The discord user id of the bot. Used to determine if a message came from
            the bot or from a user.
        """
        self.model: ChattyModel = model
        self.seq_len: int = self.model.config['seqlen']
        # # used to calculate how much context can be included in a message.
        # self.max_new_tokens: int = self.model.config['max_new_tokens']
        # # self.user_message_tag: str = self.model.format['user_message_tag']
        # # self.bot_message_tag: str = self.model.format['bot_message_tag']
        # # self.system_prompt: str = self.model.format['system_prompt']
        self.bot_user_id: int = bot_user_id

        # TODO: Split the model config and the general bot config.
        with open('config.yaml', "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

    async def compile_prompt_from_chat(
            self,
            message: discord.Message,
            character: Optional[str]=None
        ):
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

    async def compile_prompt_from_thread(
            self,
            message: discord.Message,
            character: str | None=None
        ):
        """
        Generates a prompt which includes the context from previous messages in a thread.

        Args:
            character (str): The name of a character to use. If a character is specified, the character file will populate
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
        bot_message_tag = self.model.format['bot_message_tag']
        system_prompt = self.model.config['system_prompt']

        # if the character has a special system prompt, override the default model system prompt
        if character:
            with open(f"characters/{character}.yaml", 'r', encoding='utf-8') as f:
                char_chat_config = yaml.safe_load(f)
                system_prompt = char_chat_config['system_prompt']

        # add the bot role tag so the AI knows to continue from this point.
        history.append(f"{bot_message_tag} ")
        token_count += len(bot_message_tag) // self.EST_CHARS_PER_TOKEN

        # history contains the most recent messages that were sent before the
        # current message.
        system_prompt_tokens: int = len(system_prompt) // self.EST_CHARS_PER_TOKEN
        history_token_limit: int = self.seq_len - self.model.config['max_new_tokens'] - system_prompt_tokens

        # add the user's last message
        user_tag: str = self.model.format['user_message_tag']
        text, added_tokens = self._get_text(message)
        history.append(self._form_message(user_tag, text))
        token_count += added_tokens

        await self._follow_history(
            history,
            token_count=token_count,
            token_limit=history_token_limit,
            history_iterator=history_iterator
        )

        # add the system prompt to the beginning of the prompt
        history.append(self._form_prompt_header(system_prompt))

        # convert the list into a single string
        final_prompt = "".join(reversed(history))
        return final_prompt

    def _form_prompt_header(self, system_prompt: str) -> str:
        """
        Many prompt formats include a header which may include a system prompt,
        along with some special formatting. This function takes in a system
        prompt and formats it into a header according to the model's prompt format.

        Args:
            system_prompts (str): The system prompt of the 
        """
        template = self.model.format['header_template']
        result = template.replace('<|system_prompt|>', system_prompt)

        return result

    def _get_user_tag(self, message: discord.Message) -> str:
        """
        Returns the name of the user who sent this message.
        
        Args:
            message (discord.Message): The message to retrieve the name of
        Returns:
            (str): A name for the user. This function prioritizes
                nicknames over display names, display names over global names,
                and global names over the deprecated names from before the username migration.
                If none of the above, then 'USER' is used.
        """
        user_tag: str = ""
        if not isinstance(message.channel, discord.DMChannel) and message.author.nick:
            user_tag = message.author.nick
        elif message.author.display_name:
            user_tag = message.author.display_name
        elif message.author.global_name:
            user_tag = message.author.global_name
        elif message.author.name:
            user_tag = message.author.name
        else:
            user_tag = 'user'

        return user_tag

    def _form_message(self, message_tag: str, text: str) -> str:
        """
        Generates a formatted message based on the prompt format of the model.
        Args:
            message_tag (str): The tag of the user who sent the message. Typically derived
                from the prompt format and the identity of the user.
            text (str): The text of the message.
        """
        template: str = self.model.format['message_template']
        template = template.replace("<|message_tag|>", message_tag)
        result = template.replace("<|text|>", text)

        return result

    async def _follow_history(
            self,
            history: list[str],
            token_limit: int,
            history_iterator: AsyncIterator[discord.Message],
            token_count: int = 0,

        ):
        """
        Iterates through history_iterator, retrieving .

        If it is a message by a webhook or a bot, it 

        Returns:
            history (list of str): A list of messages to include in the context,
                in order of most recent to oldest messages in the history.
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
                history.append(self._form_message(self.model.format['bot_message_tag'], text))
            else:
                history.append(self._form_message(self.model.format['user_message_tag'], text))

            # TODO: Figure out how to do multi-character conversations. Looks like not every model works well with that...
            # if message.embeds:
            #     # if the bot was speaking as a character, an embed is included with the character
            #     embed = message.embeds[0]
            #     character_name = embed.title.upper()
            #     history.append(self._form_message(self.model.format['bot_message_tag'], text))
            # else:
            #     user_tag: str = self._get_user_tag(message)
            #     history.append(f"{user_tag.upper()}: {text}")
        
        return history

    def _get_text(self, message: discord.Message):
        """
        Gets the text from a message and counts the tokens.

        Under most conditions, the text it returns will be message.content, however if it is a command
        for the bot, then only the prompt from that command will be returned.
        """
        # when the bot plays characters, it stores text in embeds rather than content
        if message.author.id == self.bot_user_id and message.embeds:
            text = message.embeds[0].description
        elif message.content.startswith(self.config['command_start_str']):
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

    