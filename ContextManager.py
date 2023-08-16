# -*- coding: utf-8 -*-
"""
Generate a prompt for the AI to respond to, given the
message history and persona.
"""
from typing import Optional
import discord
import yaml
from ChattyModel import ChattyModel

from CommandParser import ChatbotParser, CommandError, ParserExitedException

class ContextManager:
    """
    Formats prompts for the bot to generate from.
    """
    # A rough measure of how many character are in each token.
    EST_CHARS_PER_TOKEN = 3

    def __init__(self, model: ChattyModel):
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

        with open('config.yaml', "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    async def compile_prompt_from_chat(self, message: discord.Message, bot_user_id: str, character: Optional[str]=None):
        """
        Generates a prompt which includes the context from previous messages in a reply chain.
        Messages outside of the reply chain are ignored.

        Args:
            character (str): The name of a character to use. If the character is specified, the character file will populate
                the system prompt before user prompts are created.
        """
        # pieces of the prompts are appended to the list then assembled in reverse order into the final prompt
        prompt: list[str] = []
        token_count: int = 0

        system_prompt: str = ""
        if character:
            with open(f"characters/{character}.yaml", 'r', encoding='utf-8') as f:
                char_chat_config = yaml.safe_load(f)
                system_prompt = char_chat_config['system_prompt']
        else:
            system_prompt = self.model.config['system_prompt']

        # add the bot role tag so the AI knows to continue from this point.
        prompt.append(self.bot_message_tag)
        token_count += len(self.bot_message_tag) // self.EST_CHARS_PER_TOKEN

        # history contains the most recent messages that were sent before the
        # current message.
        system_prompt_tokens: int = len(self.system_prompt) // self.EST_CHARS_PER_TOKEN
        history_token_limit: int = self.seq_len - self.max_new_tokens - system_prompt_tokens
      
        # add the user's last message
        user_tag = message.author.nick if message.author.nick else message.author.global_name
        prompt.append(f"{user_tag.upper()}: {message.content}")
        token_count += len(prompt[-1]) // self.EST_CHARS_PER_TOKEN

        # go back message-by-message through the reply chain and add it to the context
        while message.reference:
            try:
                message = await message.channel.fetch_message(message.reference.message_id)
            except discord.NotFound:
                # the user may have deleted their message
                break

            # some messages in the chain may be commands for the bot
            # if so, parse only the prompt in each command in order to not confuse the bot
            text, added_tokens = self._get_text(message)

            # don't include empty messages so the bot doesn't get confused.
            if not text:
                break

            # stop retrieving context if the context would overflow
            if added_tokens + token_count > history_token_limit:
                break

            # update the prompt with this message
            if message.webhook_id or message.author.id == bot_user_id:
                prompt.append(f"{self.bot_message_tag} {text}")
            else:
                user_tag = message.author.nick if message.author.nick else message.author.global_name
                prompt.append(f"{user_tag.upper()}: {text}")
            
            token_count += added_tokens

        # the system prompt to the beginning of the prompt
        prompt.append(system_prompt)

        # convert the list into a single string
        final_prompt = "\n".join(reversed(prompt))
        return final_prompt

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

    async def compile_prompt_from_thread(self, message: discord.Message, bot_user_id: int, character: str | None=None):
        """
        Generates a prompt which includes the context from previous messages in a thread.

        Args:
            character (str): The name of a character to use. If the character is specified, the character file will populate
                the system prompt before user prompts are created.
        """

        system_prompt: str = ""
        if character:
            with open(f"characters/{character}.yaml", 'r', encoding='utf-8') as f:
                char_chat_config = yaml.safe_load(f)
                system_prompt = char_chat_config['system_prompt']
        else:
            system_prompt = self.model.config['system_prompt']

        thread: discord.Thread = message.channel

        # history contains the most recent messages that were sent before the
        # current message.
        starter_tokens: int = len(system_prompt) // self.EST_CHARS_PER_TOKEN
        system_prompt_tokens: int = len(self.system_prompt) // self.EST_CHARS_PER_TOKEN
        history_token_limit: int = self.seq_len - self.max_new_tokens - starter_tokens - system_prompt_tokens

        # pieces of the prompts are appended to the list then assembled in reverse order into the final prompt
        token_count: int = 0
        prompt: list[str] = []

        # add the bot role tag so the AI knows to continue from this point.
        prompt.append(self.bot_message_tag)

        # go back message-by-message through the thread and add it to the context
        async for thread_message in thread.history(limit=20):
            # some messages in the history may be commands for the bot
            # if so, parse only the prompt in each command in order to not confuse the bot
            text, added_tokens = self._get_text(thread_message)

            # don't include empty messages so the bot doesn't get confused.
            if not text:
                break

            # stop retrieving context if the context would overflow
            if added_tokens + token_count > history_token_limit:
                break

            # check if the message was sent by me or a webhook
            if thread_message.webhook_id or thread_message.author.id == bot_user_id:
                prompt.append(f"{self.bot_message_tag} {text}")
            else:
                user_tag = thread_message.author.nick if thread_message.author.nick else thread_message.author.global_name
                prompt.append(f"{user_tag.upper()}: {text}")
            
            token_count += added_tokens

        # parse the first message in the thread. Because discord considers the first message to be part of the TextChannel rather than the Thread,
        # it is not included in the history.
        if token_count < history_token_limit:
            start_message = await thread.parent.fetch_message(thread.id)

            # parse the first message to figure out the parameters and the original prompt
            text, added_tokens = self._get_text(start_message)

            # if it fits in the context, add it
            if added_tokens + token_count < history_token_limit:
                user_tag = start_message.author.nick if start_message.author.nick else start_message.author.global_name
                prompt.append(f"{user_tag.upper()}: {text}")

        # the system prompt to the beginning of the prompt
        prompt.append(system_prompt)

        # convert the list into a single string
        final_prompt = "\n".join(reversed(prompt))
        return final_prompt
        

