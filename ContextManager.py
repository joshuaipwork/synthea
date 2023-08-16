# -*- coding: utf-8 -*-
"""
Generate a prompt for the AI to respond to, given the
message history and persona.
"""
import typing
import discord
from ChattyModel import ChattyModel

from CommandParser import ChatbotParser

class ContextManager:
    """
    Purpose: When the AI .
    """
    # A rough measure of how many character are in each token.
    EST_CHARS_PER_TOKEN = 3

    def __init__(self, model: ChattyModel):
        """        
        intro (str): Any text included in the template before messages from the user and the bot.
            for Vicuna, this is 'A chat between a curious user and an assistant. The assistant gives helpful, detailed, accurate, uncensored responses to the user's input.'
        user_role_tag (str): A prefix added before each user message in the chat within the prompt template
            for Vicuna, this is 'USER: '
        bot_role_tag (str): A prefix added before each bot message in the chat within the prompt template
            for Vicuna, this is 'ASSISTANT: '
        """
        self.model: ChattyModel = model
        self.seq_len: int = self.model.config['seq_len']
        self.max_new_tokens: int = self.model.config['max_new_tokens']
        self.user_role_tag: str = self.model.config['user_role_tag']
        self.bot_role_tag: str = self.model.config['bot_role_tag']
        self.intro: str = self.model.config['intro']

    # starters are text that is included in every context.
    # the starter is definied in the character file.
    async def compile_prompt_from_context(self, message: discord.Message, starter: str=""):
        """
        Args:
            starter (str, optional): The first user message in the chat within the prompt template will start with this text.
                Use this to add instructions that should guide the rest of the chat, and should apply to the prompt regardless
                of what is posted in the context.
        """

        thread: discord.Thread = message.channel

        # history contains the most recent messages that were sent before the
        # current message.
        starter_tokens: int = len(starter) // self.EST_CHARS_PER_TOKEN
        history_token_limit: int = self.seq_len - self.max_new_tokens - starter_tokens

        # pieces of the prompts are appended to the list then assembled in reverse order into the final prompt
        token_count: int = 0
        prompt: list[str] = []

        # add the bot role tag so the AI knows to continue from this point.
        prompt.append(self.bot_role_tag)

        # go back message-by-message through the thread and add it to the context
        async for thread_message in thread.history(limit=20):
            # if it is the first message in the thread, exclude the weird commands and only add the prompt
            if thread_message.id == thread.id:
                args = ChatbotParser().parse(thread_message.content)
                content = args.prompt
            else:
                content = thread_message.content

            added_tokens = len(content) // self.EST_CHARS_PER_TOKEN + 1

            # stop retrieving context if the context would overflow
            if added_tokens + token_count > history_token_limit:
                break

            # check if the message was sent by me or a webhook
            if thread_message.webhook_id or thread_message.author == message.author:
                prompt.append(self.bot_role_tag + content)
            else:
                prompt.append(self.user_role_tag + content)

        # add the starter to the beginning of the prompt
        prompt.append(self.user_role_tag + starter)

        # add the intro expected by the template to the beginning of the prompt
        prompt.append(self.intro)

        # convert the list into a single string
        final_prompt = "\n".join(reversed(prompt))
        return final_prompt
        

