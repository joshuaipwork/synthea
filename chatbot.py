"""
The discord client which contains the bulk of the logic for the chatbot.
"""
import argparse
import traceback
import discord
import yaml

from ChattyModel import ChattyModel
from CommandParser import ChatbotParser, CommandError, ParserExitedException
from ContextManager import ContextManager

COMMAND_START_STR: str = '!syn '

# This example requires the 'message_content' intent.
class LLMClient(discord.Client):
    model: ChattyModel = None
    parser: ChatbotParser = None
    async def setup_hook(self):
        """
        When the bot is started and logs in, load the model.
        """
        self.model = ChattyModel()
        self.model.load_model()

        self.parser = ChatbotParser()

    async def on_ready(self):
        """
        Reports to the console that we logged in.
        """
        print(f'Logged on as {self.user}!')

    async def on_reaction_add(self, reaction: discord.Reaction, user):
        """
        Enables the bot to take a variety of actions when its posts are reacted to

        [🗑️] will tell the bot to delete its own post
        [▶️] will tell the bot to continue from this post
        """
        # TODO: make the bot add reactions as buttons on its own post
        # TODO: Figure out how to distinguish webhooks made by me from webhooks made by someone else
        # TODO: Don't delete messages if the webhook was made by someone else.
        if (reaction.message.webhook_id or reaction.message.author.id == self.user.id):
            print(reaction)
            if (reaction.emoji == '🗑️'):
                await reaction.message.delete()

    async def on_message(self, message: discord.Message):
        """
        Respond to messages sent to the bot. 

        If a message is not by a user or fails to start with the COMMAND_START_STR, then
        the message is ignored.

        Args:

        """
        # prevent bot from responding to itself
        if message.author == self.user:
            return

        # prevent bot from responding to any of its generated webhooks
        if message.webhook_id:
            return

        # by default, don't respond to messages unless it was directed at the bot
        # if the message starts with the start string, then it was definitely directed at the bot.
        message_invokes_chatbot: bool = message.content.startswith(COMMAND_START_STR)

        # if the message is part of a thread created by a chatbot command, we should respond
        message_in_chatbot_thread: bool = False
        if isinstance(message.channel, discord.Thread):
            # TODO: deal with the case that the original message was deleted.
            thread_start_message = await message.channel.parent.fetch_message(message.channel.id)
            message_in_chatbot_thread = thread_start_message.content.startswith(COMMAND_START_STR)
 
        if not (message_invokes_chatbot or message_in_chatbot_thread):
            return

        # the message was meant for the bot and we must respond
        try:
            # let the user know that we are working on their command
            await message.add_reaction("⏳")

            # parse the command
            command: str = message.content
            if message_in_chatbot_thread:
                command = thread_start_message.content
            command = command[len(COMMAND_START_STR):]

            try:
                args = self.parser.parse(command)
            except ParserExitedException as err:
                # If the parser exits prematurely without encountering an error, this indicates that the user
                # invoked the help action. Show them that help text and mark the command as successfully responded.
                await message.reply(f'```{err}```', mention_author=True)
                return
            except CommandError as err:
                # for other types of error, show them the error and mark the command as failed.
                raise err

            # now, respond to the command appropriately.
            if message_in_chatbot_thread:
                await self.respond_to_chatbot_thread(args, message)
            else:
                await self.respond_to_command(args, message)

            # let the user know that we successfully completed their task.
            await message.add_reaction("✅")
        except Exception as err:
            # let the user know that something went wrong
            await message.add_reaction("❌")
            traceback.print_exc(limit=4)
            await message.reply(str(err), mention_author=True)
        finally:
            # let the user know that the bot is done with their command.
            await message.remove_reaction("⏳", self.user)

    async def respond_to_command(self, args, message: discord.Message):
        """
        
        """
        character: str = args.character
        create_thread: bool = args.thread
        thread_name: str = args.thread_name
        text: str = " ".join(args.prompt)

        # insert the user's prompt into the prompt template specified by the model
        context_manager: ContextManager = ContextManager(self.model)

        # generate a response and send it to the user.
        if character:
            prompt = context_manager.compile_prompt_from_character(
                text=text,
                character=character
            )
            
            print(f'Generating for {message.author} with char {character} and prompt: \n{prompt}')
        
            response = self.model.generate_from_character(
                prompt=prompt,
                character=character
            )
            await self.send_response_as_character(response, character, message, create_thread)
        else:
            prompt = context_manager.compile_prompt_from_text(text=text)
            
            print(f'Generating for {message.author} with prompt: \n{prompt}')
            
            response = self.model.generate_from_defaults(prompt=prompt)
            await self.send_response_as_base(response, message, create_thread, thread_name)
        print(f'Response:\n{response}')

    async def respond_to_chatbot_thread(
            self,
            thread_args: argparse.Namespace,
            message: discord.Message):
        """
        When responding to a thread, 
        """
        character: str | None = thread_args.character

        # retrieve previous entries in this thread for this conversation
        context_manager: ContextManager = ContextManager(self.model)
        prompt = await context_manager.compile_prompt_from_chat(
            message=message,
            character=character,
            bot_user_id=self.user.id,
        )

        # generate a response and send it to the user.
        if character:
            print(f'Generating for {message.author} with char {character} on thread {message.channel} with prompt {prompt}')
            response = self.model.generate_from_character(
                prompt=prompt,
                character=character
            )
            await self.send_response_as_character(response, character, message)
        else:
            print(f'Generating for {message.author} on thread {message.channel} with prompt {prompt}')
            response = self.model.generate_from_defaults(
                prompt=prompt
            )
            await self.send_response_as_base(response, message)
        print(f'Response:\n{response}')

    async def send_response_as_base(self, response: str, message: discord.Message, create_thread: bool=False, thread_name: str=""):
        """
        Sends a simple response using the base template of the model.
        If create_thread is True, then a new thread will be created with
        the specified name.
        """
        if create_thread:
            if not thread_name:
                thread_name = f"AI chat with {message.author}"
            thread = await message.create_thread(name=thread_name)
            await thread.send(response)
        else:
            await message.reply(response, mention_author=True)

    async def send_response_as_character(self, response: str, character: str, message: discord.Message, create_thread: bool=False, thread_name: str=""):
        """
        Sends the given response in the same channel as the given message while
        using the picture and name associated with the character.
        """
        with open(f'characters/{character}.yaml', "r", encoding='utf-8') as f:
            loaded_config = yaml.safe_load(f)

            # create a temporary webhook for the bot to speak as its character
            if 'avatar' in loaded_config:
                with open(f"avatars/{loaded_config['avatar']}", "rb") as avatar_file:
                    avatar = avatar_file.read()
            else:
                avatar = None

            webhook_target = message.channel
            if isinstance(message.channel, discord.Thread):
                webhook_target = message.channel.parent

            webhook: discord.Webhook = await webhook_target.create_webhook(
                name=loaded_config['name'],
                avatar=avatar,
                reason="Chatbot character"
            )

            # send the message via its character
            if create_thread:
                if not thread_name:
                    thread_name = f"{message.author}\'s chat with {character}"
                thread = await message.create_thread(name=thread_name)
                await webhook.send(response, thread=thread)
            elif isinstance(message.channel, discord.Thread):
                await webhook.send(response, thread=message.channel)
            else:
                await webhook.send(response)

            # remove the webhook once done
            await webhook.delete()
