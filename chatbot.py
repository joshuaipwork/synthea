"""
The discord client which contains the bulk of the logic for the chatbot.
"""
import argparse
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

    async def on_reaction_add(self, reaction, user):
        """
        Enables the bot to take a variety of actions when its posts are reacted to

        [❌] will tell the bot to delete its own post
        [▶️] will tell the bot to continue from this post
        """
        # TODO: add this functionality
        # TODO: make the bot add reactions as buttons on its own post

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
        if isinstance(message.channel, discord.Thread):
            thread_start_message: discord.Message = await message.channel.fetch_message(message.channel.id)
            message_in_chatbot_thread: discord.Message = thread_start_message.content.startswith(COMMAND_START_STR)
 
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
                await message.reply(str(err), mention_author=True)
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
            await message.reply(str(err), mention_author=True)
        finally:
            # let the user know that the bot is done with their command.
            await message.remove_reaction("⏳", self.user)

    async def respond_to_command(self, args, message: discord.Message):
        """
        
        """
        character: str = args.character
        prompt: str = " ".join(args.prompt)

        # insert the user's prompt into the prompt template specified by the model
        prompt = "\n".join(self.model.config[''])

        # generate a response and send it to the user.
        if character:
            print(f'Generating with char {character} and prompt {prompt} for {message.author}')
            response = self.model.generate_from_character(
                prompt=prompt,
                character=character
            )
            print(response)
            await self.send_response_as_character(response, character, message)
        else:
            print(f'Generating with prompt {prompt} for {message.author}')
            response = self.model.generate_default(
                prompt=prompt
            )
            print(response)
            await self.send_response_as_base(response, message)       

    async def send_response_as_base(self, prompt, message):
        """
        Sends a simple response using the base template of the model.
        """
        print(f'Generating with prompt {prompt} for from {message.author}')
        output = self.model.generate_default(prompt)
        print(output)
        await message.reply(output, mention_author=True)

    async def respond_to_chatbot_thread(
            self, 
            thread_args: argparse.Namespace, 
            message: discord.Message):
        """
        When responding to a thread, 
        """
        character: str | None = thread_args.character

        # retrieve the character's background information
        starter: str = ""
        if character:
            with open(f'characters/{character}.yaml', "r", encoding='utf-8') as f:
                char_config = yaml.safe_load(f)
                starter = char_config['starter']

        # retrieve previous entries in this thread for this conversation
        context_manager: ContextManager = ContextManager()
        prompt = await context_manager.compile_prompt_from_context(
            seq_len=self.model.config['seq_len'],
            max_new_tokens=self.model.config['max_new_tokens'],
            user_role_tag=self.model.config['user_role_tag'],
            bot_role_tag=self.model.config['bot_role_tag'],
            intro=self.model.config['intro'],
            message=message,
            starter=starter,
        )

        # generate a response and send it to the user.
        if character:
            print(f'Generating with char {character} on thread {message.channel} with prompt {message.content} for {message.author}')
            response = self.model.generate_from_character(
                prompt=prompt,
                character=character
            )
            print(response)
            await self.send_response_as_character(response, character, message)
        else:
            print(f'Generating on thread {message.channel} with prompt {message.content} for {message.author}')
            response = self.model.generate_default(
                prompt=prompt
            )
            print(response)
            await self.send_response_as_base(response, message)

    async def send_response_as_character(self, response: str, character: str, message: discord.Message):
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

            webhook: discord.Webhook = await message.channel.create_webhook(
                name=loaded_config['name'],
                avatar=avatar,
                reason="Chatbot character"
            )

            # send the message via its character
            await webhook.send(response)

            # remove the webhook once done
            await webhook.delete()
