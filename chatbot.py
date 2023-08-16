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
CHAR_LIMIT: int = 2000 # discord's character limit

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
        message_invokes_chatbot: bool = False
        if message.content.startswith(COMMAND_START_STR):
            # if the message starts with the start string, then it was definitely directed at the bot.
            message_invokes_chatbot = True
        elif message.reference:
            # if the message replied to the bot, then it was directed at the bot.
            try:
                replied_message: discord.Message = await message.channel.fetch_message(message.reference.message_id)
                if replied_message.author.id == self.user.id:
                    message_invokes_chatbot = True
            except (discord.NotFound, discord.HTTPException, discord.Forbidden) as exc:
                print(exc)
            character_replied_to = await self._get_character_replied_to(message)
            if character_replied_to:
                # check if this webhook represents a character that the chatbot adopted
                message_invokes_chatbot = True

        # if the message is part of a thread created by a chatbot command, we should respond
        message_in_chatbot_thread: bool = False
        if isinstance(message.channel, discord.Thread):
            # TODO: deal with the case that the original message was deleted.
            replied_message = await message.channel.parent.fetch_message(message.channel.id)
            message_in_chatbot_thread = replied_message.content.startswith(COMMAND_START_STR)
 
        if not (message_invokes_chatbot or message_in_chatbot_thread):
            return

        # the message was meant for the bot and we must respond
        try:
            # let the user know that we are working on their command
            await message.add_reaction("⏳")

            # figure out where the parameters are so the bot can respond with the correct character or model
            command: str = message.content
            if message_in_chatbot_thread:
                # the first message in a thread has the parameters for the conversation
                command = replied_message.content

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
        prompt: str = args.prompt

        # if the user responded to the bot playing a character, respond as that character
        replied_character = await self._get_character_replied_to(message)
        if replied_character:
            character = replied_character

        # insert the user's prompt into the prompt template specified by the model
        context_manager: ContextManager = ContextManager(self.model, self.user.id)

        # generate a response and send it to the user.
        prompt = await context_manager.compile_prompt_from_chat(
            message=message,
            character=character,
        )

        if character:
            print(f'Generating for {message.author} with char {character} and prompt: \n{prompt}')
            response = self.model.generate_from_character(
                prompt=prompt,
                character=character
            )
            await self.send_response_as_character(response, character, message, create_thread)
        else:
            print(f'Generating for {message.author} with prompt: \n{prompt}')
            response = self.model.generate_from_defaults(prompt=prompt)
            await self.send_response_as_base(response, message, create_thread, thread_name)

    async def respond_to_chatbot_thread(
            self,
            thread_args: argparse.Namespace,
            message: discord.Message):
        """
        Respond to a thread started by a chatbot command.
        Unlike regular conversations in a channel, the bot will read the context of a thread
        and integrate it into its response.

        thread_args (argparse.Namespace):
        message (discord.Message):
        """
        character: str | None = thread_args.character

        # retrieve previous entries in this thread for this conversation
        context_manager: ContextManager = ContextManager(self.model, self.user.id)
        prompt = await context_manager.compile_prompt_from_thread(
            message=message,
            character=character,
        )

        # generate a response and send it to the user.
        if character:
            print(f'Generating for {message.author} with char {character} on thread {message.channel} with prompt ({len(prompt)} chars):\n {prompt}')
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

            await self.send_response(
                response,
                thread=thread
            )
        else:
            await self.send_response(
                response,
                message_to_reply=message
            )

    async def send_response_as_character(self, response: str, character: str, message: discord.Message, create_thread: bool=False, thread_name: str=""):
        """
        Sends the given response in the same channel as the given message while
        using the picture and name associated with the character.

        response (str): The response to be 
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

            # if this message was in a thread, figure out which thread it was
            thread: discord.Thread | None = None
            if create_thread:
                if not thread_name:
                    thread_name = f"{message.author}\'s chat with {character}"
                thread = await message.create_thread(name=thread_name)
            elif isinstance(message.channel, discord.Thread):
                thread = message.channel

            # send the messages
            await self.send_response(
                response,
                webhook=webhook,
                thread=thread
            )

            # remove the webhook once done
            await webhook.delete()

    async def send_response(self, response_text, message_to_reply: discord.Message | None=None, webhook: discord.Webhook | None=None, thread: discord.Thread | None=None):
        """
        Sends a response, splitting it up into multiple messages if required

        Args:
            response_text (str): The text of the message to be send in its responses
            message_to_reply (discord.Message): If provided, the response will be sent in the same channel
                as this message and the author of the message will be tagged in the response.
            webhook (discord.Webhook): If provided, the bot will respond using this webhook instead of as itself.
                Note that webhooks cannot reply to messages, so if passed message_to_reply is ignored.
                Webhooks also cannot speak in DMs, so if it is a DM, the webhook will be ignored.
            thread (discord.Thread): If provided, the response will be sent in this thread.
        """
        # split up the response into messages and send them individually
        print(f'Response ({len(response_text)} chars):\n{response_text}')

        msg_index = 0
        while msg_index * CHAR_LIMIT < len(response_text):
            message_text = response_text[msg_index * CHAR_LIMIT:(msg_index + 1) * CHAR_LIMIT]
            if message_to_reply and isinstance(message_to_reply.channel, discord.DMChannel):
                # TODO: remove duplicate code
                if msg_index == 0:
                    await message_to_reply.reply(message_text, mention_author=True)
                else:
                    await message_to_reply.channel.send(message_text, mention_author=True)
            elif webhook:
                # webhooks can't reply to anything, so message_to_reply is ignored
                if thread:
                    await webhook.send(message_text, thread=thread)
                else:
                    await webhook.send(message_text)
            elif thread:
                # It doesn't seem like you can send a message to a thread and reply at the same time
                # TODO: Spend more time verifying that this is the case
                await thread.send(message_text)
            elif message_to_reply:
                # a message in a channel
                # only reply to the first message to prevent spamming
                if msg_index == 0:
                    await message_to_reply.reply(message_text, mention_author=True)
                else:
                    await message_to_reply.channel.send(message_text, mention_author=True)
            else:
                raise ValueError("No message found to reply to!")
            msg_index += 1

    async def _get_character_replied_to(self, message: discord.Message) -> str | None:
        """
        Args:
            message (discord.Message)
        Returns:
            If the message replied to a webhook representing a character adopted by the chatbot, 
            returns the character's name. The name is the
            filename of the file which stores the character's information, as well as the string
            used with the -c option to invoke the character through a command.

            Otherwise, returns None
        """
        if not message.reference:
            return None
            
        try:
            replied_message: discord.Message = await message.channel.fetch_message(message.reference.message_id)
            if not replied_message.webhook_id:
                return None

            with open('guilds/character_mapping.yaml', mode='r', encoding='utf-8') as file:
                char_mapping = yaml.safe_load(file)
                if replied_message.guild.id not in char_mapping:
                    # the webhook isn't one from the bot
                    return None
                if replied_message.author.name in char_mapping[replied_message.guild.id]:
                    # the webhook is from the bot, return the character name
                    return char_mapping[message.guild.id][replied_message.author.name]
        except (discord.NotFound, discord.HTTPException, discord.Forbidden) as exc:
            print(exc)
        
        return None
