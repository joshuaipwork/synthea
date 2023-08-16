"""
The discord client which contains the bulk of the logic for the chatbot.
"""
import argparse
import traceback
from typing import Optional
import discord
import yaml

from ChattyModel import ChattyModel
from CommandParser import ChatbotParser, CommandError, ParserExitedException
from ContextManager import ContextManager

COMMAND_START_STR: str = '!syn '
CHAR_LIMIT: int = 2000 # discord's character limit

# This example requires the 'message_content' intent.
class LLMClient(discord.Client):
    """
    A discord client which recieves messages from users. When users send
    messages, the bot parses them and generates messages for them.
    """
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
        if reaction.message.author.id == self.user.id:
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
                response_text=response,
                thread=thread
            )
        else:
            await self.send_response(
                response_text=response,
                message_to_reply=message
            )

    async def send_response_as_character(self, response: str, character: str, message: discord.Message, create_thread: bool=False, thread_name: str=""):
        """
        Sends the given response in the same channel as the given message while
        using the picture and name associated with the character.

        response (str): The response to be 
        """
        with open(f'characters/{character}.yaml', "r", encoding='utf-8') as f:
            char_config = yaml.safe_load(f)

            # create an embed to represent the bot speaking as a character
            embed: discord.Embed = discord.Embed(
                title=char_config['name'],
                description=response,
                color=char_config['color'] if 'color' in char_config else None
            )

            # add a picture via url

            # TODO: Add local file upload options.
            file: Optional[discord.File] = None
            if 'avatar' in char_config:
            #     file = discord.File(f"avatars/{char_config['avatar']}", filename="avatar.png")
                # embed.set_thumbnail(url=f"attachment://avatar.png")
                embed.set_thumbnail(url=char_config['avatar'])

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
                response_text=response,
                embed=embed,
                file=file,
                thread=thread,
                message_to_reply=message,
            )

    async def send_response(
            self,
            message_to_reply: discord.Message=None,
            response_text: Optional[str]=None,
            embed: Optional[discord.Embed]=None,
            file: Optional[discord.Embed]=None,
            thread: Optional[discord.Thread]=None
        ):
        """
        Sends a response, splitting it up into multiple messages if required

        Args:
            message_to_reply (discord.Message): The message that the user sent to invoke the bot.
                the response will be sent in the same channel as this message
                and the author of the message will be tagged in the response.
            response_text (str): The text of the message to be send in its responses.
                If embed is None, then response_text is required.
            embed (discord.Embed or None): The embed to send in the response.
                If response_text is None, then embed is required.
            thread (discord.Thread or None): If provided, the response will be sent in this thread.
        """
        # split up the response into messages and send them individually
        print(f'Response ({len(response_text)} chars):\n{response_text}')

        msg_index = 0
        last_message = None
        if not embed and not response_text:
            raise ValueError("No embed or response text included in the response.")

        # TODO: Respect character limits in embeds.
        if embed:
            await message_to_reply.reply(
                        mention_author=True,
                        content=response_text,
                        embed=embed,
                        file=file
                    )
            return

        while msg_index * CHAR_LIMIT < len(response_text):
            message_text = response_text[msg_index * CHAR_LIMIT:(msg_index + 1) * CHAR_LIMIT]
            if message_to_reply and isinstance(message_to_reply.channel, discord.DMChannel):
                # TODO: remove duplicate code
                if msg_index == 0:
                    last_message = await message_to_reply.reply(
                        message_text,
                        mention_author=True,
                        embed=embed
                    )
                else:
                    last_message = await last_message.reply(
                        message_text,
                        mention_author=False,
                        embed=embed
                    )
            elif thread:
                # It doesn't seem like you can send a message to a thread and reply at the same time
                # TODO: Spend more time verifying that this is the case
                await thread.send(message_text)
            elif message_to_reply:
                # a message in a channel
                # only reply to the first message to prevent spamming
                if msg_index == 0:
                    last_message = await message_to_reply.reply(
                        message_text,
                        mention_author=True,
                        embed=embed,
                    )
                else:
                    last_message = await last_message.reply(
                        message_text,
                        mention_author=False,
                        embed=embed
                    )
            else:
                raise ValueError("No message found to reply to!")
            msg_index += 1

    async def _get_character_replied_to(self, message: discord.Message) -> str | None:
        """
        Args:
            message (discord.Message)
        Returns:
            If the message replied to a message representing a character adopted by the chatbot,
            returns the character's name. 
            
            The name is the filename of the file which stores the character's information,
            as well as the string used with the -c option to invoke the character through a command.

            Otherwise, returns None if the bot was speaking using its default persona. 
        """
        # if the bot was invoked 
        if not message.reference:
            return None
            
        try:
            # bot uses embeds to speak as a character
            replied_message: discord.Message = await message.channel.fetch_message(message.reference.message_id)

            # if no embed, it wasn't speaking as a character
            if not replied_message.embeds:
                return None
            embed = replied_message.embeds[0]

            # cross reference the display name of the character against the name 
            with open('guilds/character_mapping.yaml', mode='r', encoding='utf-8') as file:
                char_mapping = yaml.safe_load(file)
                if replied_message.guild.id not in char_mapping:
                    # the character may have been removed from this guild
                    return None
                if embed.title in char_mapping[replied_message.guild.id]:
                    # the bot played a character, return the character name
                    return char_mapping[message.guild.id][embed.title]
        except (discord.NotFound, discord.HTTPException, discord.Forbidden) as exc:
            print(exc)
        
        return None
