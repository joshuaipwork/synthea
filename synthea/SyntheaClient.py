"""
The discord client which contains the bulk of the logic for the chatbot.
"""
import multiprocessing
import time
import traceback
from typing import Optional
import discord
from discord import app_commands
import yaml
import asyncio
import janus
from synthea import SyntheaUtilities

from synthea.CharactersDatabase import CharactersDatabase

from synthea.SyntheaModel import SyntheaModel
from synthea.CommandParser import ChatbotParser
from synthea.ContextManager import ContextManager
from synthea.dtos.GenerationRequest import GenerationRequest
from synthea.dtos.ResponseUpdate import ResponseUpdate
from synthea.modals import CharCreationView, UpdateCharModal, CharCreationStep
from synthea.character_errors import (
    CharacterNotFoundError,
    CharacterNotOnServerError,
    ForbiddenCharacterError,
)

COMMAND_START_STR: str = "!syn "
CHAR_LIMIT: int = 2000  # discord's character limit


# This example requires the 'message_content' intent.
class SyntheaClient(discord.Client):
    """
    A discord client which recieves messages from users. When users send
    messages, the bot parses them and generates messages for them.
    """

    char_db: CharactersDatabase = None
    synced = False
    tree: app_commands.CommandTree = None

    # increments each time we respond to a user. used for the next index in in_progress_response
    response_index: int = 0

    # a message chain corresponding to a response to the user. The first element in the list
    # is the original message from the user, and following elements are the response to the
    # user, possibly spanning multiple messages.
    response_message_chains: dict[int, list[discord.Message]] = {}

    in_progress_responses: set[int] = set()

    # a map of user messages to the response index the message corresponds to.
    # used for stopping generations prematurely.
    message_id_to_response_index: dict[int, int] = {}

    def __init__(self, intents, request_queue: multiprocessing.Queue, response_queue: multiprocessing.Queue):
        super().__init__(intents=intents)
        # a queue to send user prompts to SyntheaServer
        self.request_queue: multiprocessing.Queue = request_queue
        # a queue to receive streamed inferences from SyntheaServer
        self.response_queue: multiprocessing.Queue = response_queue

        self.char_db = CharactersDatabase()

    # async def setup_hook(self):
    #     """
    #     When the bot is started and logs in, load the model.
    #     """

    async def on_ready(self):
        """
        Reports to the console that we logged in.
        """
        with open("config.yaml", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
            await self.change_presence(activity=discord.Game(name=config["activity"]))

        # sync slash commands only the first time that we are ready
        if not self.synced:
            await self.tree.sync()
            self.synced = True
            print("Synced command tree")

        await self.stream_responses()

        print(f"Logged on as {self.user}!")

    async def stream_responses(self) -> None:
        while True:
            message_update: ResponseUpdate = await asyncio.to_thread(self.response_queue.get)  # Wait for a message

            # if the response was aborted by user or previously errored, no need to update anything
            if message_update.response_index not in self.in_progress_responses:
                continue

            split_text: list[str] = SyntheaUtilities.split_text(message_update.new_message)
            message_chain: list[discord.Message] = self.response_message_chains[message_update.response_index]

            # if generation encounters an error, let the user know and cancel further generations
            if message_update.error is not None:
                await message_chain[0].remove_reaction("⏳", self.user)
                await message_chain[0].remove_reaction("📝", self.user)
                await message_chain[0].add_reaction("❌")
                await message_chain[0].reply(message_update.error, mention_author=True)
                traceback.print_exc(limit=4)
                self.in_progress_responses.discard(message_update.response_index)
                continue

            # if this is the first message we've sent to the user, create a new message and let the user know we're working on it
            if len(message_chain) == 1:
                await message_chain[0].remove_reaction("⏳", self.user)
                await message_chain[0].add_reaction("📝")
                await message_chain[0].add_reaction("🗑️")
                message_chain.append(await message_chain[0].reply(split_text[-1]))

            # if we've overflown the length of the most recent message, create a new message
            elif len(message_chain) < len(split_text) + 1:
                await message_chain[-1].edit(content=split_text[-2])
                message_chain.append(await message_chain[-1].reply(split_text[-1]))

            # otherwise, update the most recent message with the new text
            else:
                await message_chain[-1].edit(content=split_text[-1])

            if message_update.message_is_completed:
                # update the original message from the user with a completed emoji
                try:
                    await message_chain[0].remove_reaction("📝", self.user)
                    await message_chain[0].add_reaction("✅")
                except Exception as e:
                    continue
                # since message is completed, no need to continue responding to it
                self.in_progress_responses.discard(message_update.response_index)

    async def on_reaction_add(self, reaction: discord.Reaction, _user):
        """
        Enables the bot to take a variety of actions when its posts are reacted to

        [🗑️] will tell the bot to delete its own post
        [▶️] will tell the bot to stop generating
        """
        # TODO: make the bot add reactions as buttons on its own post
        # TODO: Figure out how to distinguish webhooks made by me from webhooks made by someone else
        # TODO: Don't delete messages if the webhook was made by someone else.
        if reaction.message.author.id == self.user.id:
            if reaction.emoji == "🗑️":
                await reaction.message.delete()

        if reaction.message.id in self.message_id_to_response_index:
            response_id: int = self.message_id_to_response_index[reaction.message.id]
            response_chain: list[discord.Message] = self.response_message_chains[response_id]

            if reaction.emoji == "🗑️":
                self.in_progress_responses.discard(response_id)
                # delete all messages in the chain, starting with the last
                for message in reversed(response_chain[1:]):
                    await message.delete()
                del self.response_message_chains[response_id]
                del self.message_id_to_response_index[reaction.message.id]
                await reaction.message.remove_reaction("📝", self.user)
                await reaction.message.remove_reaction("⏳", self.user)
                await reaction.message.add_reaction("⚠️")
            if reaction.emoji == "🛑":
                self.in_progress_responses.discard(response_id)
                await reaction.message.remove_reaction("📝", self.user)
                await reaction.message.remove_reaction("⏳", self.user)
                await reaction.message.add_reaction("⚠️")
            # if reaction.emoji == "🔁":
            #     # TODO: regenerate the response.
            #     await reaction.message.delete()
            #     for message in reversed(response_chain[:1]):
            #         await message.delete()

    async def on_message(self, message: discord.Message):
        """
        Respond to messages sent to the bot.

        If a message is not by a user or fails to start with the COMMAND_START_STR, then
        the message is ignored.
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
                replied_message: discord.Message = await message.channel.fetch_message(
                    message.reference.message_id
                )
                if replied_message.author.id == self.user.id:
                    message_invokes_chatbot = True
            except (discord.NotFound, discord.HTTPException, discord.Forbidden) as exc:
                print(exc)
            character_replied_to = await self._get_character_replied_to(message)
            if character_replied_to:
                # check if this webhook represents a character that the chatbot adopted
                message_invokes_chatbot = True

        if not message_invokes_chatbot:
            return

        # the message was meant for the bot and we must respond
        try:
            await message.add_reaction("🛑")
            await message.add_reaction("🗑️")
            await message.add_reaction("⏳")
            await self.respond_to_user(message)

        # if error, let the user know what went wrong
        # pylint: disable-next=broad-exception-caught
        except Exception as err:
            await message.add_reaction("❌")
            traceback.print_exc(limit=4)
            await message.reply(f"❌ {err}", mention_author=True)

    async def respond_to_user(self, message_from_user: discord.Message):
        """
        Generates and send a response to a user message from the chatbot

        Args:
            message (str): The message to respond to
        """

        # parse the command the user sent
        command: str = message_from_user.clean_content
        parser = ChatbotParser()
        args = parser.parse(command)

        context_manager = ContextManager(self.user.id)

        char_id: str = args.character
        prompt: str = args.prompt

        # if the user responded to the bot playing a character, respond as that character
        replied_char_id = await self._get_character_replied_to(message_from_user)
        if replied_char_id:
            char_id = replied_char_id

        # send a response as a character
        # if char_id:
        #     can_access = self.char_db.can_access_character(
        #         char_id=char_id,
        #         user_id=message.author.id,
        #         server_id=message.guild.id if message.guild else None,
        #     )
        #     if not can_access:
        #         raise CharacterNotOnServerError()

        #     char_data = self.char_db.load_character(char_id)
        #     prompt = await context_manager.compile_prompt_from_chat(
        #         message, system_prompt=char_data["system_prompt"]
        #     )

        #     print(f"Resp for {message.author} with char {char_id}")
        #     response = await self.queue_for_generation(prompt)
        #     # await self.send_response_as_character(response, char_data, message)
        # # send a simple plaintext response without adopting a character
        # else:
        # generate a response and send it to the user.
        prompt = await context_manager.generate_prompt_from_chat(message_from_user)
        print(f"Generating for {message_from_user.author}")
        await self.queue_for_generation(message_from_user, prompt)
        # await self.send_response_as_base(response, message)

    async def queue_for_generation(self, message_from_user: discord.Message, prompt: str) -> None:
        """
        Sends a prompt to the generation queue. When the server is available,
        it will take up the prompt and generate a response.
        """
        # compile the prompt from the chat history
        self.response_message_chains[self.response_index] = [message_from_user]
        self.in_progress_responses.add(self.response_index)
        self.message_id_to_response_index[message_from_user.id] = self.response_index

        self.request_queue.put(
            GenerationRequest(
                self.response_index,
                prompt
            )
        )

        self.response_index += 1

    async def send_response_as_base(self, response: str, message: discord.Message):
        """
        Sends a simple response using the base template of the model.
        """
        await self.send_response(response_text=response, message_to_reply=message)

    async def send_response_as_character(
        self, response: str, char_data: dict[str, str], message: discord.Message
    ):
        """
        Sends the given response in the same channel as the given message while
        using the picture and name associated with the character.

        response (str): The response to be sent
        """
        if not char_data:
            raise CharacterNotFoundError()

        # create an embed to represent the bot speaking as a character
        char_name = (
            char_data["display_name"]
            if "display_name" in char_data
            else char_data["id"]
        )
        embed: discord.Embed = discord.Embed(
            title=char_name,
            description=response,
        )

        # add a picture via url
        if "avatar_link" in char_data:
            embed.set_thumbnail(url=char_data["avatar_link"])

        # add the id to the footer so the bot knows what char sent this
        embed.set_footer(text=char_data["id"])

        # send the message with embed
        await self.send_response(
            response_text=response,
            embed=embed,
            message_to_reply=message,
        )

    async def send_response(
        self,
        message_to_reply: discord.Message = None,
        response_text: Optional[str] = None,
        embed: Optional[discord.Embed] = None,
        _file: Optional[discord.Embed] = None,
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
        # print(f"Response ({len(response_text)} chars):\n{response_text}")

        if not embed and not response_text:
            # TODO: Decide if sending a default response or raising an error is better.
            # For now, ominous default response because it's funny
            response_text = "..."
            # raise ValueError("No embed or response text included in the response.")

        # TODO: Respect character limits in embeds.
        if embed:
            await message_to_reply.reply(mention_author=True, embed=embed)
            return


        # output message with no embeds
        msg_index = 0
        last_message = None
        messages = SyntheaUtilities.split_text(response_text)
        for message in messages:
            if message_to_reply:
                # a message in a channel
                # only reply to the first message to prevent spamming
                if msg_index == 0:
                    last_message = await message_to_reply.reply(
                        message,
                        mention_author=True,
                        embed=embed,
                    )
                else:
                    last_message = await last_message.reply(
                        message, mention_author=False, embed=embed
                    )
            else:
                raise ValueError("No message found to reply to!")
            msg_index += 1

    async def _get_character_replied_to(self, message: discord.Message) -> str | None:
        """
        Determines if a user replied to a character.

        Args:
            message (discord.Message): The message the user sent.
        Returns:
            (str, optional): The id of the character who sent the message the user replied to.
                If there is no character or this message is not a reply, returns None instead
        """
        # if the bot was invoked without replying to a message, no character was replied to.
        if not message.reference:
            return None

        try:
            # bot uses embeds to speak as a character
            replied_message: discord.Message = await message.channel.fetch_message(
                message.reference.message_id
            )

            # if no embed, it wasn't speaking as a character
            if (
                not replied_message.embeds
                or not replied_message.author.id == self.user.id
            ):
                return None
            embed = replied_message.embeds[0]

            # ids are embedded in the footer, so retrieve that
            return embed.footer.text

        # if we can't retrieve the replied message (maybe deleted), just say no char
        except (discord.NotFound, discord.HTTPException, discord.Forbidden) as exc:
            print(exc)

        return None