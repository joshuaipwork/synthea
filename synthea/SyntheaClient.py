"""
The discord client which contains the bulk of the logic for the chatbot.
"""
import math
import multiprocessing
import random
import traceback
from typing import Optional
import discord
from discord import app_commands
import openai
import yaml
import asyncio
from synthea import SyntheaUtilities

from synthea.CharactersDatabase import CharactersDatabase

from synthea.CommandParser import ChatbotParser, ParsedArgs
from synthea.Config import Config
from synthea.ContextManager import ContextManager
from synthea.dtos.GenerationRequest import GenerationRequest
from synthea.dtos.ResponseUpdate import ResponseUpdate
from synthea.character_errors import (
    CharacterNotFoundError,
    CharacterNotOnServerError,
)

COMMAND_START_STR: str = "!syn "
CHAR_LIMIT: int = 2000  # discord's character limit
FOOTER_PATTERN: str = r"^(.*) \| (\d+)$"
SYSTEM_TAG = "System"

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

    def __init__(self, intents):
        with open("config.yaml", "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

        super().__init__(intents=intents)
        self.openai: openai.AsyncOpenAI = openai.AsyncOpenAI(
            api_key=self.config["api_key"],
            base_url=self.config["api_base_url"]
        )
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

        print(f"Logged on as {self.user}!")

    async def on_reaction_add(self, reaction: discord.Reaction, user):
        """
        Enables the bot to take a variety of actions when its posts are reacted to

        [🗑️] will tell the bot to delete its own post
        [▶️] will tell the bot to stop generating
        """
        # TODO: make the bot add reactions as buttons on its own post
        # TODO: Figure out how to distinguish webhooks made by me from webhooks made by someone else
        # TODO: Don't delete messages if the webhook was made by someone else.
        if user != self.user and reaction.message.author.id == self.user.id:
            responded_message = reaction.message
            if reaction.emoji == "🗑️":
                await reaction.message.delete()
            # if reaction.emoji == "🛑":
            #     self.in_progress_responses.discard(response_id)
            #     await reaction.message.remove_reaction("📝", self.user)
            #     await reaction.message.remove_reaction("⏳", self.user)
            #     await reaction.message.add_reaction("⚠️")

            # regenerate the response
            if reaction.emoji == "🔁":
                user_message = await reaction.message.channel.fetch_message(reaction.message.reference.message_id)
                
                # TODO: regenerate the response.
                await reaction.message.delete()
                await user_message.remove_reaction("❌", self.user)
                await user_message.remove_reaction("⚠️", self.user)
                await user_message.remove_reaction("✅", self.user)
                await user_message.add_reaction("⏳")

                # create a new response
                await self.respond_to_user(user_message)
                await user_message.remove_reaction("⏳", self.user)

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
            # await message.add_reaction("🛑")
            await message.add_reaction("⏳")
            await self.respond_to_user(message)
            await message.add_reaction("✅")

        # if error, let the user know what went wrong
        # pylint: disable-next=broad-exception-caught
        except Exception as err:
            await message.add_reaction("❌")
            traceback.print_exc(limit=4)
            err_string = f"{err}"[:1024]
            await message.reply(f"❌ {err_string}", mention_author=True)

        await message.remove_reaction("⏳", self.user)

    async def respond_to_user(self, message_from_user: discord.Message):
        """
        Generates and send a response to a user message from the chatbot

        Args:
            message (str): The message to respond to
        """
        config: Config = Config()

        ### Deal with the case that the user made a command in this message
        command: str = message_from_user.clean_content
        parser: ChatbotParser = ChatbotParser()
        args: ParsedArgs = parser.parse(command)

        # if the user wants to use this as the system prompt going forward, just
        # give them a checkmark and wait for further prompts
        if args.use_as_system_prompt:
            await self.send_response_as_system("Conversation started...", message_from_user)
            return

        # read the history to find the current applicable command
        context_manager = ContextManager(self.user.id)
        chat_history, args = await context_manager.generate_chat_history_from_chat(
            message_from_user, system_prompt=config.system_prompt
        )

        char_id: str = None
        model: str = config.default_model
        if args:
            if model:
                model = args.model
            char_id = args.character

        # if the user responded to the bot playing a character, respond as that character
        replied_char_id = await self._get_character_replied_to(message_from_user)
        if replied_char_id:
            char_id = replied_char_id

        # send a response as a character
        if char_id and char_id != SYSTEM_TAG:
            can_access = self.char_db.can_access_character(
                char_id=char_id,
                user_id=message_from_user.author.id,
                server_id=message_from_user.guild.id if message_from_user.guild else None,
            )
            if not can_access:
                raise CharacterNotOnServerError()

            char_data = self.char_db.load_character(char_id)

            system_prompt: str = ""
            if "system_prompt" in char_data and char_data["system_prompt"]:
                system_prompt += char_data["system_prompt"]
            if "example_messages" in char_data and char_data["example_messages"]:
                system_prompt += "\n\n Here are some examples of how to speak:\n"
                system_prompt += char_data["example_messages"]

            chat_history, _ = await context_manager.generate_chat_history_from_chat(
                message_from_user, system_prompt=system_prompt
            )

            print(f"Resp for {message_from_user.author} with char {char_id}")
            response = await self.queue_for_generation(model, chat_history)
            await self.send_response_as_character(response, char_data, message_from_user)
        # send a simple plaintext response without adopting a character
        else:
        # generate a response and send it to the user.
            # chat_history = await context_manager.generate_chat_history_from_chat(
            #     message_from_user, system_prompt=self.config["system_prompt"]
            # )
            print(f"Generating for {message_from_user.author}")
            response = await self.queue_for_generation(model, chat_history)
            await self.send_response_as_base(response, message_from_user)

    async def queue_for_generation(self, model: str, chat_history: list[dict[str, str]]) -> str:
        """
        Sends a prompt to the server for generation. When the server is available,
        it will take up the prompt and generate a response.
        """
        config: Config = Config()
        print(chat_history)
        chat_completion = await self.openai.chat.completions.create(
            messages=chat_history,
            model=model,
            max_tokens=config.max_new_tokens,
            presence_penalty=config.presence_penalty,
            frequency_penalty=config.frequency_penalty,
            temperature=config.temperature,
            seed=-1,
            top_p=config.top_p
        )
        # TODO: Add error handling

        return chat_completion.choices[0].message.content

    async def send_response_as_base(self, response: str, message: discord.Message):
        """
        Sends a simple response using the base template of the model.
        """
        # create an embed to extend the character count
        embed: discord.Embed = discord.Embed(
            description=response,
        )

        await self.send_response(response_text=response, embed=embed, message_to_reply=message)

    async def send_response_as_system(self, response: str, message: discord.Message):
        """
        Sends a simple response annotated as system. System-annotated messages
        are ignored for chat history purposes.
        """
        # create an embed to extend the character count
        embed: discord.Embed = discord.Embed(
            description=response,
        )
        embed.set_footer(text=SYSTEM_TAG)
        await self.send_response(response_text=response, embed=embed, message_to_reply=message, add_buttons=False)

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
        add_buttons: bool = True,
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

        bot_message: discord.Message = await message_to_reply.reply(mention_author=True, embed=embed)

        # add controls
        if add_buttons:
            await bot_message.add_reaction("🗑️")
            await bot_message.add_reaction("🔁")

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