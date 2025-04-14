"""
The discord client which contains the bulk of the logic for the chatbot.
"""
from datetime import datetime
from io import BytesIO
import logging
import re
import traceback
from typing import Optional
import discord
from discord import app_commands
import yaml
import asyncio

from synthea.CharactersDatabase import CharactersDatabase

from synthea.CommandParser import ChatbotParser, ParsedArgs
from synthea.Config import Config
from synthea.ContextManager import ContextManager
from synthea.ImageModel import ImageModel
from synthea.LanguageModel import LanguageModel
from synthea.Model import Model
from synthea.dtos.GenerationResponse import GenerationResponse
from synthea.character_errors import (
    CharacterNotFoundError,
    CharacterNotOnServerError,
)

CHAR_LIMIT: int = 2000  # discord's character limit
DISCORD_EMBED_LIMIT: int = 4000  # discord's character limit
FOOTER_PATTERN: str = r"^(.*) \| (\d+)$"
CHAT_TAG_PATTERN: str = r'^[^:\n]{2,32}:\s(.*)$'
SYSTEM_TAG = "System"

CLIENT_LOGGER = logging.getLogger("synthea-client-logger")
CLIENT_LOGGER.handlers.clear()  # Clear any existing handlers
CLIENT_LOGGER.propagate = False  # Prevent propagation to parent loggers
console_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s", datefmt="%Y-%m-%d:%H:%M:%S")
console_handler.setFormatter(formatter)
CLIENT_LOGGER.addHandler(console_handler)

# This example requires the 'message_content' intent.
class SyntheaClient(discord.Client):
    """
    A discord client which recieves messages from users. When users send
    messages, the bot parses them and generates messages for them.
    """

    char_db: CharactersDatabase = None
    synced = False
    tree: app_commands.CommandTree = None
    client_logger = None

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
        super().__init__(intents=intents)

        self.language_model: LanguageModel = LanguageModel()
        self.image_model: ImageModel = ImageModel()
        self.config: Config = Config()
        self.char_db = CharactersDatabase()

    # async def setup_hook(self):
    #     """
    #     When the bot is started and logs in, load the model.
    #     """
   
    def measure_time(func):
        async def async_wrapper(*args, **kwargs):
            start_time = datetime.now()
            async with asyncio.timeout(None):
                result = await func(*args, **kwargs)
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            CLIENT_LOGGER.info(f"Async function {func.__name__} took {execution_time:.4f} seconds to finish.")
            return result

        def sync_wrapper(*args, **kwargs):
            start_time = datetime.now()
            result = func(*args, **kwargs)
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            CLIENT_LOGGER.info(f"Function {func.__name__} took {execution_time:.4f} seconds to execute.")
            return result

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

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
            CLIENT_LOGGER.info("Synced command tree")

        CLIENT_LOGGER.info(f"Logged on as {self.user}!")

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

        If a message is not by a user or fails to start with the command start string, then
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
        if message.content.lower().startswith(self.config.command_start_str.lower()):
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
                CLIENT_LOGGER.error(exc)
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
            async with asyncio.timeout(7200):
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

    @measure_time
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
        
        # if the user just wants to create an image, just create an image and
        # ignore the LLM
        if args.use_image_model:
            response: GenerationResponse = GenerationResponse()
            response.final_output = args.prompt
            generated_images = await self.image_model.get_images_from_prompt(args.prompt)
            for node_id in generated_images:
                for image_data in generated_images[node_id]:
                    response.images.append(image_data)
            await self.send_response_as_base(response, message_from_user)
            return

        # read the history to find the current applicable command
        context_manager = ContextManager(self.user.id)
        system_prompt: str = self._generate_system_prompt()
        chat_history, args = await context_manager.generate_chat_history_from_chat(
            message_from_user, system_prompt=system_prompt
        )

        char_id: str = None
        model: Model = self.language_model

        # if the user responded to the bot playing a character, respond as that character
        replied_char_id = await self._get_character_replied_to(message_from_user)
        if replied_char_id:
            char_id = replied_char_id

        # parse the chat again if it's a character
        # TODO: simplify
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

        response: GenerationResponse = await model.queue_for_chat_generation(chat_history)
        response.final_output = self._preprocess_final_output(response.final_output)
        
        # check for image generation tags, and generate an image if so
        if ("<image>" in response.final_output and "</image>" in response.final_output):
            image_prompt: str = re.search(r'<image>(.*?)</image>', response.final_output, re.DOTALL).group(1)
            generated_images = await self.image_model.get_images_from_prompt(image_prompt)
            response.final_output = re.sub(r'<image>.*?</image>', '', response.final_output, flags=re.DOTALL)
            for node_id in generated_images:
                for image_data in generated_images[node_id]:
                    response.images.append(image_data)
            response.reasoning += f"\nIMAGE PROMPT: {image_prompt}"

        if char_id and char_id != SYSTEM_TAG:
            CLIENT_LOGGER.info(f"Responded to {message_from_user.author} with char {char_id}")
            CLIENT_LOGGER.info(response)
            await self.send_response_as_character(response, char_data, message_from_user)
        else:
            CLIENT_LOGGER.info(f"Responded to {message_from_user.author}")
            CLIENT_LOGGER.info(response.final_output)
            await self.send_response_as_base(response, message_from_user)

    def _preprocess_final_output(self, final_output: str) -> str:
        """
        Does some simple preprocessing to improve the quality of responses
        """
        final_output = re.sub(r'Message from \".*?\":\n', '', final_output)
        if final_output.lower().startswith(f"Message from {self.config.bot_name}".lower()):
            final_output = final_output[len(f"Message from {self.config.bot_name}"):]
        if final_output.lower().startswith(f"Message from Syn".lower()):
            final_output = final_output[len(f"Message from Syn"):]

        # remove roleplay chat tags
        # This regex now uses a capture group to match the rest of the line
        match = re.match(CHAT_TAG_PATTERN, final_output, flags=re.DOTALL)
        if match:
            # If there's a match, return the captured group (rest of the line)
            final_output = match.group(1)
        if final_output.lower().startswith("Syn:".lower()):
            final_output = final_output[len("Syn:"):]
        if len(final_output) > DISCORD_EMBED_LIMIT:
            final_output = final_output[:DISCORD_EMBED_LIMIT]

        # remove stop words at end
        for stop_word in self.config.stop_words:
            if final_output.endswith(stop_word):
                final_output = final_output[:len(final_output)-len(stop_word)]
        return final_output

    async def send_response_as_base(self, response: GenerationResponse, message: discord.Message):
        """
        Sends a simple response using the base template of the model.
        """
        # create an embed to extend the character count
        embed: discord.Embed = discord.Embed(
            description=response.final_output
        )

        await self.send_response(embed=embed, message_to_reply=message, files=self.convert_generation_response_to_files(response))

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
        await self.send_response(embed=embed, message_to_reply=message, add_buttons=False)

    async def send_response_as_character(
        self, response: GenerationResponse, char_data: dict[str, str], message: discord.Message
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
            description=response.final_output,
        )

        # add a picture via url
        if "avatar_link" in char_data:
            embed.set_thumbnail(url=char_data["avatar_link"])

        # add the id to the footer so the bot knows what char sent this
        embed.set_footer(text=char_data["id"])

        # send the message with embed
        await self.send_response(
            embed=embed,
            message_to_reply=message,
            files=self.convert_generation_response_to_files(response)
        )

    async def send_response(
        self,
        message_to_reply: discord.Message = None,
        embed: Optional[discord.Embed] = None,
        add_buttons: bool = True,
        files: Optional[list[discord.File]] = None,
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

        if not embed:
            # TODO: Decide if sending a default response or raising an error is better.
            # For now, ominous default response because it's funny
            embed = discord.Embed(
                description="...",
            )
            # raise ValueError("No embed or response text included in the response.")

        bot_message: discord.Message = await message_to_reply.reply(mention_author=True, embed=embed, files=files)

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

    def convert_generation_response_to_files(self, response: GenerationResponse) -> list[discord.File]: 
        """
        Converts a generation response to a set of files to embed with the
        discord message.
        """
        files: list[discord.File] = []
        if (response.reasoning):
            CLIENT_LOGGER.info(f"Appending reasoning file to the message.")
            buffer = BytesIO(response.reasoning.encode())
            reasoning_file: discord.File = discord.File(buffer, filename=ContextManager.REASONING_TXT_FILE_NAME)
            files.append(reasoning_file)

        if (response.images):
            CLIENT_LOGGER.info(f"Appending {len(response.images)} images to the message.")
            for index, image in enumerate(response.images):
                buffer = BytesIO(image)
                files.append(discord.File(buffer, filename=f"image_{index}.png"))
        
        CLIENT_LOGGER.info(f"Appending {len(files)} files to the message.")
        return files
    
    def _generate_system_prompt(self):
        system_prompt: str = self.config.system_prompt 

        if self.config.can_use_reasoning:
            system_prompt = self.config.reasoning_system_prompt + "\n" + system_prompt
        if self.config.can_process_images:
            system_prompt += f"\n{self.config.image_generation_system_prompt}"

        return system_prompt