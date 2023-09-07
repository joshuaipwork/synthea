"""
The discord client which contains the bulk of the logic for the chatbot.
"""
import traceback
from typing import Optional
import discord
from discord import app_commands
import yaml
from synthea.CharactersDatabase import CharactersDatabase

from synthea.SyntheaModel import SyntheaModel
from synthea.CommandParser import ChatbotParser
from synthea.ContextManager import ContextManager
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

    model: SyntheaModel = None
    char_db: CharactersDatabase = None
    synced = False

    async def setup_hook(self):
        """
        When the bot is started and logs in, load the model.
        """
        self.model = SyntheaModel()
        self.model.load_model()
        self.char_db = CharactersDatabase()

    async def on_ready(self):
        """
        Reports to the console that we logged in.
        """
        with open("config.yaml", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
            await self.change_presence(activity=discord.Game(name=config["activity"]))

        # sync slash commands only the first time that we are ready
        if not self.synced:
            await tree.sync()
            self.synced = True
            print("Synced command tree")

        print(f"Logged on as {self.user}!")

    async def on_reaction_add(self, reaction: discord.Reaction, _user):
        """
        Enables the bot to take a variety of actions when its posts are reacted to

        [🗑️] will tell the bot to delete its own post
        [▶️] will tell the bot to continue from this post
        """
        # TODO: make the bot add reactions as buttons on its own post
        # TODO: Figure out how to distinguish webhooks made by me from webhooks made by someone else
        # TODO: Don't delete messages if the webhook was made by someone else.
        if reaction.message.author.id == self.user.id:
            if reaction.emoji == "🗑️":
                await reaction.message.delete()
            if reaction.emoji == "🔁":
                # regenerate the response.
                await reaction.message.delete()

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
            await message.add_reaction("⏳")
            await self.respond_to_user(message)
            await message.add_reaction("✅")

        # if error, let the user know what went wrong
        # pylint: disable-next=broad-exception-caught
        except Exception as err:
            await message.add_reaction("❌")
            traceback.print_exc(limit=4)
            await message.reply(f"❌ {err}", mention_author=True)
        # after running, indicate the bot is done with their command.
        finally:
            await message.remove_reaction("⏳", self.user)

    async def respond_to_user(self, message: discord.Message):
        """
        Generates and send a response to a user message from the chatbot

        Args:
            message (str): The message to respond to
        """

        # parse the command the user sent
        command: str = message.clean_content
        parser = ChatbotParser()
        args = parser.parse(command)

        context_manager = ContextManager(self.model, self.user.id)

        char_id: str = args.character
        prompt: str = args.prompt

        # if the user responded to the bot playing a character, respond as that character
        replied_char_id = await self._get_character_replied_to(message)
        if replied_char_id:
            char_id = replied_char_id

        # send a response as a character
        if char_id:
            can_access = self.char_db.can_access_character(
                char_id=char_id,
                user_id=message.author.id,
                server_id=message.guild.id if message.guild else None,
            )
            if not can_access:
                raise CharacterNotOnServerError()

            char_data = self.char_db.load_character(char_id)
            prompt = await context_manager.compile_prompt_from_chat(
                message, system_prompt=char_data["system_prompt"]
            )

            print(f"Resp for {message.author} with char {char_id} prompt: \n{prompt}")
            response = self.model.generate(prompt=prompt)
            await self.send_response_as_character(response, char_data, message)
        # send a simple plaintext response without adopting a character
        else:
            # generate a response and send it to the user.
            prompt = await context_manager.compile_prompt_from_chat(message)
            print(f"Generating for {message.author} with prompt: \n{prompt}")
            response = self.model.generate(prompt=prompt)
            await self.send_response_as_base(response, message)

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
        print(f"Response ({len(response_text)} chars):\n{response_text}")

        msg_index = 0
        last_message = None
        if not embed and not response_text:
            # TODO: Decide if sending a default response or raising an error is better.
            # For now, ominous default response because it's funny
            response_text = "..."
            # raise ValueError("No embed or response text included in the response.")

        # TODO: Respect character limits in embeds.
        if embed:
            await message_to_reply.reply(mention_author=True, embed=embed)
            return

        while msg_index * CHAR_LIMIT < len(response_text):
            message_text = response_text[
                msg_index * CHAR_LIMIT : (msg_index + 1) * CHAR_LIMIT
            ]
            if message_to_reply:
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
                        message_text, mention_author=False, embed=embed
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


with open("config.yaml", "r", encoding="utf-8") as file:
    token = yaml.safe_load(file)["client_token"]

# set up the discord client. The client and takes actions on our behalf
intents = discord.Intents.all()
intents.message_content = True
intents.presences = True
intents.members = True
client = SyntheaClient(intents=intents)

# set up slash commands. It's pretty gross having this here along with the client,
# but it doesn't seem like there's a better way to do it
tree = app_commands.CommandTree(client)


@tree.command(
    name="create_character",
    description="Create a character from scratch.",
)
async def create_character_ui(interaction: discord.Interaction):
    """Opens the create_character UI for the user."""
    with open(
        "synthea/menu_dialogs/create_character.yaml", "r", encoding="utf-8"
    ) as dialog_file:
        dialogs = yaml.safe_load(dialog_file)
    await interaction.response.send_message(
        dialogs[CharCreationStep.ID.value]["text"],
        view=CharCreationView(),
        ephemeral=True,
    )


@tree.command(
    name="update_character",
    description="Update a character",
)
async def update_character_ui(interaction: discord.Interaction, char_id: str):
    """Opens the update_character UI for the user."""
    try:
        # will raise errors if the character can't be updated
        modal = UpdateCharModal(char_id, interaction)
        await interaction.response.send_modal(modal)
    except CharacterNotFoundError as err:
        await interaction.response.send_message(f"❌ {err}", ephemeral=True)
    except ForbiddenCharacterError as err:
        await interaction.response.send_message(f"❌ {err}", ephemeral=True)


@tree.command(
    name="delete_character",
    description="Delete an character you own",
    guild=discord.Object(id=1085939230284460102),
)
async def delete_character(interaction: discord.Interaction, char_id: str):
    """Deletes a character, throwing an error if not owned"""
    try:
        client.char_db.delete_character(char_id, interaction.user.id)
        await interaction.response.send_message(
            f"{char_id} was deleted.", ephemeral=True
        )
    except CharacterNotFoundError as err:
        await interaction.response.send_message(f"❌ {err}", ephemeral=True)
    except ForbiddenCharacterError as err:
        await interaction.response.send_message(f"❌ {err}", ephemeral=True)


@tree.command(
    name="add_character",
    description="Let anyone on this server use your character",
)
async def add_character(interaction: discord.Interaction, char_id: str):
    try:
        if not interaction.guild:
            await interaction.response.send_message(
                "❌ You are not speaking from a server!", ephemeral=True
            )
        # will raise errors if the character can't be updated
        client.char_db.add_character_to_server(
            char_id=char_id, user_id=interaction.user.id, server_id=interaction.guild.id
        )
        await interaction.response.send_message(
            f"{char_id} has been added to the server!"
        )
    except CharacterNotFoundError as err:
        await interaction.response.send_message(f"❌ {err}", ephemeral=True)
    except ForbiddenCharacterError as err:
        await interaction.response.send_message(f"❌ {err}", ephemeral=True)


@tree.command(
    name="remove_character",
    description="Stop allowing anyone from this server to use your character",
)
async def remove_character(interaction: discord.Interaction, char_id: str):
    try:
        if not interaction.guild:
            await interaction.response.send_message(
                "❌ You are not speaking from a server!", ephemeral=True
            )
        # will raise errors if the character can't be updated
        client.char_db.remove_character_from_server(
            char_id=char_id, user_id=interaction.user.id, server_id=interaction.guild.id
        )
        await interaction.response.send_message(
            f"{char_id} has been removed from the server!"
        )
    except CharacterNotFoundError as err:
        await interaction.response.send_message(f"❌ {err}", ephemeral=True)
    except ForbiddenCharacterError as err:
        await interaction.response.send_message(f"❌ {err}", ephemeral=True)


@tree.command(
    name="list_characters",
    description="Show a list of public characters on this server",
)
async def send_server_char_list(interaction: discord.Interaction):
    """Sends a list of public characters on the server to the interacter"""
    # public so others can see it
    # TODO: Paginate
    if not interaction.guild:
        await interaction.response.send_message(
            """❌ You are not on a server.
            Did you want to get a list of your owned characters?
            Use /list_owned_characters instead.""",
            ephemeral=True,
        )

    char_list = client.char_db.list_server_characters(interaction.guild.id)
    if not char_list:
        await interaction.response.send_message(
            "There are no public characters on this server.", ephemeral=True
        )

    await interaction.response.send_message(format_list(char_list), ephemeral=True)


@tree.command(
    name="list_owned_characters",
    description="Show a list of characters you own",
)
async def send_owned_char_list(interaction: discord.Interaction):
    """Sends a list of characters the user owns to the interacter"""
    char_list = client.char_db.list_user_characters(interaction.user.id)
    if not char_list:
        await interaction.response.send_message(
            "You don't own any characters.", ephemeral=True
        )

    await interaction.response.send_message(format_list(char_list), ephemeral=True)


def format_list(char_list: list[dict[str, str]]) -> str:
    """Generates a formatted text version of a list of characters and descriptions"""
    output = ""
    # I'd love to make a table, but discord doesn't support it. Markdown lists are the best I have
    for char in char_list:
        output += f'\n{char["id"]}'
        if "display_name" in char and char["display_name"]:
            output += f" ({char['display_name']})"
        if "description" in char and char["description"]:
            output += f'\n> {char["description"]}'
    return output


client.run(token)
