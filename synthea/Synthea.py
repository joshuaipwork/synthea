"""
The starting point for the program
"""

import multiprocessing
import traceback
from typing import Optional
import discord
from discord import app_commands
import torch
import yaml
import asyncio
import janus

from synthea.CharactersDatabase import CharactersDatabase
from synthea.SyntheaClient import SyntheaClient
from synthea.SyntheaServer import SyntheaServer
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


def run_server_for_test(request_queue, response_queue):
    # Synchronous wrapper to run the async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    server: SyntheaServer = SyntheaServer(request_queue, response_queue)
    loop.run_until_complete(server.run())
    loop.close()

async def test_client(request_queue, response_queue):
    while True:
        response_update: ResponseUpdate = await asyncio.to_thread(response_queue.get)
        print(response_update.new_message)
        if response_update.message_is_completed: print("Message completed.")

def run_client_for_test(request_queue, response_queue):
    # Synchronous wrapper to run the async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.run_until_complete(test_client(request_queue, response_queue))
    loop.close()

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    request_queue: multiprocessing.Queue = multiprocessing.Queue()
    response_queue: multiprocessing.Queue = multiprocessing.Queue()

    with open("config.yaml", "r", encoding="utf-8") as file:
        token = yaml.safe_load(file)["client_token"]

    # set up the discord client. The client and takes actions on our behalf
    intents = discord.Intents.all()
    intents.message_content = True
    intents.presences = True
    intents.members = True
    client = SyntheaClient(intents=intents, request_queue=request_queue, response_queue=response_queue)

    # set up slash commands. It's pretty gross having this here along with the client,
    # but it doesn't seem like there's a better way to do it
    tree = app_commands.CommandTree(client)
    client.tree = tree

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


    with open("config.yaml", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    try:
        server_process = multiprocessing.Process(target=run_server_for_test, args=(request_queue, response_queue))
        server_process.start()


        client.run(config["client_token"])

        # responder_process = multiprocessing.Process(target=run_client_for_test, args=(request_queue, response_queue))
        # responder_process.start()

    finally:
        server_process.terminate()
        # client_process.terminate()
        torch.cuda.empty_cache()