import discord
from discord import TextStyle, ui
from discord.interactions import Interaction
from synthea.CharactersDatabase import CharactersDatabase
from synthea.character_errors import (
    ForbiddenCharacterError,
)


class UpdateCharModal(ui.Modal):
    """
    A modal used to update all the attributes of a character at once.
    """

    def __init__(self, char_id: str, interaction: Interaction):
        super().__init__(title=f"Update Character {char_id}")

        # used to update the character at each step
        self.char_db = CharactersDatabase()
        self.char_id = char_id

        # can raise CharacterNotFoundError if the character doesn't exist
        is_owner = self.char_db.is_character_owner(char_id, interaction.user.id)
        if not is_owner:
            raise ForbiddenCharacterError()
        char_data = self.char_db.load_character(char_id)

        self.name = ui.TextInput(
            label="Name",
            default=char_data["display_name"],
            max_length=2000,
        )
        self.add_item(self.name)
        self.avatar = ui.TextInput(
            label="Avatar link",
            default=char_data["avatar_link"],
            max_length=200,
        )
        self.add_item(self.avatar)
        self.description = ui.TextInput(
            label="Description",
            default=char_data["description"],
            max_length=200,
        )
        self.add_item(self.description)
        self.system_prompt = ui.TextInput(
            label="AI Instructions",
            style=TextStyle.paragraph,
            default=char_data["system_prompt"],
            max_length=2000,
        )
        self.add_item(self.system_prompt)

    # pylint: disable-next=arguments-differ
    async def on_submit(self, interaction: discord.Interaction):
        """When submitted, update database with new records."""
        self.char_db.update_character(
            self.char_id, interaction.user.id, "avatar_link", self.avatar.value
        )
        self.char_db.update_character(
            self.char_id, interaction.user.id, "description", self.description.value
        )
        self.char_db.update_character(
            self.char_id, interaction.user.id, "display_name", self.name.value
        )
        self.char_db.update_character(
            self.char_id, interaction.user.id, "system_prompt", self.system_prompt.value
        )
        await interaction.response.send_message(
            "Your character has been updated!", ephemeral=True
        )
