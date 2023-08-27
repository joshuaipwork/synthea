from enum import Enum
from typing import Callable
import yaml
import discord
from discord import TextStyle, ui
from discord.enums import ButtonStyle
from discord.interactions import Interaction
from synthea.CharactersDatabase import CharactersDatabase
from synthea.character_errors import (
    DuplicateCharacterError,
    InvalidCharacterIDError,
    ForbiddenCharacterError,
)


class CharCreationStep(Enum):
    """
    The steps, in order, of the character creation menu
    Each step contains a string, which is used to identify the menu step dialog
    in menu_dialogs/create_character.yaml. It is also the name of the column that
    will be updated with the user's submission in the character database.
    """

    ID = "id"
    NAME = "display_name"
    SYSTEM_PROMPT = "system_prompt"
    AVATAR = "avatar_link"
    DESCRIPTION = "description"
    OUTRO = "outro"


class CharCreationView(ui.View):
    """
    A view used to navigate the character creation menu and access modals for
    for creating characters. This view is shown with the /create_character command.
    """

    def __init__(self):
        super().__init__(timeout=300)

        # the order of steps and the callback to run when each step's modal is submitted
        self.step_index = 0
        self.steps = [
            (CharCreationStep.ID, self.enter_id),
            (
                CharCreationStep.NAME,
                lambda interaction, new_val: self.enter_value(
                    interaction, new_val, CharCreationStep.NAME
                ),
            ),
            (
                CharCreationStep.SYSTEM_PROMPT,
                lambda interaction, new_val: self.enter_value(
                    interaction, new_val, CharCreationStep.SYSTEM_PROMPT
                ),
            ),
            (
                CharCreationStep.AVATAR,
                lambda interaction, new_val: self.enter_value(
                    interaction, new_val, CharCreationStep.AVATAR
                ),
            ),
            (
                CharCreationStep.DESCRIPTION,
                lambda interaction, new_val: self.enter_value(
                    interaction, new_val, CharCreationStep.DESCRIPTION
                ),
            ),
            (CharCreationStep.OUTRO, None),
        ]

        # used to update the character at each step
        self.char_db = CharactersDatabase()

        # caches the answers the user gives for each step
        self.data_dict = {}

        # retrieve data on modals and descriptions from yaml
        with open(
            "synthea/menu_dialogs/create_character.yaml", "r", encoding="utf-8"
        ) as file:
            self.dialogs = yaml.safe_load(file)

        # Create navigation buttons
        self.previous_step_button = ui.Button(label="<", style=ButtonStyle.blurple)
        self.enter_button = ui.Button(label="Enter", style=ButtonStyle.primary)
        self.next_step_button = ui.Button(label=">", style=ButtonStyle.blurple)

        self.previous_step_button.callback = self.go_to_previous_step
        self.enter_button.callback = self.open_update_modal
        self.next_step_button.callback = self.go_to_next_step

        self._update_buttons()

        self.add_item(self.previous_step_button)
        self.add_item(self.enter_button)
        self.add_item(self.next_step_button)

    def _update_buttons(self):
        """Updates which buttons can be accessed after each step"""
        current_step, callback = self.steps[self.step_index]
        # if the current step has no callback, disable the enter button
        if not callback:
            self.enter_button.disabled = True

        # can't move past the last step in the menu
        if self.step_index == len(self.steps) - 1:
            self.next_step_button.disabled = True
        # can't go to next step if this step isn't finished
        elif current_step not in self.data_dict:
            self.next_step_button.disabled = True
        else:
            self.next_step_button.disabled = False

        # can't move before the first step in the menu
        if self.step_index == 0:
            self.previous_step_button.disabled = True
        else:
            self.previous_step_button.disabled = False

    async def go_to_next_step(self, interaction: discord.Interaction):
        """Moves to the next step in the menu."""
        self.step_index += 1
        self._update_buttons()
        new_text: str = self.dialogs[self.steps[self.step_index][0].value]["text"]
        await interaction.response.edit_message(content=new_text, view=self)

    async def go_to_previous_step(self, interaction: discord.Interaction):
        """Moves to the previous step in the menu"""
        self.step_index -= 1
        self._update_buttons()
        new_text: str = self.dialogs[self.steps[self.step_index][0].value]["text"]
        await interaction.response.edit_message(content=new_text, view=self)

    async def open_update_modal(self, interaction: discord.Interaction):
        """Sends a modal to the user for them to enter in data"""
        current_step, callback = self.steps[self.step_index]
        await interaction.response.send_modal(
            self._EnterModal(
                current_step,
                dialogs=self.dialogs,
                callback=callback,
                title=self.dialogs[current_step.value]["modal_title"],
            )
        )

    async def enter_id(self, interaction: discord.Interaction, new_id: str):
        """When a user enters a valid id, it creates a character."""
        try:
            self.char_db.create_character(new_id, interaction.user.id)
            self.data_dict[CharCreationStep.ID] = new_id
            await self.go_to_next_step(interaction)
        except DuplicateCharacterError:
            new_text: str = self.dialogs["duplicate_id"]["text"]
            await interaction.response.edit_message(content=new_text, view=self)
        except InvalidCharacterIDError:
            new_text: str = self.dialogs["invalid_id"]["text"]
            await interaction.response.edit_message(content=new_text, view=self)

    async def enter_value(
        self,
        interaction: Interaction,
        new_value: str,
        field: CharCreationStep,
    ):
        """Sends a modal to the user for them to enter in data"""
        cid = self.data_dict[CharCreationStep.ID]
        self.char_db.update_character(cid, interaction.user.id, field.value, new_value)
        self.data_dict[CharCreationStep.NAME] = new_value
        await self.go_to_next_step(interaction)

    class _EnterModal(ui.Modal):
        """A modal which allows a user to enter data during character creation."""

        def __init__(
            self,
            step: CharCreationStep,
            dialogs: dict[str, dict[str, str]],
            callback: Callable[[discord.Interaction, str], None],
            title: str,
            timeout: float | None = None,
        ) -> None:
            """
            Creates the modal, taking in the standard variables of a modal, along with
            the callback to be perfomed on submit.

            Args:
                step (CharCreationStep): The step of character creation this modal represents.
                dialogs (dict): The dialog data from the menu.
                callback (Callable): The callback to run on submitting
            """
            super().__init__(title=title, timeout=timeout)
            text_data = dialogs[step.value]
            self.callback = callback

            # determine max length, and update textbox style accordingly
            max_length = text_data["modal_max_length"]
            if max_length > 200:
                style = TextStyle.paragraph
            else:
                style = TextStyle.short

            # create the input and add it to the modal
            self.value_input = ui.TextInput(
                label=text_data["modal_title"],
                placeholder=text_data["modal_placeholder"],
                max_length=max_length,
                style=style,
            )
            self.add_item(self.value_input)

        # pylint: disable-next=arguments-differ
        async def on_submit(self, interaction: discord.Interaction):
            await self.callback(interaction, self.value_input.value)

    async def on_timeout(self) -> None:
        """On timeout, disable all the buttons"""
        self.previous_step_button.disabled = True
        self.enter_button.disabled = True
        self.next_step_button.disabled = True
        return await super().on_timeout()


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
