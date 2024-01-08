from enum import Enum

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