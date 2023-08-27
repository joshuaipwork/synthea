class DuplicateCharacterError(ValueError):
    """
    Raised when a user tries to create a character that already exists.
    """

    def __init__(self, message="A character with that name exists already."):
        super().__init__(message)


class InvalidCharacterIDError(ValueError):
    """
    Raised when a user tries to create a character with an invalid ID.
    """

    def __init__(
        self,
        message="Character IDs must be one word containing only letters, numbers, and underscores.",
    ):
        super().__init__(message)


class CharacterNotOnServerError(ValueError):
    """
    Raised when a user tries to invoke a character that hasn't been added to
    the server. It is intentionally ambiguous as to whether or not such a
    character exists to avoid exposing private characters.
    """

    def __init__(
        self,
        message="""Cannot find a character by that name.""",
    ):
        super().__init__(message)


class CharacterNotFoundError(ValueError):
    """
    Raised when a user tries to do something to a character that doesn't exist.
    """

    def __init__(
        self,
        message="There is no character by that name.",
    ):
        super().__init__(message)


class ForbiddenCharacterError(ValueError):
    """
    Raised when a user tries to edit a character
    which doesn't belong to them.
    """

    def __init__(self, message="That character belongs to another user."):
        super().__init__(message)
