# pylint: disable=missing-function-docstring, redefined-outer-name, line-too-long
import os
import pytest
from synthea.CharactersDatabase import CharactersDatabase
from synthea.character_errors import (
    CharacterNotFoundError,
    ForbiddenCharacterError,
    DuplicateCharacterError,
)


@pytest.fixture(scope="module")
def manager():
    # delete the test database if it exists
    if os.path.exists("test_characters.db"):
        os.remove("test_characters.db")

    manager = CharactersDatabase(use_test=True)
    yield manager  # This will return the SQL object to the test functions

    # don't delete the test database so it can be inspected later


def test_load_invalid_character(manager: CharactersDatabase):
    assert manager.load_character("invalid_character") is None


@pytest.mark.dependency()
def test_create_character(manager: CharactersDatabase):
    manager.create_character("test_character", 100)
    assert manager.can_access_character("test_character", 100)
    char = manager.load_character("test_character")
    assert char is not None
    assert char["id"] == "test_character"
    assert char["owner"] == 100


@pytest.mark.dependency(depends=["test_create_character"])
def test_create_duplicate_character(manager: CharactersDatabase):
    with pytest.raises(DuplicateCharacterError):
        manager.create_character("duplicate_character", 100)
        manager.create_character("duplicate_character", 400)


@pytest.mark.dependency(depends=["test_create_character"])
def test_load_forbidden_character(manager: CharactersDatabase):
    manager.create_character("test_load_forbidden_character", 100)
    assert not manager.can_access_character(
        "test_load_forbidden_character", user_id=500
    )
    assert not manager.is_character_owner("test_load_forbidden_character", user_id=500)


@pytest.mark.dependency(depends=["test_create_character"])
def test_update_character(manager: CharactersDatabase):
    manager.create_character("update_test_char", 100)
    manager.update_character(
        char_id="update_test_char",
        user_id=100,
        column_name="description",
        new_value="test description",
    )
    char = manager.load_character("update_test_char")
    assert char["description"] == "test description"


@pytest.mark.dependency(depends=["test_create_character"])
def test_update_invalid_character(manager: CharactersDatabase):
    with pytest.raises(CharacterNotFoundError):
        manager.update_character(
            char_id="update_invalid_char_test_char",
            user_id=100,
            column_name="description",
            new_value="Some valid value",
        )


@pytest.mark.dependency(depends=["test_create_character"])
def test_update_invalid_field(manager: CharactersDatabase):
    with pytest.raises(ValueError):
        manager.create_character("update_invalid_field_test_char", 100)
        manager.update_character(
            char_id="update_invalid_field_test_char",
            user_id=100,
            column_name="invalid_column",
            new_value="Some valid value",
        )


@pytest.mark.dependency(depends=["test_create_character"])
def test_add_character_to_server(manager: CharactersDatabase):
    manager.add_character_to_server(
        char_id="test_character",
        user_id=100,
        server_id=200,
    )
    # once a character has been added to a server, other users on that server
    # should be able to access it.
    assert manager.can_access_character(
        char_id="test_character",
        user_id=500,
        server_id=200,
    )


@pytest.mark.dependency(depends=["test_create_character"])
def test_user_can_access_own_character(manager: CharactersDatabase):
    manager.create_character(
        char_id="owner_access_test",
        user_id=100,
    )
    # owners can access their character in DMs and on servers
    assert manager.can_access_character(
        char_id="owner_access_test",
        user_id=100,
        server_id=200,
    )
    assert manager.can_access_character(
        char_id="owner_access_test",
        user_id=100,
    )


@pytest.mark.dependency(depends=["test_add_character_to_server"])
def test_remove_character_from_server(manager: CharactersDatabase):
    manager.add_character_to_server(
        char_id="test_character",
        user_id=100,
        server_id=200,
    )
    manager.remove_character_from_server(
        char_id="test_character",
        user_id=100,
        server_id=200,
    )
    # once a character has been removed from a server, other users on that server
    # shouldn't be able to access it.
    assert not manager.can_access_character(
        char_id="test_character",
        server_id=200,
    )


@pytest.mark.dependency(depends=["test_create_character"])
def test_delete_character(manager: CharactersDatabase):
    manager.create_character("delete_test_character", 100)
    manager.delete_character("delete_test_character", user_id=100)
    assert manager.load_character("delete_test_character") is None


@pytest.mark.dependency(depends=["test_create_character"])
def test_delete_unowned_character(manager: CharactersDatabase):
    with pytest.raises(ForbiddenCharacterError):
        manager.create_character("delete_unowned_test_character", 100)
        manager.delete_character("delete_unowned_test_character", user_id=500)


def test_list_empty_server(manager: CharactersDatabase):
    assert len(manager.list_server_characters(-100)) == 0


def test_list_user_no_chars(manager: CharactersDatabase):
    assert len(manager.list_user_characters(-100)) == 0


@pytest.mark.dependency(
    depends=["test_create_character", "test_add_character_to_server"]
)
def test_list_server(manager: CharactersDatabase):
    # add one character, list it
    manager.create_character("list_server_test_char_1", 100)
    manager.add_character_to_server("list_server_test_char_1", 100, 400)
    manager.update_character(
        "list_server_test_char_1", 100, "description", "test_description"
    )
    char_list = manager.list_server_characters(400)

    assert len(char_list) == 1
    assert char_list[0]["id"] == "list_server_test_char_1"
    assert char_list[0]["description"] == "test_description"

    # add another character, list both
    manager.create_character("list_server_test_char_2", 100)
    manager.add_character_to_server("list_server_test_char_2", 100, 400)

    char_list = manager.list_server_characters(400)

    assert len(char_list) == 2
    assert char_list[0]["id"] == "list_server_test_char_1"
    assert char_list[1]["id"] == "list_server_test_char_2"
    assert char_list[0]["description"] == "test_description"
    assert char_list[1]["description"] is None


def test_own_invalid_character(manager: CharactersDatabase):
    with pytest.raises(CharacterNotFoundError):
        manager.is_character_owner("invalid_character", user_id=500)
