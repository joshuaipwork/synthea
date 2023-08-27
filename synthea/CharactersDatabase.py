import re
import sqlite3
from typing import Any, Optional
from .character_errors import *

conn = sqlite3.connect("mydata.db")

char_id_PATTERN = r"^\w+$"  # The regex pattern for valid strings
EDITABLE_COLUMNS = ["description", "display_name", "avatar_link", "system_prompt"]


class CharactersDatabase:
    """
    A wrapper for the characters database. Allows other modules to
    create, update, and retrieve characters stored in the characters database.

    Attributes:
        _conn (sqlite3.Connection): The SQLite database connection.
        _cursor (sqlite3.Cursor): The cursor for executing SQL commands.
    """

    def __init__(self, use_test=False):
        """
        Initializes a new or existing database named 'characters.db' and
        creates (or verifies the existence of) a table named 'characters'.

        Args:
            use_test (bool): If true, then the database will be opened
                from test_characters.db rather than characers.db.
        """
        if use_test:
            db_file = "test_characters.db"
        else:
            db_file = "characters.db"

        # Connect to a database (or create it if it doesn't exist)
        self._conn = sqlite3.connect(db_file)
        self._conn.row_factory = sqlite3.Row  # return rows as dicts
        self._conn.execute("PRAGMA foreign_keys = 1")  # enforce foreign keys
        self._cursor = self._conn.cursor()

        # Create the tables if they don't exist
        self._cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS characters (
                id TEXT NOT NULL PRIMARY KEY,
                owner INTEGER NOT NULL,
                description TEXT,
                display_name TEXT,
                system_prompt TEXT,
                avatar_link TEXT,
                model TEXT
            );
            """
        )
        self._cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS servers (
                server_id INTEGER PRIMARY KEY
            );
            """
        )
        self._cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS server_characters (
                server_id INTEGER,
                char_id TEXT,
                FOREIGN KEY (server_id) REFERENCES servers(server_id) ON DELETE CASCADE,
                FOREIGN KEY (char_id) REFERENCES characters(id) ON DELETE CASCADE,
                PRIMARY KEY (server_id, char_id)
            );
            """
        )

    def is_character_owner(self, char_id: str, user_id: int) -> bool:
        """
        Checks if a user is the owner of a character. Character owners can
        modify or delete characters. They can also add or remove characters
        from a server.

        Args:
            char_id (str): The character to check.
            user_id (int): The id of the user to check.
        Returns:
            (bool): True if the character is owned by the user, otherwise False.
        Raises:
            (CharacterNotFoundError): If no character by char_id is found in the DB.
        """
        char_id = char_id.lower()

        query = """
            SELECT COUNT(*)
            FROM characters c
            WHERE c.id = ?
        """
        self._cursor.execute(query, (char_id,))
        count = self._cursor.fetchone()[0]

        if count == 0:
            raise CharacterNotFoundError()

        query = """
            SELECT COUNT(*)
            FROM characters c
            WHERE c.id = ? AND c.owner = ?
        """
        self._cursor.execute(query, (char_id, user_id))
        count = self._cursor.fetchone()[0]

        return count > 0

    def can_access_character(
        self,
        char_id: str,
        user_id: Optional[int] = None,
        server_id: Optional[int] = None,
    ):
        """
        Checks if a user can access a character. Users can access characters in
        two different circumstances:
        1. The user is the owner of the character.
        2. The character has been added to the server where the user invokes the bot.

        Args:
            user_id (int, optional): The id of the user who is trying to access the character.
                Use this when checking if the user can access a character in DMs.
            server_id (int, optional): The id of the server in which the user is
                accessing this character.
                If this is in a DM, leave it as None.
        Raises:
            (ValueError): If neither a user_id nor a server_id was passed.
        """
        char_id = char_id.lower()
        if not user_id and not server_id:
            raise ValueError("No user_id or server_id to check character access rights")

        if server_id:
            # any user can access a character who has been added to a server
            query = """
                SELECT COUNT(*)
                FROM characters c
                JOIN server_characters sc ON c.id = sc.char_id
                WHERE c.id = ? AND (sc.server_id = ? OR c.owner = ?)
            """
            self._cursor.execute(query, (char_id, server_id, user_id))
            count = self._cursor.fetchone()[0]

            if count > 0:
                return True

        # Owners can always access their character, whether in DMs or on servers
        return self.is_character_owner(char_id, user_id)

    def load_character(
        self,
        char_id: str,
    ) -> dict | None:
        """
        Loads a character from the database,
        then returns the character as a dict.

        If the character doesn't exist, returns None.
        """
        char_id = char_id.lower()
        query = """
            SELECT *
            FROM characters c
            WHERE c.id = ?
        """
        self._cursor.execute(query, (char_id,))
        rows = self._cursor.fetchall()
        return dict(rows[0]) if rows else None

    def create_character(self, char_id: str, user_id: int):
        """
        Adds a character to the database.
        """
        char_id = char_id.lower()
        # check if the character exists
        if self.load_character(char_id):
            raise DuplicateCharacterError()
        # make sure the character ID conforms to our requirements.
        if not re.match(char_id_PATTERN, char_id):
            raise InvalidCharacterIDError()

        query = """
            INSERT INTO characters (id, owner)
            VALUES (?, ?)
            """

        # add a new character.
        self._cursor.execute(query, (char_id, user_id))
        self._conn.commit()

    def delete_character(self, char_id: str, user_id: int):
        """
        Deletes a character.

        Args:
            char_id (str): The character to delete.
            user_id (int): The discord user id of the user who wants to delete
                this character.
        Raises:
            (CharacterNotFoundError): If the character doesn't exist
            (ForbiddenCharacterError): If the user doesn't own this character
        """
        char_id = char_id.lower()
        char = self.load_character(char_id)
        if not char:
            raise CharacterNotFoundError()
        elif char["owner"] != user_id:
            raise ForbiddenCharacterError()

        query = """
                DELETE FROM characters
                WHERE id = ?
            """

        # add a new character.
        self._cursor.execute(query, (char_id,))
        self._conn.commit()

    def update_character(
        self, char_id: str, user_id: int, column_name: str, new_value: Any
    ):
        """
        Updates a field in a character. The character must be owned
        by the user.
        """
        char_id = char_id.lower()
        char = self.load_character(char_id)
        if not char:
            raise CharacterNotFoundError()
        elif char["owner"] != user_id:
            raise ForbiddenCharacterError()

        # Check if the column name is editable
        if column_name not in EDITABLE_COLUMNS:
            raise ValueError(f"Invalid column name {column_name}")

        # Prepare the SQL query
        query = f"""
            UPDATE characters
            SET {column_name} = ?
            WHERE id = ?
        """

        # Execute the query
        self._cursor.execute(query, (new_value, char_id))
        self._conn.commit()

    def remove_character_from_server(self, char_id: str, user_id: int, server_id: int):
        """
        Removes a character from a server. Such a character cannot be invoked
        by users on that server.
        """
        char_id = char_id.lower()
        # make sure the character exists and the user can edit it.
        char = self.load_character(char_id)
        if not char:
            raise CharacterNotFoundError()
        if not self.is_character_owner(char_id, user_id):
            raise ForbiddenCharacterError()

        query = """
            DELETE FROM server_characters
            WHERE char_id = ? AND server_id = ?
            """
        self._cursor.execute(query, (char_id, server_id))
        self._conn.commit()

    def add_character_to_server(self, char_id: str, user_id: int, server_id: int):
        """
        Adds a character to a server.
        The user must be the owner of the character.

        Args:
            char_id (str): The character to add.
            user_id (int): The user who wants to remove the character
                from a server.
        """
        char_id = char_id.lower()
        # make sure the character exists and the user can edit it.
        char = self.load_character(char_id)
        if not char:
            raise CharacterNotFoundError()
        if not self.is_character_owner(char_id, user_id):
            raise ForbiddenCharacterError()

        # add server to list of servers the bot is on
        query = """
            INSERT OR IGNORE INTO servers (server_id)
            VALUES (?)
            """
        self._cursor.execute(query, (server_id,))
        query = """
            INSERT OR IGNORE INTO server_characters (char_id, server_id)
            VALUES (?, ?)
            """
        self._cursor.execute(query, (char_id, server_id))
        self._conn.commit()

    def list_user_characters(self, user_id: int, offset=0):
        """
        Returns a list of the characters a user owns along with descriptions,
        if any.

        Args:
            user_id (int): The user to list owned characters from
        """
        # TODO: Limit and paginate this
        query = """
            SELECT id, description, display_name
            FROM characters
            WHERE owner = ?
            ORDER BY id ASC
            LIMIT 5
            OFFSET ?
            """
        self._cursor.execute(query, (user_id, offset))
        name_list = [dict(row) for row in self._cursor.fetchall()]
        return name_list

    def list_server_characters(self, server_id: int, offset=0) -> list[dict[str, str]]:
        """
        Returns a list of the characters on a server along with descriptions,
        if any.

        Args:
            char_id (str): The character to add.
            server_id (int): The server to list characters from
        """
        # TODO: Limit and paginate this
        query = """
            SELECT c.id, c.description, c.display_name
            FROM characters c
            JOIN server_characters sc ON sc.char_id = c.id
            WHERE server_id = ?
            ORDER BY c.id ASC
            LIMIT 5
            OFFSET ?
            """
        self._cursor.execute(query, (server_id, offset))
        name_list = [dict(row) for row in self._cursor.fetchall()]
        return name_list

    def __del__(self):
        """
        When the CharacterManager is deleted, clean up DB objects
        """
        if self._conn:
            self._conn.close()
