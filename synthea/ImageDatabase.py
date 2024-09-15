import re
import sqlite3
from typing import Any, Optional
from .character_errors import *

conn = sqlite3.connect("images.db")

char_id_PATTERN = r"^\w+$"  # The regex pattern for valid strings
EDITABLE_COLUMNS = ["description", "display_name", "avatar_link", "system_prompt", "example_messages"]


class ImageDatabase:
    """
    A wrapper for the image database. 

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
            db_file = "test_images.db"
        else:
            db_file = "images.db"

        # Connect to a database (or create it if it doesn't exist)
        self._conn = sqlite3.connect(db_file)
        self._conn.row_factory = sqlite3.Row  # return rows as dicts
        self._conn.execute("PRAGMA foreign_keys = 1")  # enforce foreign keys
        self._cursor = self._conn.cursor()

        # Create the tables if they don't exist
        self._cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS images (
                url TEXT NOT NULL PRIMARY KEY,
                description TEXT
            );
            """
        )

    def get_image_description(
        self,
        url: str,
    ) -> str | None:
        """
        Loads an image description from the database
        by its url

        If the image doesn't exist, returns None.
        """
        query = """
            SELECT description
            FROM images i
            WHERE i.url = ? 
        """
        self._cursor.execute(query, (url,))
        rows = self._cursor.fetchall()
        return rows[0]['description'] if rows else None

    def add_image_description(self, url: str, description: int):
        """
        Adds an image and its description to the database.
        """

        query = """
            INSERT INTO images (url, description)
            VALUES (?, ?)
            """

        # add a new character.
        self._cursor.execute(query, (url, description))
        self._conn.commit()

    def __del__(self):
        """
        When the ImageDatabase is deleted, clean up DB objects
        """
        if self._conn:
            self._conn.close()
