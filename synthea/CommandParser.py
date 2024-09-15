import argparse
from typing import IO, NoReturn
import yaml

from synthea.Config import Config


class CommandError(ValueError):
    """
    Indicates that the command parser encountered an error while parsing.
    """


class ParserExitedException(Exception):
    """
    Indicates that argparse would have exited if this were a command line command
    rather than a discord bot command
    """


class CommandParser(argparse.ArgumentParser):
    """
    A wrapper over argparse to make it better suited for parsing discord
    bot commands.
    """

    def error(self, message):
        """
        By default, argparse exits the program on error.
        This makes it so that it raises an exception instead.
        """
        raise CommandError(message)

    def exit(self, status: int = 0, message: str | None = None) -> NoReturn:
        """
        Some actions, like asking for help or encountering an error, will exit the program after running
        This makes it so that it raises an exception instead so the bot can return that to the user.
        """
        raise ParserExitedException(self.format_help())

    def print_help(self, file: IO[str] | None = None) -> None:
        """Overriden to prevent console spam"""


class ParsedArgs:
    def __init__(self, character=None, use_as_system_prompt=False, use_image_model=False, prompt=None):
        self.character = character
        self.use_as_system_prompt: bool = use_as_system_prompt
        self.use_image_model: bool = use_image_model
        self.prompt: str = prompt

class ChatbotParser:
    def __init__(self):
        self.parser = CommandParser(
            exit_on_error=False,
            prog="!syn",
            description="This bot is an interface for chatting with large language models.",
            add_help=True,
        )
        self.parser.add_argument(
            "-c",
            "-char",
            "--character",
            action="store",
            default=None,
            help="The character for the bot to assume in its response.",
        )
        self.parser.add_argument(
            "-im",
            "--use-image-model",
            action="store_true",
            default=None,
            dest="use_image_model",
            help="Use the image model to generate the response.",
        )
        self.parser.add_argument(
            "-sp",
            "-system-prompt",
            "--use-as-system-prompt",
            action="store_true",
            default=None,
            dest="use_as_system_prompt",
            help="Save the prompt text as the system prompt for the remainder of the reply chain.",
        )
        self.parser.add_argument(
            "prompt", nargs=argparse.REMAINDER, help="The prompt to give the bot."
        )

        # load config
        self.config = Config()

    def parse(self, command: str) -> ParsedArgs:
        """
        Parses a command given by the user.
        """
        # remove the command start string if it was present.
        if command.lower().startswith(self.config.command_start_str.lower()):
            command = command[len(self.config.command_start_str):]

        # convert the parsed args into an object for better type matching
        args: ParsedArgs = self.parser.parse_args(command.split(), namespace=ParsedArgs())
        args.prompt = " ".join(args.prompt)
        return args
