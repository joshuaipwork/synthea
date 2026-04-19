import argparse
from dataclasses import dataclass
import re
from typing import IO, NoReturn
import yaml

from config import Config
from synthea.exceptions import InvalidImageDimensionsException

config = Config()

class CommandError(ValueError):
    """
    Indicates that the command parser encountered an error while parsing.
    """


class ParserExitedException(Exception):
    """
    Indicates that argparse would have exited if this were a command line command
    rather than a discord bot command
    """
    def __init__(self, msg: str):
        self.message = msg

class CommandParser(argparse.ArgumentParser):
    """
    A wrapper over argparse to make it better suited for parsing discord
    bot commands.
    """
    def parse_args(self, args=None, namespace=None):
        parsed_args: ParsedArgs = super().parse_args(args, namespace)
        if parsed_args.help:
            # We want to show different help based on whether other flags are present
            if parsed_args.use_image_model:  # If -im was provided (even without a value, it might be set to the default or True)
                raise ParserExitedException(f'''
                    ```usage: !syn -im [-h] [-dim "[width]x[height]"] prompt

                    This bot is an interface for using AI models. 
                    When used with the -im option, it generates an image instead of
                    sending a prompt to a language model

                    positional arguments:
                    prompt                The prompt to use with the image model.

                    options:
                    -h, --help            show this help message and exit
                    -d, -dim, --dimensions 
                                            Create an image with these dimensions.
                                            Use the form [width]x[height],
                                            for example 1024x1024.
                    ```''')
            else:
                raise ParserExitedException(f'''
                    ```usage: !syn [-h] [-c CHARACTER] [-im] [-sp] [-d] [-m MODEL] prompt

                    This bot is an interface for using AI models. 

                    positional arguments:
                    prompt                The prompt to give the bot.

                    options:
                    -h, --help            show this help message and exit
                    -c CHARACTER, -char CHARACTER, --character CHARACTER
                                            The character for the bot to assume in its response.
                    -im, --use-image-model
                                            Generates an image instead of contacting the LLM.
                                            For more information, use -im -h to get image-specific options.
                    -sp, -system-prompt, --use-as-system-prompt
                                            Save the prompt text as the system prompt
                                            for the remainder of the reply chain.
                    -dry, --dry-run
                                            If passed, return the final prompt,
                                            rather than the bot's response.
                    -m MODEL, -model MODEL, --model MODEL
                                            The language model to use.
                    ```''')
        return parsed_args

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
        raise ParserExitedException(f'```{self.format_help()}```')

    def print_help(self, file: IO[str] | None = None) -> None:
        """Overriden to prevent console spam"""

@dataclass
class ParsedArgs:
    character: str = None
    use_as_system_prompt: bool = False
    use_image_model: bool = False
    prompt: str = None
    dry_run: bool = False
    model: str = None
    dimensions: str = None
    image_width: str = None
    image_height: str = None
    help: bool = False

class ChatbotParser:
    def image_dimensions(self, value: str) -> str:
        """Validate and parse dimensions in the form '[width]x[length]'."""
        pattern = r'^\d+x\d+$'
        print(value)
        match = re.match(pattern, value.lower())
        if not match:
            raise argparse.ArgumentTypeError(
                f"Dimensions must be in the format '[width]x[length]', for instance 1000x1000. Got: '{value}'"
            )
        return value.lower()

    def __init__(self):
        self.parser = CommandParser(
            exit_on_error=False,
            prog="!syn",
            description="This bot is an interface for chatting with large language models.",
            add_help=False
        )
        self.parser.add_argument(
            '-h',
            '--help',
            action='store_true',
            help='Show help message'
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
            help="Generates an image instead of contacting the LLM.",
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
            "-dry",
            "--dry-run",
            default=None,
            action="store_true",
            dest="dry_run",
            help="If passed, return the final prompt, rather than the bot's response.",
        )
        self.parser.add_argument(
            "-m",
            "-model",
            "--model",
            action="store",
            default=None,
            dest="model",
            help="Which model to use.",
        )
        self.parser.add_argument(
            "-d",
            "-dim",
            "--dimensions",
            action="store",
            type=self.image_dimensions,
            default=None,
            dest="dimensions",
            help="Create an image with these dimensions. Use the form [width]x[height], for instance 1000x1000.",
        )
        self.parser.add_argument(
            "prompt", nargs=argparse.REMAINDER, help="The prompt to give the bot."
        )

    def parse(self, command: str) -> ParsedArgs:
        """
        Parses a command given by the user.
        """
        # remove the command start string if it was present.
        if command.lower().startswith(config.command_start_str.lower()):
            command = command[len(config.command_start_str):]

        # convert the parsed args into an object for better type matching
        args: ParsedArgs = self.parser.parse_args(command.split(), namespace=ParsedArgs())

        # post-process some args
        args.model = args.model.lower() if args.model else None
        args.prompt = " ".join(args.prompt)

        if args.dimensions:
            args.image_width, args.image_height = self._parse_dimensions(args.dimensions)
        return args

    def _parse_dimensions(self, value: str) -> tuple[int, int]:        
        values = value.split('x')
        if len(values) != 2:
            raise InvalidImageDimensionsException(f"Couldn't parse '{value}' into a width and height")

        width, height = int(values[0]), int(values[1])

        if width < 16 or height < 16:
            raise InvalidImageDimensionsException(f"Invalid dimensions {value} - minimum size is 16x16")
        if width > 16384 or height > 16384:
            raise InvalidImageDimensionsException(f"Invalid dimensions {value} - maximum size is 16384x16384")
        if width * height > config.image_maximum_pixels:
            raise InvalidImageDimensionsException(
                f"Dimensions {value} exceed the maximum pixel count of {config.image_maximum_pixels}")
    
        return width, height
