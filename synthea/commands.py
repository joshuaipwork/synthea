import argparse
import re
from dataclasses import dataclass
from typing import IO, NoReturn

from synthea.config import Config
from synthea.exceptions import InvalidImageDimensionsException

config = Config()


class CommandError(ValueError):
    """Indicates that the command parser encountered an error while parsing.
    """


class ParserExitedException(Exception):
    """Indicates that argparse would have exited if this were a command line command
    rather than a discord bot command.

    Carries the help message the bot should show to the user, split into
    pages for the paginated help view.
    """

    def __init__(self, pages: list[str], title: str = "📖 Help"):
        self.pages: list[str] = pages
        self.title: str = title
        super().__init__("Command parser exited with help output.")


class CommandParser(argparse.ArgumentParser):
    """A wrapper over argparse to make it better suited for parsing discord
    bot commands.
    """

    # keeps each help page comfortably below discord's 2000 character message limit
    MAX_HELP_PAGE_CHARS: int = 1800

    def parse_args(self, args=None, namespace=None):
        parsed_args: ParsedArgs = super().parse_args(args, namespace)
        if parsed_args.help:
            # We want to show different help based on whether other flags are present
            if parsed_args.use_image_model:  # If -im was provided (even without a value, it might be set to the default or True)
                raise ParserExitedException(
                    self.image_help_pages(),
                    title=f"🖼️ {config.bot_name} Image Generation Help",
                )
            raise ParserExitedException(
                self.help_pages(), title=f"📖 {config.bot_name} Help",
            )
        return parsed_args

    def error(self, message):
        """By default, argparse exits the program on error.
        This makes it so that it raises an exception instead.
        """
        raise CommandError(message)

    def exit(self, status: int = 0, message: str | None = None) -> NoReturn:
        """Some actions, like asking for help or encountering an error, will exit the program after running
        This makes it so that it raises an exception instead so the bot can return that to the user.
        """
        raise ParserExitedException(
            self.help_pages(), title=f"📖 {config.bot_name} Help",
        )

    def print_help(self, file: IO[str] | None = None) -> None:
        """Overriden to prevent console spam"""

    def help_pages(self) -> list[str]:
        """Formats the general help message as a list of pages for the
        paginated help view.
        """
        intro = self._intro_page(self._build_usage())
        return self._assemble_pages(intro, self._argument_entries())

    def image_help_pages(self) -> list[str]:
        """Formats the image-generation-specific help message as a list of
        pages for the paginated help view.
        """
        intro = self._intro_page(
            self._build_usage(
                include_dests={"help", "dimensions"}, fixed_flags=("-im",),
            ),
            description="When used with the -im option, this bot generates an image instead of sending a prompt to a language model.",
        )
        return self._assemble_pages(
            intro,
            self._argument_entries(
                include_dests={"help", "dimensions"},
                prompt_help="The prompt to use with the image model.",
            ),
        )

    def _intro_page(self, usage: str, description: str | None = None) -> str:
        """Builds the first help page, containing the usage line and a
        short description of the bot.
        """
        lines = ["**Usage**", f"`{usage}`"]
        text = description if description is not None else self.description
        if text:
            lines.append(text)
        return "\n\n".join(lines)

    def _build_usage(
        self,
        include_dests: set[str] | None = None,
        fixed_flags: tuple[str, ...] = (),
    ) -> str:
        """Builds the usage line, e.g. `!syn [-h] [-c CHARACTER] prompt`.

        Optional actions can be restricted to `include_dests`; positional
        arguments are always shown. `fixed_flags` are flags that are always
        present (like `-im` in the image generation usage line).
        """
        parts = [self.prog, *fixed_flags]
        for action in self._actions:
            if action.help is argparse.SUPPRESS:
                continue
            if action.option_strings:
                if include_dests is not None and action.dest not in include_dests:
                    continue
                parts.append(f"[{self._format_usage_flag(action)}]")
            elif action.nargs == argparse.REMAINDER:
                parts.append(action.dest)
            else:
                parts.append(f"<{action.dest}>")
        return " ".join(parts)

    @staticmethod
    def _format_usage_flag(action: argparse.Action) -> str:
        """Formats a single option for the usage line, e.g. `-c CHARACTER`.
        """
        option = action.option_strings[0]
        if action.nargs == 0:
            return option
        if action.choices:
            return f"{option} {{{'|'.join(action.choices)}}}"
        return f"{option} {action.dest.upper()}"

    def _argument_entries(
        self,
        include_dests: set[str] | None = None,
        prompt_help: str | None = None,
    ) -> list[str]:
        """Formats each documented argument as a markdown entry, with
        positional arguments first. `include_dests` (if given) restricts
        which optional actions are shown.
        """
        entries: list[str] = []
        positionals = [a for a in self._actions if not a.option_strings]
        options = [a for a in self._actions if a.option_strings]
        for action in positionals + options:
            if action.help is argparse.SUPPRESS:
                continue
            if action.option_strings:
                if include_dests is not None and action.dest not in include_dests:
                    continue
                entries.append(f"**`{self._format_flags(action)}`**\n> {action.help}")
            else:
                entries.append(f"**`{action.dest}`**\n> {prompt_help or action.help or ''}")
        return entries

    @staticmethod
    def _format_flags(action: argparse.Action) -> str:
        """Formats all of an option's flags, e.g. `-c CHARACTER, -char CHARACTER, --character CHARACTER`.
        """
        if action.nargs == 0:
            return ", ".join(action.option_strings)
        metavar = action.dest.upper()
        return ", ".join(f"{opt} {metavar}" for opt in action.option_strings)

    def _assemble_pages(self, intro: str, entries: list[str]) -> list[str]:
        """Splits the argument entries across pages so that no page exceeds
        discord's message length limit.
        """
        pages: list[str] = [intro]
        current: list[str] = []
        current_len = 0
        separator = "\n\n"
        for entry in entries:
            added_len = len(entry) + (len(separator) if current else 0)
            if current and current_len + added_len > self.MAX_HELP_PAGE_CHARS:
                pages.append(separator.join(current))
                current, current_len = [entry], len(entry)
            else:
                current.append(entry)
                current_len += added_len
        if current:
            pages.append(separator.join(current))
        return pages


@dataclass
class ParsedArgs:
    character: str = None
    # whether to use the given prompt as the system prompt instead of the next turn prompt
    use_as_system_prompt: bool = False
    use_image_model: bool = False
    prompt: str = None
    model: str = None
    reasoning_effort: str = None
    dimensions: str = None
    image_width: str = None
    image_height: str = None
    help: bool = False


class ChatbotParser:
    def image_dimensions(self, value: str) -> str:
        """Validate and parse dimensions in the form '[width]x[length]'."""
        pattern = r"^\d+x\d+$"
        print(value)
        match = re.match(pattern, value.lower())
        if not match:
            raise argparse.ArgumentTypeError(
                f"Dimensions must be in the format '[width]x[length]', for instance 1000x1000. Got: '{value}'",
            )
        return value.lower()

    def __init__(self):
        self.parser = CommandParser(
            exit_on_error=False,
            prog="!syn",
            description="This bot is an interface for chatting with large language models.",
            add_help=False,
        )
        self.parser.add_argument(
            "-h", "--help", action="store_true", help="Show this help message.",
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
            help="Generates an image instead of contacting the LLM. For more information, use -im -h to get image-specific options.",
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
            "-m",
            "-model",
            "--model",
            action="store",
            default=None,
            dest="model",
            help="Which model to use.",
        )
        self.parser.add_argument(
            "-re",
            "-reasoning-effort",
            "--reasoning-effort",
            action="store",
            choices=["low", "medium", "high"],
            default=None,
            dest="reasoning_effort",
            help="How much effort the model should spend reasoning. One of: low, medium, high.",
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
            "prompt", nargs=argparse.REMAINDER, help="The prompt to give the bot.",
        )

    def parse(self, command: str) -> ParsedArgs:
        """Parses a command given by the user.
        """
        # remove the command start string if it was present.
        if command.lower().startswith(config.command_start_str.lower()):
            command = command[len(config.command_start_str) :]

        # convert the parsed args into an object for better type matching
        args: ParsedArgs = self.parser.parse_args(
            command.split(), namespace=ParsedArgs(),
        )

        # post-process some args
        args.model = args.model.lower() if args.model else None
        args.prompt = " ".join(args.prompt)

        if args.dimensions:
            args.image_width, args.image_height = self._parse_dimensions(
                args.dimensions,
            )
        return args

    def _parse_dimensions(self, value: str) -> tuple[int, int]:
        values = value.split("x")
        if len(values) != 2:
            raise InvalidImageDimensionsException(
                f"Couldn't parse '{value}' into a width and height",
            )

        width, height = int(values[0]), int(values[1])

        if width < 16 or height < 16:
            raise InvalidImageDimensionsException(
                f"Invalid dimensions {value} - minimum size is 16x16",
            )
        if width > 16384 or height > 16384:
            raise InvalidImageDimensionsException(
                f"Invalid dimensions {value} - maximum size is 16384x16384",
            )
        if width * height > config.image_maximum_pixels:
            raise InvalidImageDimensionsException(
                f"Dimensions {value} exceed the maximum pixel count of {config.image_maximum_pixels}",
            )

        return width, height
