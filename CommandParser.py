import argparse
import discord

class CommandError(ValueError):
    """
    Indicates that the command parser encountered an error while parsing.
    """
class ParserExitedException(Exception):
    """
    Indicates that argparse would have exited if this were a command line command
    rather than a discord bot command
    """
class ParserHelpException(Exception):
    """
    The message is expected to contain a help message for the parser's use.
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

    def exit(self):
        """
        Some actions, like asking for help or encountering an error, will exit the program after running
        This makes it so that it raises an exception instead so the bot can return that to the user.
        """
        raise ParserExitedException(self.format_help())

    def print_help(self):
        """
        Overriden to prevent console spam.
        """

class ChatbotParser:
    def __init__(self):
        self.parser = CommandParser(
            exit_on_error=False,
            prog="!syn",
            description="This bot is an interface for chatting with large language models."
        )
        self.parser.add_argument(
            '-c', '-char', '--character',
            action='store',
            default=None,
            help="The character for the bot to assume in its response."
        )
        self.parser.add_argument(
            '-t', '--thread',
            action='store_true',
            help="If passed, the bot will create a new thread to contain its response. Further responses by the bot in that thread \
                will read the history of the thread into the context."
        )
        self.parser.add_argument(
            '-n', '--thread_name',
            default="",
            help="If the bot creates a new thread to contain its response, it will use this name. Otherwise, a generic name is used."
        )
        self.parser.add_argument(
            'prompt',
            nargs="+",
            help="The prompt to give the bot."
        )

    def parse(self, command: str) -> argparse.Namespace:
        """

        """
        args = self.parser.parse_args(command.split())
        return args