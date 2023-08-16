import argparse
import discord
import yaml

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
            '-C', '-com', '--command',
            action='store',
            default=None,
            help="Used to invoke utility functions. Note that commands will override the normal behavior \
                of the bot where it generates a response to your message. Follow this argument with the name of \
                the command to invoke. EG: --command list-characters"
        )
        self.parser.add_argument(
            '-c', '-char', '--character',
            action='store',
            default=None,
            help="The character for the bot to assume in its response."
        )
        self.parser.add_argument(
            'prompt',
            nargs="+",
            help="The prompt to give the bot."
        )

        # load config
        with open('config.yaml', 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

    def parse(self, command: str) -> argparse.Namespace:
        """
        Parses a command given by the user.
        """
        # remove the command start string if it was present.
        if command.startswith(self.config['command_start_str']):
            command = command[len(self.config['command_start_str']) + 1:]
        args = self.parser.parse_args(command.split())
        args.prompt = " ".join(args.prompt)
        return args