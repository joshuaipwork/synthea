"""Tests for the discord command parsing logic in synthea.commands."""

# pylint: disable=missing-function-docstring, redefined-outer-name

import argparse

import pytest

from synthea.commands import (
    ChatbotParser,
    CommandError,
    CommandParser,
    ParserExitedException,
)
from synthea.exceptions import InvalidImageDimensionsException


class TestCommandParser:
    def test_error_raises_command_error(self, command_parser):
        with pytest.raises(CommandError):
            command_parser.error("some parse error")

    def test_exit_raises_parser_exited(self, command_parser):
        with pytest.raises(ParserExitedException):
            command_parser.exit(status=0, message="done")

    def test_help_flag_raises_parser_exited(self, command_parser):
        with pytest.raises(ParserExitedException):
            command_parser.parse_args(["-h"])

    def test_print_help_does_not_raise(self, command_parser, capsys):
        command_parser.print_help()
        captured = capsys.readouterr()
        assert captured.out == ""


class TestImageDimensions:
    def test_valid_dimensions(self, parser):
        assert parser.image_dimensions("1024x1024") == "1024x1024"

    def test_dimensions_are_lowercased(self, parser):
        assert parser.image_dimensions("1024X1024") == "1024x1024"

    @pytest.mark.parametrize(
        "value",
        ["1024", "1024y1024", "x", "abcxdef", "10x", "x10", "1024x1024x1024"],
    )
    def test_invalid_format_raises(self, parser, value):
        with pytest.raises(argparse.ArgumentTypeError):
            parser.image_dimensions(value)


class TestParseDimensions:
    def test_parses_width_and_height(self, parser):
        assert parser._parse_dimensions("1024x1024") == (1024, 1024)

    def test_minimum_size_allowed(self, parser):
        assert parser._parse_dimensions("16x16") == (16, 16)

    def test_maximum_size_allowed(self, parser):
        # one dimension at the max, the other small enough to stay in the pixel limit
        assert parser._parse_dimensions("16384x100") == (16384, 100)
        assert parser._parse_dimensions("100x16384") == (100, 16384)

    @pytest.mark.parametrize("value", ["15x100", "100x15", "16385x100", "100x16385"])
    def test_out_of_bounds_raises(self, parser, value):
        with pytest.raises(InvalidImageDimensionsException):
            parser._parse_dimensions(value)

    def test_exceeding_pixel_limit_raises(self, parser):
        # config.image_maximum_pixels is 3686400; 2000x2000 = 4,000,000 > limit
        with pytest.raises(InvalidImageDimensionsException):
            parser._parse_dimensions("2000x2000")

    def test_within_pixel_limit_ok(self, parser):
        assert parser._parse_dimensions("1920x1080") == (1920, 1080)


class TestChatbotParser:
    def test_basic_prompt(self, parser):
        args = parser.parse("!syn Hello world")
        assert args.prompt == "Hello world"

    def test_prompt_without_command_start(self, parser):
        args = parser.parse("Hello world")
        assert args.prompt == "Hello world"

    def test_character_flag(self, parser):
        args = parser.parse("!syn -c Bob tell me a joke")
        assert args.character == "Bob"
        assert args.prompt == "tell me a joke"

    def test_character_short_flag(self, parser):
        args = parser.parse("!syn -char Bob hi")
        assert args.character == "Bob"

    def test_model_flag_is_lowercased(self, parser):
        args = parser.parse("!syn -m DeepSeek hi")
        assert args.model == "deepseek"

    def test_model_absent_is_none(self, parser):
        args = parser.parse("!syn hi")
        assert args.model is None

    def test_image_model_flag(self, parser):
        args = parser.parse("!syn -im a picture of a cat")
        assert args.use_image_model is True

    def test_system_prompt_flag(self, parser):
        args = parser.parse("!syn -sp remember this context")
        assert args.use_as_system_prompt is True

    def test_dimensions_flag_sets_width_and_height(self, parser):
        args = parser.parse("!syn -d 1024x512 a dog")
        assert args.dimensions == "1024x512"
        assert args.image_width == 1024
        assert args.image_height == 512

    def test_invalid_dimensions_raise_argument_error(self, parser):
        # an ArgumentTypeError raised by the type callable surfaces as ArgumentError
        with pytest.raises(argparse.ArgumentError):
            parser.parse("!syn -d not_dimensions a dog")

    def test_help_raises_parser_exited(self, parser):
        with pytest.raises(ParserExitedException):
            parser.parse("!syn -h")

    def test_defaults_are_false(self, parser):
        args = parser.parse("!syn hello")
        assert args.use_as_system_prompt is False
        assert args.use_image_model is False
        assert args.help is False
