"""Tests for the text/dimension helpers in synthea.utilities."""

# pylint: disable=missing-function-docstring

from types import SimpleNamespace

import pytest

from synthea.exceptions import InvalidImageDimensionsException
from synthea.utilities import parse_dimensions, split_text, split_text_smartly


@pytest.fixture()
def dim_config():
    # image_maximum_pixels matches config.yaml
    return SimpleNamespace(config=SimpleNamespace(image_maximum_pixels=3_686_400))


def _all_words(text):
    return set(text.split())


class TestParseDimensions:
    def test_valid_dimensions(self, dim_config):
        assert parse_dimensions(dim_config, "1024x1024") == (1024, 1024)

    def test_minimum_size_allowed(self, dim_config):
        assert parse_dimensions(dim_config, "16x16") == (16, 16)

    def test_maximum_size_allowed(self, dim_config):
        # one dimension at the max, the other small enough to stay in the pixel limit
        assert parse_dimensions(dim_config, "16384x100") == (16384, 100)
        assert parse_dimensions(dim_config, "100x16384") == (100, 16384)

    @pytest.mark.parametrize("value", ["1024", "10x10x10"])
    def test_wrong_part_count_raises(self, dim_config, value):
        with pytest.raises(InvalidImageDimensionsException):
            parse_dimensions(dim_config, value)

    @pytest.mark.parametrize("value", ["1024x", "abcxdef"])
    def test_non_numeric_raises_value_error(self, dim_config, value):
        # empty/non-numeric components fail int() before the bounds checks
        with pytest.raises(ValueError):
            parse_dimensions(dim_config, value)

    @pytest.mark.parametrize("value", ["15x100", "100x15", "16385x100", "100x16385"])
    def test_out_of_bounds_raises(self, dim_config, value):
        with pytest.raises(InvalidImageDimensionsException):
            parse_dimensions(dim_config, value)

    def test_exceeding_pixel_limit_raises(self, dim_config):
        # 2000x2000 = 4,000,000 > 3,686,400 pixel limit
        with pytest.raises(InvalidImageDimensionsException):
            parse_dimensions(dim_config, "2000x2000")

    def test_within_pixel_limit_ok(self, dim_config):
        assert parse_dimensions(dim_config, "1920x1080") == (1920, 1080)


class TestSplitText:
    def test_short_text_is_single_piece(self):
        assert split_text("hello world") == ["hello world"]

    def test_empty_text_yields_empty_list(self):
        assert split_text("") == []

    def test_long_text_splits_at_max_length(self):
        text = "a" * 4000
        pieces = split_text(text, max_length=1800)
        assert len(pieces) == 3
        assert all(len(piece) <= 1800 for piece in pieces)
        assert "".join(pieces) == text


class TestSplitTextSmartly:
    def test_short_text_is_single_piece(self):
        assert split_text_smartly("Hello world.") == ["Hello world."]

    def test_empty_text_yields_empty_list(self):
        assert split_text_smartly("") == []

    def test_multiple_paragraphs_kept_in_one_piece(self):
        text = "First paragraph.\nSecond paragraph."
        pieces = split_text_smartly(text, max_length=2000)
        assert pieces == ["First paragraph.\nSecond paragraph."]

    def test_every_piece_is_within_max_length(self):
        text = ("word " * 2000) + "\n" + ("sentence one. sentence two. " * 300)
        pieces = split_text_smartly(text, max_length=2000)
        assert pieces
        assert all(len(piece) <= 2000 for piece in pieces)

    def test_no_content_is_lost(self):
        text = (
            "Introduction paragraph.\n"
            + ("packedno spaces here\n" * 50)
            + "\nSentence one. Sentence two. Sentence three."
        )
        pieces = split_text_smartly(text, max_length=2000)
        assert pieces
        source_words = _all_words(text)
        piece_words = set().union(*(_all_words(p) for p in pieces))
        assert source_words == piece_words

    def test_no_empty_pieces(self):
        text = "aaa bbb\n" + ("c" * 5000)
        pieces = split_text_smartly(text, max_length=2000)
        assert pieces
        assert all(p.strip() for p in pieces)

    def test_very_long_word_is_split(self):
        text = "z" * 5000
        pieces = split_text_smartly(text, max_length=2000)
        assert all(len(piece) <= 2000 for piece in pieces)
        assert "".join(pieces) == text
