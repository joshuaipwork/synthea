"""Shared fixtures and test configuration.

Tests rely on ``config.yaml`` being readable from the current working
directory (several modules instantiate :class:`Config` at import time), so we
change into the repository root before any tests are collected.
"""

import os

import pytest

from synthea.commands import ChatbotParser, CommandParser
from synthea.config import Config

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session", autouse=True)
def _run_from_repo_root():
    previous_cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    yield
    os.chdir(previous_cwd)


@pytest.fixture()
def config() -> Config:
    return Config()


@pytest.fixture()
def parser() -> ChatbotParser:
    return ChatbotParser()


@pytest.fixture()
def command_parser() -> CommandParser:
    return CommandParser()
