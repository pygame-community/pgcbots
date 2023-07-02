"""This file is a part of the source code for PygameCommunityBot.
This project has been licensed under the MIT license.
Copyright (c) 2023-present pygame-community.

This file defines helper constants for its package module. 
"""

from typing import Any
import discord


LOG_LEVEL_NAMES: set[str] = {
    "CRITICAL",
    "FATAL",
    "ERROR",
    "WARN",
    "WARNING",
    "INFO",
    "DEBUG",
    "NOTSET",
}

DEFAULT_CONFIG: dict[str, Any] = {  # default bot configuration settings
    "intents": discord.Intents.default().value,
    "command_prefix": "!",
    "final_prefix": None,
    "mention_as_command_prefix": False,
    "extensions": [],
    "log_level": None,
}
