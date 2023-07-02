"""This file is a part of the source code for pgcbots.
This project has been licensed under the MIT license.
Copyright (c) 2023-present pygame-community.

This module implements all functionality to get a bot
application up and running.
"""

from .cli import make_bot_cli
from .bot import *
from .pgcbot import *
from .config_parsing import default_config_parser_mapping
