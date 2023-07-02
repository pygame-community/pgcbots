"""This file is a part of the source code for pgcbots.
This project has been licensed under the MIT license.
Copyright (c) 2023-present pygame-community.
"""

from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Any,
    Collection,
    Literal,
    TypedDict,
)
from sqlalchemy.ext.asyncio import AsyncEngine

if TYPE_CHECKING:
    from typing_extensions import Required, NotRequired  # type: ignore

ellipsis = type(Ellipsis)


class ConfigDatabase(TypedDict):
    name: str
    url: str
    connect_args: NotRequired[dict[str, Any]]


class Database(TypedDict):
    name: str
    engine: AsyncEngine
    url: str
    connect_args: NotRequired[dict[str, Any]]


class ExtensionData(TypedDict):
    name: str
    revision_number: int
    auto_migrate: bool
    db_prefix: str
    data: bytes | None


class ConfigAuthentication(TypedDict):
    token: str


class ConfigExtensionDict(TypedDict):
    name: str
    package: NotRequired[str]
    config: NotRequired[dict[str, Any]]


class Config(TypedDict, total=False):
    """Helper ``TypedDict`` for defining bot configuration data."""

    authentication: Required[ConfigAuthentication | dict[str, Any]]
    intents: int

    owner_id: int | None
    owner_ids: Collection[int]
    owner_role_ids: Collection[int]
    manager_role_ids: Collection[int]

    command_prefix: str | list[str] | tuple[str, ...]
    mention_as_command_prefix: bool

    extensions: list[ConfigExtensionDict] | tuple[ConfigExtensionDict, ...]

    databases: list[ConfigDatabase] | tuple[ConfigDatabase, ...]
    main_database_name: str
    auto_migrate: bool

    log_level: Literal[
        "CRITICAL",
        "FATAL",
        "ERROR",
        "WARN",
        "WARNING",
        "INFO",
        "DEBUG",
        "NOTSET",
    ]

    log_directory: str
    log_filename: str
    log_file_extension: str


class Revision(TypedDict):
    date: str
    description: str
    migrate: dict[str, list[str]]
    rollback: dict[str, list[str]]
    delete: NotRequired[dict[str, list[str]]]
