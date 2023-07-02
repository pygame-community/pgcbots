"""This file is a part of the source code for pgcbots.
This project has been licensed under the MIT license.
Copyright (c) 2023-present pygame-community

This file defines some important utility functions for the library.
"""

import asyncio
from collections import OrderedDict, defaultdict, deque
from collections.abc import Mapping, MutableMapping
import collections
from dataclasses import dataclass
import datetime
import importlib.util
import io
import itertools
import logging
import logging.handlers
from math import log10
import os
import re
import sys
import types
from typing import (
    Any,
    Callable,
    Collection,
    Coroutine,
    Generic,
    Iterable,
    Literal,
    Mapping,
    Sequence,
    TypeVar,
    Union,
)

import discord
from discord.utils import _ColourFormatter
import emoji

from pgcbots.constants import UNSET
from . import regex_patterns

from typing_extensions import Self


def join_readable(joins: list[str]):
    """Join a list of strings, in a human readable way"""
    if not joins:
        return ""

    preset = ", ".join(joins[:-1])
    if preset:
        return f"{preset} and {joins[-1]}"

    return joins[-1]


def clamp(value, min_, max_):
    """Returns the value clamped between a maximum and a minumum"""
    value = value if value > min_ else min_
    return value if value < max_ else max_


def is_emoji_equal(
    partial_emoji: discord.PartialEmoji,
    emoji: str | discord.Emoji | discord.PartialEmoji,
):
    """Utility to compare a partial emoji with any other kind of emoji"""
    if isinstance(emoji, discord.PartialEmoji):
        return partial_emoji == emoji

    if isinstance(emoji, discord.Emoji):
        if partial_emoji.is_unicode_emoji():
            return False

        return emoji.id == partial_emoji.id

    return str(partial_emoji) == emoji


def format_discord_link(link: str, guild_id_or_me: int | Literal["@me"]):
    """Format a discord link to a channel or message"""
    link = link.lstrip("<").rstrip(">").rstrip("/")

    for prefix in (
        f"https://discord.com/channels/{guild_id_or_me}/",
        f"https://www.discord.com/channels/{guild_id_or_me}/",
    ):
        if link.startswith(prefix):
            link = link[len(prefix) :]

    return link


def progress_bar(
    pct: float, full_bar: str = "█", empty_bar: str = "░", divisions: int = 10
):
    """A simple horizontal progress bar generator."""
    pct = 0 if pct < 0 else 1 if pct > 1 else pct
    return full_bar * (int(divisions * pct)) + empty_bar * (
        divisions - int(divisions * pct)
    )


TIME_UNITS = (
    ("w", "weeks", 604800),
    ("d", "days", 86400),
    ("h", "hours", 3600),
    ("m", "minutes", 60),
    ("s", "seconds", 1),
    ("ms", "miliseconds", 1e-03),
    ("\u03bcs", "microseconds", 1e-06),
    ("ns", "nanoseconds", 1e-09),
)


def format_time_by_units(
    dt: datetime.timedelta | int | float,
    decimal_places: int = 4,
    value_unit_space: bool = True,
    full_unit_names: bool = False,
    multi_units: bool = False,
    whole_units: bool = True,
    fract_units: bool = True,
):
    """Format the given relative time in seconds into a string of whole and/or
    fractional time unit(s). The time units range from nanoseconds to weeks.


    Parameters
    ----------
    dt : datetime.timedelta | int | float
        The relative input time in seconds.
    decimal_places : int, optional
        The decimal places to be used in the formatted output time. Only applies when
        `multi_units` is `False`. Defaults to 4.
    value_unit_space : bool, optional
        Whether to add a whitespace character before the name of a time unit.
        Defaults to True.
    full_unit_names : bool, optional
        Use full unit names (like 'week' instead of 'w'). Defaults to False.
    multi_units : bool, optional
        Whether the formatted output string should use multiple units in descending order
        to divide up the input time. Defaults to False.
    whole_units : bool, optional
        Whether whole units above or equal to one should be used as units.
        Defaults to True.
    fractional_units : bool, optional
        Whether fractional units less than or equal to one should be used as units.
        Defaults to True.

    Returns
    -------
    str
        The formatted time string.

    Raises
    ------
    ValueError
        `dt` was not positive.
    """

    if isinstance(dt, datetime.timedelta):
        dt = dt.total_seconds()

    if dt < 0:
        raise ValueError("argument 'seconds' must be a positive number")

    name_idx = 1 if full_unit_names else 0
    space = " " if value_unit_space else ""

    if multi_units:
        result: list[str] = []
        start_idx = 0
        stop_idx = 7

        if whole_units and not fract_units:
            start_idx = 0
            stop_idx = 4

        elif not whole_units and fract_units:
            start_idx = 4
            stop_idx = 7

        elif not whole_units and not fract_units:
            raise ValueError(
                "the arguments 'whole_units' and 'fract_units' cannot both be False"
            )

        for i in range(start_idx, stop_idx + 1):
            unit_tuple = TIME_UNITS[i]
            name = unit_tuple[name_idx]
            unit_value = unit_tuple[2]

            value = dt // unit_value
            if value or (not result and unit_value == 1):
                dt -= value * unit_value
                if full_unit_names and value == 1:
                    name = name[:-1]  # type: ignore
                result.append(f"{value}{space}{name}")

        return join_readable(result)

    else:
        for unit_tuple in TIME_UNITS:
            name = unit_tuple[name_idx]
            unit_value = unit_tuple[2]
            if dt >= unit_value:
                return f"{dt / unit_value:.0{decimal_places}f}{space}{name}"


STORAGE_UNITS = (
    ("GB", "gigabytes", 1_000_000_000),
    ("MB", "megabytes", 1_000_000),
    ("KB", "kilobytes", 1_000),
    ("B", "bytes", 1),
)

BASE_2_STORAGE_UNITS = (
    ("GiB", "gibibytes", 1073741824),
    ("MiB", "mebibytes", 1048576),
    ("KiB", "kibibytes", 1024),
    ("B", "bytes", 1),
)


def format_byte(
    size: int,
    decimal_places: int | None = None,
    full_unit_names: bool = False,
    base_2_units: bool = False,
):
    """Format the given size in bytes into a string denoting
    the size with an equal or larger size unit. The units
    range from bytes to gigabytes.

    Parameters
    ----------
    size : int
        The size in bytes.
    decimal_places : int | None, optional
        The exact decimal places to display in the formatting output.
        If omitted, the Python `float` class's string version will be used to
        automatically choose the needed decimal places. Defaults to None.
    full_unit_names : bool, optional
        Use full unit names (like 'gigabyte' instead of 'GB'). Defaults to False.
    base_2_units : bool, optional
        Whether powers of 2 should be used as size units. Defaults to False.

    Returns
    -------
    str
        The formatted size string.
    """

    name_idx = 1 if full_unit_names else 0

    units = BASE_2_STORAGE_UNITS if base_2_units else STORAGE_UNITS

    for unit_tuple in units:
        name = unit_tuple[name_idx]
        unit_value = unit_tuple[2]
        if size >= unit_value:
            if decimal_places is None:
                return f"{size / unit_value} {name}"
            else:
                return f"{size / unit_value:.0{decimal_places}f} {name}"


def extract_markdown_mention_id(markdown_mention: str) -> int:
    """Extract the id '123456789696969' from a Discord role, user or channel markdown
    mention string with the structures '<@6969...>', '<@!6969...>', '<@&6969...>'
    or '<#6969...>'.
    Does not validate for the existence of those ids.

    Parameters
    ----------
    markdown_mention : str
        The mention string.

    Returns
    -------
    int
        The extracted integer id.

    Raises
    ------
    ValueError
        Invalid Discord markdown mention string.
    """

    match = re.match(regex_patterns.USER_ROLE_CHANNEL_MENTION, markdown_mention)
    if match is None:
        raise ValueError(
            "invalid Discord markdown mention string: Must be a guild role, channel "
            "or user mention"
        )

    return int(match.group(1))


def is_markdown_mention(string: str) -> bool:
    """Whether the given input string matches one of the structures of a valid Discord
    markdown mention string which are '<@6969...>', '<@!6969...>', '<@&6969...>'
    or '<#6969...>'.
    Does not validate for the actual existence of the mention targets.

    Parameters
    ----------
    string : str
        The string to check for.

    Returns
    -------
    bool
        ``True`` if condition is met, ``False`` otherwise.
    """
    return bool(re.match(regex_patterns.USER_ROLE_CHANNEL_MENTION, string))


def extract_markdown_custom_emoji_id(markdown_emoji: str) -> int:
    """Extract the id '123456789696969' from a Discord markdown custom emoji string with
    the structure '<:custom_emoji:123456789696969>'. Also includes animated emojis.
    Does not validate for the existence of the ids.

    Parameters
    ----------
    markdown_emoji  : str
        The markdown custom emoji string.

    Returns
    -------
    int
        The extracted integer id.

    Raises
    ------
    ValueError
        Invalid Discord markdown custom emoji string.
    """

    emoji_pattern = regex_patterns.CUSTOM_EMOJI
    match = re.match(emoji_pattern, markdown_emoji)
    if match is None:
        raise ValueError(
            "invalid Discord markdown custom emoji string: Must have the structure "
            "'<[a]:emoji_name:emoji_id>'"
        )

    return int(match.group(3))


def is_markdown_custom_emoji(string: str) -> bool:
    """Whether the given string matches the structure of a custom Discord markdown emoji
    string with the structure '<:custom_emoji:123456789696969>'. Also includes animated
    emojis.
    Does not validate for the existence of the specified emoji markdown.

    Parameters
    ----------
    string : str
        The string to check for.

    Returns
    -------
    bool
        ``True`` if condition is met, ``False`` otherwise.
    """
    return bool(re.match(regex_patterns.CUSTOM_EMOJI, string))


def is_emoji_shortcode(string: str) -> bool:
    """Whether the given string is a valid unicode emoji shortcode or alias shortcode.
    This function uses the `emoji` package for validation.

    Parameters
    ----------
    string : str
        The string to check for.

    Returns
    -------
    bool
        `True` if condition is met, `False` otherwise.
    """
    return (
        bool(re.match(regex_patterns.EMOJI_SHORTCODE, string))
        and emoji.emojize(string) != string
    )


def is_unicode_emoji(string: str) -> bool:
    """Whether the given string matches a valid unicode emoji.
    This function uses the `emoji` package for validation.

    Parameters
    ----------
    string : str
        The string to check for.

    Returns
    -------
    bool
        `True` if condition is met, `False` otherwise.
    """
    return emoji.is_emoji(string)


def shortcode_to_unicode_emoji(string: str) -> str:
    """Convert the given emoji shortcode to a valid unicode emoji,
    if possible. This function uses the `emoji` package for shortcode parsing.

    Parameters
    ----------
    string : str
        The emoji shortcode.

    Returns
    -------
    str
        The unicode emoji.
    """
    if is_emoji_shortcode(string):
        return emoji.emojize(string, language="alias")

    return string


def extract_markdown_timestamp(markdown_timestamp: str) -> int:
    """Extract the UNIX timestamp '123456789696969' from a Discord markdown
    timestamp string with the structure '<t:6969...>' or
    '<t:6969...[:t|T|d|D|f|F|R]>'.
    Does not check the extracted timestamps for validity.

    Parameters
    ----------
    markdown_timestamp : str
        The markdown timestamp string.

    Returns
    -------
    int
        The extracted UNIX timestamp.

    Raises
    ------
    ValueError
        invalid Discord markdown timestamp string.
    """

    ts_md_pattern = regex_patterns.UNIX_TIMESTAMP
    match = re.match(ts_md_pattern, markdown_timestamp)
    if match is None:
        raise ValueError("invalid Discord markdown timestamp string")

    return int(match.group(1))


def is_markdown_timestamp(string: str) -> int:
    """Whether the given string matches the structure of a Discord markdown timestamp
    string with the structure '<t:6969...>' or '<t:6969...[:t|T|d|D|f|F|R]>'.
    Does not check timestamps for validity.

    Parameters
    ----------
    string : str
        The string to check for.

    Returns
    -------
    bool
        `True` if condition is met, `False` otherwise.

    Raises
    ------
    ValueError
        Invalid Discord markdown timestamp string.
    """

    return bool(re.match(regex_patterns.UNIX_TIMESTAMP, string))


def code_block(string: str, max_characters: int = 2048, code_type: str = "") -> str:
    """Formats text into discord code blocks"""
    string = string.replace("```", "\u200b`\u200b`\u200b`\u200b")
    len_ticks = 7 + len(code_type)

    if len(string) > max_characters - len_ticks:
        return f"```{code_type}\n{string[:max_characters - len_ticks - 4]} ...```"
    else:
        return f"```{code_type}\n{string}```"


def have_permissions_in_channels(
    members_or_roles: discord.Member
    | discord.Role
    | Sequence[Union[discord.Member, discord.Role]],
    channels: discord.abc.GuildChannel
    | discord.Thread
    | Sequence[Union[discord.abc.GuildChannel, discord.Thread]],
    *permissions: str,
    member_role_bool_func: Callable[[Iterable[Any]], bool] = all,
    permission_bool_func: Callable[[Iterable[Any]], bool] = all,
    channel_bool_func: Callable[[Iterable[Any]], bool] = all,
) -> bool:
    """Checks if the given permission(s) apply/applies to the given member(s) or role(s) in the
    given channel(s) and returns a boolean value. This function allows you
    to evaluate this kind of query: Do/Does ... the given member(s) or role(s) have ...
    the given permission(s) in ... the given channel(s)?

    Parameters
    ----------
    members_or_roles : discord.Member | discord.Role | Sequence[Union[discord.Member, discord.Role]]
        The target Discord member(s) or role(s).
    channels : discord.abc.GuildChannel | discord.DMChannel | Sequence[Union[discord.abc.GuildChannel, discord.DMChannel]]
        The target channel(s) to check permissions on.
    *permissions : str
        The lowercase attribute name(s) to check for
        the `discord.Permissions` data class, which represent the different available
        permissions. These must match those supported by `discord.py`.
    member_role_bool_func : Callable[[Iterable[Any]], bool], optional
        A function that takes in the result of `channel_bool_func()` for every of
        the given role(s) or member(s) as an iterable and returns a boolean value.
        Defaults to `all`.
    permission_bool_func : Callable[[Iterable[Any]], bool], optional
        A function that takes in an iterable of the boolean values for every permission
        and returns a boolean value. Defaults to `all`.
    channel_bool_func : Callable[[Iterable[Any]], bool], optional
        A function that takes in the result of `permission_bool_func()` for every of
        the given channel(s) as an iterable and returns a boolean value.
        Defaults to `all`.

    Returns
    -------
    bool
        `True` if condition is met, `False` otherwise.
    """

    if isinstance(
        channels, (discord.abc.GuildChannel, discord.DMChannel, discord.Thread)
    ):
        channels = (channels,)

    if isinstance(members_or_roles, (discord.Member, discord.Role)):
        members_or_roles = (members_or_roles,)

    return member_role_bool_func(
        channel_bool_func(
            permission_bool_func(
                getattr(channel_perms, perm_name) for perm_name in permissions
            )
            for channel_perms in (ch.permissions_for(m_or_r) for ch in channels)
        )
        for m_or_r in members_or_roles
    )


def create_markdown_timestamp(
    dt: int | float | datetime.datetime, tformat: str = "f"
) -> str:
    """Get a discord timestamp formatted string that renders it correctly on the
    discord end. dt can be UNIX timestamp or datetime object while tformat
    can be one of:
    "f" (default) short datetime
    "F" long datetime
    "t" short time
    "T" long time
    "d" short date
    "D" long date
    "R" relative time (does not have much precision)
    """
    if isinstance(dt, datetime.datetime):
        dt = dt.replace(tzinfo=datetime.timezone.utc).timestamp()
    return f"<t:{int(dt)}:{tformat}>"


def recursive_mapping_compare(
    source_mapping: Mapping,
    target_mapping: Mapping,
    compare_func: Callable[[Any, Any], bool] | None = None,
    ignore_keys_missing_in_source: bool = False,
    ignore_keys_missing_in_target: bool = False,
    _final_bool: bool = True,
):
    """Compare the key and values of one mapping with those of another,
    But recursively do the same for mapping values that are mappings as well.
    based on the answers in
    https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    """

    if compare_func is None:
        compare_func = lambda d1, d2: d1 == d2

    if not ignore_keys_missing_in_source and (
        target_mapping.keys() >= source_mapping.keys()
    ):
        return False

    for k, v in source_mapping.items():
        if isinstance(v, Mapping) and isinstance(target_mapping.get(k, None), Mapping):
            _final_bool = recursive_mapping_compare(
                target_mapping[k],
                v,
                compare_func=compare_func,
                ignore_keys_missing_in_source=ignore_keys_missing_in_source,
                ignore_keys_missing_in_target=ignore_keys_missing_in_target,
                _final_bool=_final_bool,
            )
            if not _final_bool:
                return False
        else:
            if k not in target_mapping:
                if ignore_keys_missing_in_target:
                    continue
                _final_bool = False
            else:
                _final_bool = compare_func(v, target_mapping[k])
                if not _final_bool:
                    return False

    return _final_bool


def recursive_mapping_update(
    old_mapping: MutableMapping,
    update_mapping: Mapping,
    add_new_keys: bool = True,
    skip_value: str = UNSET,
):
    """Update one mapping with another, similar to `dict.update()`,
    But recursively update mapping values that are mappings as well.
    based on the answers in
    https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    """
    for k, v in update_mapping.items():
        if isinstance(v, dict):
            new_value = recursive_mapping_update(
                old_mapping.get(k, old_mapping.__class__()),
                v,
                add_new_keys=add_new_keys,
                skip_value=skip_value,
            )
            if new_value != skip_value:
                if k not in old_mapping:
                    if not add_new_keys:
                        continue
                old_mapping[k] = new_value

        elif v != skip_value:
            if k not in old_mapping:
                if not add_new_keys:
                    continue
            old_mapping[k] = v

    return old_mapping


def recursive_mapping_delete(
    old_mapping: MutableMapping,
    update_mapping: Mapping,
    skip_value: str = UNSET,
    inverse: bool = False,
):
    """Delete mapping entries present in another,
    But recursively do the same for mapping values that are mappings as well.
    based on the answers in
    https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    """
    if inverse:
        for k, v in tuple(old_mapping.items()):
            if isinstance(v, MutableMapping):
                lower_update_dict = {}
                if isinstance(update_mapping, MutableMapping):
                    lower_update_dict = update_mapping.get(
                        k, update_mapping.__class__()
                    )

                new_value = recursive_mapping_delete(
                    v, lower_update_dict, skip_value=skip_value, inverse=inverse
                )
                if (
                    new_value != skip_value
                    and isinstance(update_mapping, MutableMapping)
                    and k not in update_mapping
                ):
                    old_mapping[k] = new_value
                    if not new_value:
                        del old_mapping[k]
            elif (
                v != skip_value
                and isinstance(update_mapping, MutableMapping)
                and k not in update_mapping
            ):
                del old_mapping[k]
    else:
        for k, v in update_mapping.items():
            if isinstance(v, MutableMapping):
                new_value = recursive_mapping_delete(
                    old_mapping.get(k, old_mapping.__class__()),
                    v,
                    skip_value=skip_value,
                )
                if new_value != skip_value and k in old_mapping:
                    old_mapping[k] = new_value
                    if not new_value:
                        del old_mapping[k]

            elif v != skip_value and k in old_mapping:
                del old_mapping[k]
    return old_mapping


def recursive_mapping_cast(
    old_mapping: MutableMapping,
    cast_to: type[Mapping],
    cast_from: type[MutableMapping] | tuple[type[MutableMapping], ...] | None = None,
):
    """Recursively cast the `MutableMapping` given as input to the `Mapping` type specified in `cast_to`,
    whilst doing the same for its `MutableMapping` object values. This function is in-place, but returns
    a casted version of the mapping given as input. To prevent an original mapping from being modified,
    a deep-copied version of the mapping should be passed to this function.

    Parameters
    ----------
    old_mapping : MutableMapping
        The input mapping.
    cast_to : type[Mapping]
        The mapping type to cast to.
    cast_from : type[MutableMapping] | tuple[type[MutableMapping], ...] | None, optional
        The mapping type whose found instances should be casted. Defaults to None.

    Raises
    ------
    TypeError
        `old_mapping`, `cast_from` or `cast_to` were of an invalid type.

    Returns
    -------
    Mapping
        The casted version of a mapping given as input.
    """

    if not isinstance(old_mapping, MutableMapping):
        raise TypeError(
            f"'old_mapping' must be a MutableMapping, not {old_mapping.__class__.__name__}"
        )
    if cast_from is None:
        cast_from = old_mapping.__class__

    elif isinstance(cast_from, tuple) and not all(
        isinstance(c, type) for c in cast_from
    ):
        raise TypeError(
            f"'cast_from' must be a MutableMapping subclass or a tuple containing them, not {cast_from.__class__.__name__}"
        )
    elif not isinstance(cast_from, tuple):
        if not isinstance(cast_from, type) or not issubclass(cast_from, MutableMapping):
            raise TypeError(
                f"'cast_from' must be a MutableMapping subclass or a tuple containing them, not {cast_from.__name__}"
            )
    elif not issubclass(cast_to, Mapping):
        raise TypeError(f"'cast_to' must be a Mapping subclass, not {cast_to.__name__}")

    return _recursive_mapping_cast(old_mapping, cast_to, cast_from)


def _recursive_mapping_cast(
    old_mapping: MutableMapping,
    cast_to: type[Mapping],
    cast_from: type[MutableMapping] | tuple[type[MutableMapping], ...],
):
    for k in old_mapping:
        if isinstance(old_mapping[k], cast_from):
            old_mapping[k] = _recursive_mapping_cast(old_mapping[k], cast_to, cast_from)
    return cast_to(old_mapping)  # type: ignore


class FastChainMap(collections.ChainMap):
    """A drop-in replacement for ChainMap with
    optimized versions of some of its superclass's methods.
    """

    def __init__(self, *maps, ignore_defaultdicts: bool = False) -> None:
        """Initialize a FastChainMap by setting *maps* to the given mappings.
        If no mappings are provided, a single empty dictionary is used.
        *ignore_defaultdicts* can be used to speed up lookups by ignoring
        defaultdict objects during key lookup.
        """
        self.maps = list(maps) or [{}]  # always at least one map
        self.ignore_defaultdicts = ignore_defaultdicts

    def __getitem__(self, key):
        if self.ignore_defaultdicts:
            for mapping in self.maps:
                if key in mapping:
                    return mapping[key]

        else:
            for mapping in self.maps:
                if not isinstance(mapping, defaultdict):
                    if key in mapping:
                        return mapping[key]
                    continue

                try:
                    return mapping[key]  # can't use 'key in mapping' with defaultdict
                except KeyError:
                    pass

        return self.__missing__(key)  # support subclasses that define __missing__

    def get(self, key, default=None):
        return self[key] if any(key in m for m in self.maps) else default

    def __iter__(self):
        return iter({k: None for k in itertools.chain(*reversed(self.maps))})


def class_getattr_unique(
    cls: type,
    name: str,
    filter_func: Callable[[Any], bool] = lambda obj: True,
    check_dicts_only: bool = False,
    _id_set=None,
) -> list[Any]:
    values = []
    value_obj = UNSET

    if _id_set is None:
        _id_set = set()

    if check_dicts_only:
        if name in cls.__dict__:
            value_obj = cls.__dict__[name]
    else:
        if hasattr(cls, name):
            value_obj = getattr(cls, name)

    if (
        value_obj is not UNSET
        and id(value_obj) not in _id_set
        and filter_func(value_obj)
    ):
        values.append(value_obj)
        _id_set.add(id(value_obj))

    for base_cls in cls.__mro__[1:]:
        values.extend(
            class_getattr_unique(
                base_cls,
                name,
                filter_func=filter_func,
                check_dicts_only=check_dicts_only,
                _id_set=_id_set,
            )
        )
    return values


def class_getattr_unique_mapping(
    cls: type,
    name: str,
    filter_func: Callable[[Any], bool] | None = None,
    check_dicts_only: bool = False,
    _id_set=None,
) -> dict[type, Any]:
    values = {}
    value_obj = UNSET

    if _id_set is None:
        _id_set = set()

    if check_dicts_only:
        if name in cls.__dict__:
            value_obj = cls.__dict__[name]
    else:
        if hasattr(cls, name):
            value_obj = getattr(cls, name)

    if (
        value_obj is not UNSET
        and id(value_obj) not in _id_set
        and (filter_func(value_obj) if filter_func else True)
    ):
        values[cls] = value_obj
        _id_set.add(id(value_obj))

    for base_cls in cls.__mro__[1:]:
        values.update(
            class_getattr_unique_mapping(
                base_cls,
                name,
                filter_func=filter_func,
                check_dicts_only=check_dicts_only,
                _id_set=_id_set,
            )
        )

    return values


def class_getattr(
    cls: type,
    name: str,
    default: Any = UNSET,
    /,
    filter_func: Callable[[Any], bool] = lambda obj: True,
    check_dicts_only: bool = False,
    _is_top_lvl=True,
):
    value_obj = UNSET
    if check_dicts_only:
        if name in cls.__dict__:
            value_obj = cls.__dict__[name]
    else:
        if hasattr(cls, name):
            value_obj = getattr(cls, name)

    if value_obj is not UNSET and filter_func(value_obj):
        return value_obj

    for base_cls in cls.__mro__[1:]:
        value_obj = class_getattr(
            base_cls, name, check_dicts_only=check_dicts_only, _is_top_lvl=False
        )
        if value_obj is not UNSET:
            return value_obj

    if default is UNSET:
        if not _is_top_lvl:
            return UNSET
        raise AttributeError(
            f"could not find the attribute '{name}' in the __mro__ hierarchy of class "
            f"'{cls.__name__}'"
        ) from None

    return default


_T = TypeVar("_T")
_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


class BoundedOrderedDict(OrderedDict[_KT, _VT]):
    """A subclass of `OrderedDict` that pops its oldest items
    if a maximum length was set for it, similar to `deque`.
    """

    def __init__(self, *args: Any, maxlen: int | None = None, **kwds: Any) -> None:
        super().__init__(*args, **kwds)
        if maxlen is not None:
            if not isinstance(maxlen, int):
                raise TypeError(
                    "argument 'maxlen' must be a nonzero integer of type 'int', not "
                    f"'{maxlen.__class__.__name__}'"
                )

            elif maxlen < 1:
                raise TypeError(
                    f"argument 'maxlen' must be a nonzero integer of type 'int'"
                )

            self.__maxlen = maxlen
        else:
            self.__maxlen = None

        if self:
            self._check_maxlen_reached()

    @property
    def maxlen(self) -> int | None:
        return self.__maxlen

    def __setitem__(self, __key: _KT, __value: _VT) -> None:
        super().__setitem__(__key, __value)
        self._check_maxlen_reached()
        return None

    def setdefault(self, key: _KT, default: _T | None = None) -> _T | _VT | None:
        """Insert key with a value of default if key is not in the dictionary.

        Return the value for key if key is in the dictionary, else default.
        """
        value = super().setdefault(key, default=default)  # type: ignore
        if value is not default:
            self._check_maxlen_reached()

        return value

    def _check_maxlen_reached(self) -> None:
        length = len(self)
        maxlen = self.__maxlen
        if maxlen is not None and length > maxlen:
            for _ in range(length - maxlen):
                self.popitem(last=False)


class DequeProxy(Sequence[_T], Generic[_T]):
    __slots__ = ("__deque",)

    def __init__(self, deque_obj: deque[_T]) -> None:
        self.__deque = deque_obj

    @property
    def maxlen(self) -> int | None:
        return self.__deque.maxlen

    def __copy__(self) -> Self:
        return self.__class__(self.__deque)

    copy = __copy__

    def count(self, __x: Any) -> int:
        return self.__deque.count(__x)

    def index(self, __x: Any, **kwargs) -> int:
        return self.__deque.index(__x, **kwargs)

    def __len__(self) -> int:
        return self.__deque.__len__()

    def __lt__(self, __other: deque, /):
        return self.__deque.__lt__(__other)

    def __le__(self, __other: deque, /):
        return self.__deque.__le__(__other)

    def __gt__(self, __other: deque, /):
        return self.__deque.__lt__(__other)

    def __ge__(self, __other: deque, /):
        return self.__deque.__ge__(__other)

    def __eq__(self, __o: object, /):
        return self.__deque.__eq__(__o)

    def __ne__(self, __o: object, /):
        return self.__deque.__ne__(__o)

    def __bool__(self) -> bool:
        return bool(self.__deque)

    def __contains__(self, __o: object) -> bool:
        return self.__deque.__contains__(__o)

    def __getitem__(self, __index: int, /):
        return self.__deque.__getitem__(__index)

    def __hash__(self) -> int:
        return self.__deque.__hash__()

    def __iter__(self):
        return self.__deque.__iter__()

    def __reversed__(self):
        return self.__deque.__reversed__()

    def __add__(self, __other: deque, /):
        return self.__deque.__add__(__other)

    def __mul__(self, __other: int, /):
        return self.__deque.__mul__(__other)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__deque!r})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.__deque!s})"


_global_task_set: set[
    asyncio.Task
] = set()  # prevents asyncio.Task objects from disappearing due
# to reference loss, not to be modified manually


def hold_task(task: asyncio.Task[Any]) -> None:
    """Store an `asyncio.Task` object in a container to place a protective reference
    on it in order to prevent its loss at runtime. The given task will be given a
    callback that automatically removes it from the container when it is done.

    Parameters
    ----------
    task : asyncio.Task[Any]
        The task.
    """
    if task in _global_task_set:
        return

    _global_task_set.add(task)
    task.add_done_callback(_global_task_set_remove_callback)


def _global_task_set_remove_callback(task: asyncio.Task):
    if task in _global_task_set:
        _global_task_set.remove(task)

    task.remove_done_callback(_global_task_set_remove_callback)


def import_module_from_path(module_name: str, file_path: str) -> types.ModuleType:
    abs_file_path = os.path.abspath(file_path)
    spec = importlib.util.spec_from_file_location(module_name, abs_file_path)
    if spec is None:
        raise ImportError(
            f"failed to generate module spec for module named '{module_name}' at '{abs_file_path}'"
        )
    module = importlib.util.module_from_spec(spec)  # type: ignore
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore
    except FileNotFoundError as fnf:
        del sys.modules[module_name]
        raise ImportError(
            f"failed to find code for module named '{module_name}' at '{abs_file_path}'"
        ) from fnf
    except Exception:
        del sys.modules[module_name]
        raise

    return module


def unimport_module(module: types.ModuleType) -> None:
    """Unimport a module, by deleting it from `sys.modules`.
    Note that this will not remove any existing outer references
    to the module.

    Parameters
    ----------
    module : ModuleType
        The module object.
    """
    if module.__name__ in sys.modules:
        del sys.modules[module.__name__]


async def message_delete_reaction_listener(
    client: discord.Client | discord.AutoShardedClient,
    message: discord.Message,
    invoker: discord.Member | discord.User,
    emoji: discord.Emoji | discord.PartialEmoji | str,
    role_whitelist: Collection[discord.Role | int] | None = None,
    timeout: float | None = None,
    on_delete: Callable[[discord.Message], Coroutine[Any, Any, Any]]
    | Callable[[discord.Message], Any]
    | None = None,
):
    """Allows for a message to be deleted using a specific reaction.
    If any HTTP-related exceptions are raised by `discord.py` within this function,
    it will fail silently.

    Parameters
    ----------
    message : :class:`discord.Message`
        The message to use.
    invoker : :class:`discord.Member` | :class:`discord.User`
        The member/user who can delete a message.
    emoji : :class:`discord.Emoji` | :class:`discord.PartialEmoji` | :class:`str`):
        The emoji to listen for.
    role_whitelist : Collection[:class:`discord.Role` | :class:`int`] | None, optional
        A collection of roles or role IDs whose users' reactions can also be picked up by this function.
    timeout : :class:`float` | None, optional
        A timeout for waiting, before automatically removing any added reactions and returning silently.
    on_delete : Callable[[:class:`discord.Message`], Coroutine[Any, Any, Any]] | Callable[[:class:`discord.Message`], Any] | None, optional
        A (coroutine) function to call when a message is successfully deleted via the reaction. Defaults to `None`.

    Raises
    ------
    TypeError
        Invalid argument types.
    """

    role_whitelist_set = set(
        r.id if isinstance(r, discord.Role) else r for r in (role_whitelist or ())
    )

    if not isinstance(emoji, (discord.Emoji, discord.PartialEmoji, str)):
        raise TypeError("invalid emoji given as input.")

    try:
        try:
            await message.add_reaction(emoji)
        except discord.HTTPException:
            return

        check = None
        if isinstance(invoker, discord.Member):
            check = (
                lambda event: event.message_id == message.id
                and (
                    event.user_id == invoker.id
                    or any(
                        role.id in role_whitelist_set
                        for role in getattr(event.member, "roles", ())[1:]
                    )
                )
                and is_emoji_equal(event.emoji, emoji)
            )
        elif isinstance(invoker, discord.User):
            if isinstance(message.channel, discord.DMChannel):
                check = lambda event: event.message_id == message.id and is_emoji_equal(
                    event.emoji, emoji
                )
            else:
                check = (
                    lambda event: event.message_id == message.id
                    and event.user_id == invoker.id
                    and is_emoji_equal(event.emoji, emoji)
                )
        else:
            raise TypeError(
                "argument 'invoker' expected discord.Member/.User, "
                f"not {invoker.__class__.__name__}"
            )

        event: discord.RawReactionActionEvent = await client.wait_for(
            "raw_reaction_add", check=check, timeout=timeout
        )

        try:
            await message.delete()
        except discord.HTTPException:
            pass
        else:
            if on_delete is not None:
                await discord.utils.maybe_coroutine(on_delete, message)

    except asyncio.TimeoutError:
        try:
            await message.clear_reaction(emoji)
        except discord.HTTPException:
            pass


def _raise_exc(exc: BaseException):
    raise exc


class ParserMapping(
    dict[
        str,
        Callable[[str, Any, MutableMapping[str, Any]], Any]
        | "ParserMapping"
        | "ParserMappingValue",
    ]
):
    """A `dict` subclass that parses and/or validates mapping objects based on the
    structure and callback values of the input mapping given to it. The parsing and/or
    validating occurs in the order of definition of the key-value pairs of the input
    mapping. Input mapping fields can be marked as required using
    `ParserMappingValue(..., required=True)` as a value.

    `ParserMapping` instances can be arbitrarily nested inside input mappings of
    outer `ParserMapping` as values, to define more complex reqirements for the
    mappings to be validated and/or parsed.


    Examples
    --------

    ```py
    import re

    def raise_exception(exc):
        raise exc

    default_config_parser_mapping = ParserMapping(
        {
            "username": str,
            "password": lambda key, value, values_map: value # `values_map` is the value of the
            if isinstance(value, str) and len(value) > 8
            else raise_exception(
                ParserMapping.ParsingError(
                    f"value for field '{key}' must be a string longer than 8 characters"
                )
            ),
            "email": ParserMappingValue(
                lambda key, value, values_map: value
                if isinstance(value, str) and re.match(r"^[\\w\\.]+@[\\w\\.]+$")
                else raise_exception(
                    ParserMapping.ParsingError(
                        f"value for field '{key}' must be a string that is a valid email"
                    )
                ),
                required=True,
            ),
            ...: ...,
        },
    )

    parsed = default_config_parser_mapping.parse(
        {"username": "abc", "password": 123456789}
    )  # will raise an exception, as "email" is missing and "password" is of the wrong type.
    ```
    """

    __slots__ = (
        "_key",
        "_parent",
        "require_all",
        "reject_unknown",
    )

    class ParsingError(Exception):
        """A class for :class:`ParserMapping` related parsing errors."""

        pass

    def __init__(
        self,
        mapping: Mapping[
            str,
            Callable[[str, Any, MutableMapping[str, Any]], Any]
            | type
            | "ParserMapping"
            | "ParserMappingValue",
        ],
        require_all: bool = False,
        reject_unknown: bool = False,
    ):
        self._key: str | None = None
        self._parent: ParserMapping | None = None
        self.require_all = require_all
        self.reject_unknown = reject_unknown

        if not isinstance(mapping, Mapping):
            raise TypeError("argument 'mapping' must be a mapping object")

        temp_mapping = {}

        current_pmv = None

        _parser_lambda_map: dict[str, type] = {}
        # helper dictionary to hold classes passed as values to 'mapping', to avoid
        # local scope bugs with lambda functions defined in this method

        for k, v in tuple(mapping.items()):  # begin
            if isinstance(
                v, ParserMappingValue
            ):  # A ParserMappingValue was explicitly declared
                current_pmv = v
                v = v.value

            if isinstance(
                v, self.__class__
            ):  # build parent-child references with nested ParserMappings
                v._parent = self
                v._key = k
            elif isinstance(
                v, type
            ):  # convert class object to a validator using isinstance()
                cls = v

                callback = (
                    lambda key, value, mapping: value
                    if isinstance(value, _parser_lambda_map[key])  # type: ignore
                    else _raise_exc(
                        self.__class__.ParsingError(
                            f"value "
                            + (
                                f"at fully qualified key '{qk}': "
                                if (qk := self._get_qualified_key())
                                else ": "
                            )
                            + f"must be an instance of '{v.__name__}' not '{type(value).__name__}'"
                        )
                    )
                )

                _parser_lambda_map[k] = v

                if current_pmv:
                    current_pmv.value = callback
                else:
                    temp_mapping[k] = v = callback

            elif not callable(v):
                raise ValueError(
                    f"value for mapping key '{k}' is not a class, callable, ParserMappingValue or a "
                    f"'{self.__class__.__name__}' object"
                )

            current_pmv = None

        self.update(mapping)
        self.update(temp_mapping)

    def _get_qualified_key(self, sep: str = ".") -> str:
        keys = []
        if self._key:
            keys.append(self._key)
        curr_parent = self._parent

        while curr_parent is not None:
            if curr_parent._key is not None:
                keys.append(
                    f"'{curr_parent._key}'"
                    if "." in curr_parent._key
                    else curr_parent._key
                )
            curr_parent = curr_parent._parent

        return sep.join(reversed(keys))

    def parse(
        self, input_mapping: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        if not isinstance(input_mapping, MutableMapping):
            raise self.__class__.ParsingError(
                f"value"
                + (
                    f"at fully qualified key '{qk}': "
                    if (qk := self._get_qualified_key())
                    else ": "
                )
                + "must be an instance of a mutable mapping "
                "type instantiable without arguments"
            )

        if self.require_all and len(input_mapping) < len(self):
            raise self.__class__.ParsingError(
                "Parsing failed"
                + (
                    f"at fully qualified key " + f"'{qk}': "
                    if (qk := self._get_qualified_key())
                    else ": "
                )
                + "All fields are required "
            )

        elif self.reject_unknown:
            for key in input_mapping:
                if key not in self:
                    raise self.__class__.ParsingError(
                        "Parsing failed"
                        + (
                            f"at fully qualified key " + f"'{qk}': "
                            if (qk := self._get_qualified_key())
                            else ": "
                        )
                        + f"Key '{key}' is unknown"
                    )

        try:
            output_mapping = input_mapping.__class__()
        except TypeError as t:
            raise self.__class__.ParsingError(
                f"value"
                + (
                    f"at fully qualified key '{qk}': "
                    if (qk := self._get_qualified_key())
                    else ": "
                )
                + "must be an instance of a mutable mapping "
                "type instantiable without arguments (e.g. dict)"
            ) from t

        for k, v_or_pmv in self.items():
            was_pmv = False
            if isinstance(v_or_pmv, ParserMappingValue):
                was_pmv = True
                v = v_or_pmv.value
            else:
                v = v_or_pmv

            if k in input_mapping:
                if isinstance(v, self.__class__):
                    output_mapping[k] = v.parse(input_mapping[k])
                else:
                    output_mapping[k] = v(k, input_mapping[k], output_mapping)  # type: ignore
            elif was_pmv and v_or_pmv.required or self.require_all:  # type: ignore
                raise self.__class__.ParsingError(
                    f"mapping "
                    + (
                        f"at fully qualified key '{qk}' "
                        if (qk := self._get_qualified_key())
                        else " "
                    )
                    + f"is missing required key '{k}' "
                )

        return output_mapping

    def __or__(self, __value: Mapping[str, Any]) -> "ParserMapping":
        new_mapping = ParserMapping.__new__(ParserMapping)
        new_mapping.update(**self, **__value)
        for attr in self.__slots__:
            setattr(new_mapping, attr, getattr(self, attr))

        return new_mapping

    def __ior__(self, __value: Mapping[str, Any]) -> "ParserMapping":
        self.update(__value)
        return self

    def __ror__(self, __value: Mapping[str, Any]) -> "ParserMapping":
        self.update(__value)
        return self


@dataclass
class ParserMappingValue:
    value: Callable[[str, Any, MutableMapping], Any] | ParserMapping | type
    required: bool = False


class DefaultFormatter(logging.Formatter):
    default_msec_format = "%s.%03d"

    def formatException(self, ei):
        return f"...\n{super().formatException(ei)}\n..."


class ANSIColorFormatter(_ColourFormatter):
    FORMATS = {
        level: DefaultFormatter(
            f"\x1b[30;1m%(asctime)s\x1b[0m {colour}%(levelname)-8s\x1b[0m \x1b[35m%(name)s\x1b[0m %(message)s",
        )
        for level, colour in _ColourFormatter.LEVEL_COLOURS
    }


class RotatingTextIOHandler(logging.handlers.BaseRotatingHandler):
    """A subclass of `BaseRotatingHandler` that uses in-memory string buffers to
    write logging information to."""

    def __init__(
        self,
        maxBytes: int = 0,
        backupCount: int = 0,
        errors=None,
    ):
        self.maxBytes: int = maxBytes
        self.backupCount: int = backupCount

        self.baseFilename = ""
        self.mode = "a"
        self.encoding = "utf-8"
        self.errors = errors
        self.delay = False

        self.stream: io.StringIO = io.StringIO()
        self.streams: deque[io.StringIO] = deque()
        self.streams.append(self.stream)
        logging.Handler.__init__(self)

    def close(self):
        """
        Closes the stream.
        """
        self.acquire()
        try:
            try:
                if self.stream:
                    try:
                        self.flush()
                    finally:
                        for stream in self.streams:
                            if hasattr(stream, "close"):
                                stream.close()
                        self.stream = None  # type: ignore
            finally:
                # Issue #19523: call unconditionally to
                # prevent a handler leak when delay is set
                logging.StreamHandler.close(self)
        finally:
            self.release()

    def emit(self, record):
        """
        Emit a record.

        Output the record to the file, catering for rollover as described
        in doRollover().
        """
        try:
            if self.shouldRollover(record):
                self.doRollover()

            msg = self.format(record)
            stream = self.stream
            # issue 35046: merged two stream.writes into one.
            stream.write(msg + self.terminator)
            self.flush()
        except RecursionError:  # See issue 36272
            raise
        except Exception:
            self.handleError(record)

    def doRollover(self):
        """
        Do a rollover, as described in __init__().
        """
        old_stream = self.stream
        self.stream = io.StringIO()
        if self.backupCount > 0:
            self.streams.appendleft(self.stream)
            for _ in range(len(self.streams) - self.backupCount):
                self.streams.pop().close()
        else:
            old_stream.close()

    def shouldRollover(self, record):
        """
        Determine if rollover should occur.

        Basically, see if the supplied record would cause the file to exceed
        the size limit we have.
        """
        if self.maxBytes > 0:  # are we rolling over?
            msg = "%s\n" % self.format(record)
            self.stream.seek(0, 2)  # due to non-posix-compliant Windows feature
            if self.stream.tell() + len(msg) >= self.maxBytes:
                return True
        return False

    def __repr__(self):
        level = logging.getLevelName(self.level)
        return "<%s %s (%s)>" % (self.__class__.__name__, ":memory:", level)


class RotatingFileHandler(logging.handlers.RotatingFileHandler):
    """A :class:`logging.handlers.RotatingFileHandler` subclass that supports padding backup numbers
    with zeros for alphabetical file name sorting, as well as adding a file
    extension as a suffix behind backup numbers.
    """

    def __init__(
        self,
        filename,
        mode="a",
        maxBytes=0,
        backupCount=0,
        encoding=None,
        delay=False,
        errors=None,
        extension=None,
    ):
        self.extension = extension.removeprefix(".") if extension else None
        self.file_number_width = int(log10(backupCount)) + 1 if backupCount > 0 else 0
        super().__init__(
            filename,
            mode,
            maxBytes=maxBytes,
            backupCount=backupCount,
            encoding=encoding,
            delay=delay,
            errors=errors,
        )

    def _open(self):
        return open(
            self.baseFilename + f".{self.extension}" if self.extension else "",
            self.mode,
            encoding=self.encoding,
            errors=self.errors,
        )

    def doRollover(self):
        """
        Do a rollover, as described in __init__().
        """
        if self.stream:
            self.stream.close()
            self.stream = None  # type: ignore

        extension = f".{self.extension}" if self.extension else ""
        if self.backupCount > 0:
            truncBaseFileName = self.baseFilename
            zerosuff_start_idx = truncBaseFileName.rfind(".") + 1
            zero_suff = truncBaseFileName[zerosuff_start_idx:]

            if zero_suff.isnumeric() and all(
                s == "0" for s in zero_suff
            ):  # handle potential trailing zeros in file name
                truncBaseFileName = truncBaseFileName[: zerosuff_start_idx - 1]

            for i in range(self.backupCount - 1, 0, -1):
                sfn = self.rotation_filename(
                    f"{truncBaseFileName}.{i:0>{self.file_number_width}}{extension}"
                )
                dfn = self.rotation_filename(
                    f"{truncBaseFileName}.{i+1:0>{self.file_number_width}}{extension}"
                )
                if os.path.exists(sfn):
                    if os.path.exists(dfn):
                        os.remove(dfn)
                    os.rename(sfn, dfn)

            dfn = self.rotation_filename(
                f"{truncBaseFileName}.{1:0>{self.file_number_width}}{extension}"
            )  # add zero-padding to the backup number
            if os.path.exists(dfn):
                os.remove(dfn)
            self.rotate(self.baseFilename + extension, dfn)
        if not self.delay:
            self.stream = self._open()  # type: ignore


class DummyHandler(logging.NullHandler):
    """A subclass of `logging.NullHandler` that handles `LogRecord` objects to e.g.
    call filters.
    """

    def handle(self, record: logging.LogRecord) -> bool:
        return super(logging.NullHandler, self).handle(record)


class QueuingFilter(logging.Filter):
    """A logging filter that stores filtered log records in a queue."""

    def __init__(
        self,
        name: str = "",
        queue_level: int = logging.NOTSET,
        maxlen: int | None = None,
    ) -> None:
        super().__init__(name)
        self.queue: deque[logging.LogRecord] = deque(maxlen=maxlen)
        self.queue_level = queue_level

    def filter(self, record: logging.LogRecord) -> bool:
        if (can_pass := super().filter(record)) and record.levelno >= self.queue_level:
            self.queue.append(record)
        return can_pass


DEFAULT_FORMATTER = DefaultFormatter(
    "[{asctime}] [ {levelname:<8} ] {name} -- {message}", style="{"
)

DEFAULT_FORMATTER_REGEX = r"\[(\d{4}-\d\d-\d\d.\d\d:\d\d:\d\d\.\d\d\d)\] *\[ *(\S+) *\] *(.+) -- ((?:(?!\n\[(\d{4}-\d\d-\d\d.\d\d:\d\d:\d\d\.\d\d\d)\])\n|.)+)"
# detects formatted output written by DEFAULT_FORMATTER, including appended tracebacks
# on followup lines

ANSI_FORMATTER = ANSIColorFormatter()
