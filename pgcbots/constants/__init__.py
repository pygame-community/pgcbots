"""
This file is a part of the source code for pgcbots.
This project has been licensed under the MIT license.
Copyright (c) 2023-present pygame-community

This file defines some constants used across the library.
"""

from typing import Any


class _SingletonMeta(type):
    def __init__(cls, name, bases, dct):
        super(_SingletonMeta, cls).__init__(name, bases, dct)
        super(_SingletonMeta, cls).__setattr__(f"_{cls.__name__}__inst", None)

    def __call__(cls, *args, **kw):
        if getattr(cls, f"_{cls.__name__}__inst", None) is None:
            super(_SingletonMeta, cls).__setattr__(
                f"_{cls.__name__}__inst",
                super(_SingletonMeta, cls).__call__(*args, **kw),
            )
        return getattr(cls, f"_{cls.__name__}__inst")

    def __setattr__(cls, name: str, value: object):
        if (
            name == f"_{cls.__name__}__inst"
            and getattr(cls, f"_{cls.__name__}__inst", None) is not None
        ):
            raise ValueError("cannot modify the specified attribute")

    def __delattr__(cls, name: str):
        if name == f"_{cls.__name__}__inst":
            raise ValueError("cannot delete the specified attribute")


class _UnsetType(metaclass=_SingletonMeta):
    __slots__ = ()

    def __eq__(self, other):
        return False

    def __bool__(self):
        return False

    def __hash__(self):
        return 0

    def __repr__(self):
        return "Unset"


# sentinel singleton for unused variables
UNSET: Any = _UnsetType()

# helpful constants
DEFAULT_FILESIZE_LIMIT = 8_000_000  # bytes
