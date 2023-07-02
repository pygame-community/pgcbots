"""This file is a part of the source code for PygameCommunityBot.
This project has been licensed under the MIT license.
Copyright (c) 2023-present pygame-community.
"""

import asyncio
from collections import OrderedDict
import logging
from typing import TYPE_CHECKING, Any, Sequence, Union

import discord
from discord.ext import commands
from discord.utils import MISSING
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncConnection

from pgcbots.constants import UNSET

if TYPE_CHECKING:
    from pgcbots.bot import PGCBot, AutoShardedPGCBot

import pgcbots.utils
from pgcbots.utils.pagination import EmbedPaginator

from .types import Revision

BotT = commands.Bot | commands.AutoShardedBot
PGCBotT = Union["PGCBot", "AutoShardedPGCBot"]

UnsetType = type(UNSET)

_logger = logging.getLogger(__name__)


class BaseExtensionCog(commands.Cog):
    def __init__(self, bot: BotT, theme_color: int | discord.Color = 0) -> None:
        super().__init__()
        self.bot = bot
        self.theme_color = discord.Color(int(theme_color))
        self._global_cached_response_messages = (
            hasattr(bot, "cached_response_messages")
            and hasattr(bot, "cached_response_messages_maxsize")
            and isinstance(bot.cached_response_messages, OrderedDict)  # type: ignore
        )

        if self._global_cached_response_messages:
            self.cached_response_messages: OrderedDict[
                int, discord.Message
            ] = bot.cached_response_messages  # type: ignore
            self.cached_response_messages_maxsize = int(
                bot.cached_response_messages_maxsize  # type: ignore
            )
        else:
            self.cached_response_messages: OrderedDict[
                int, discord.Message
            ] = OrderedDict()
            self.cached_response_messages_maxsize: int = 50

        self._global_cached_embed_paginators = (
            hasattr(bot, "cached_embed_paginators")
            and hasattr(bot, "cached_embed_paginators_maxsize")
            and isinstance(bot.cached_embed_paginators, OrderedDict)  # type: ignore
        )

        if self._global_cached_embed_paginators:
            self.cached_embed_paginators: OrderedDict[  # type: ignore
                int, tuple[EmbedPaginator, asyncio.Task[None]]
            ] = bot.cached_embed_paginators  # type: ignore
            self.cached_embed_paginators_maxsize = int(
                bot.cached_embed_paginators_maxsize  # type: ignore
            )
        else:
            self.cached_embed_paginators: OrderedDict[
                int, tuple[EmbedPaginator, asyncio.Task[None]]
            ] = OrderedDict()
            self.cached_embed_paginators_maxsize: int = 50

    async def cog_after_invoke(self, ctx: commands.Context[BotT]) -> None:
        if (
            not self._global_cached_response_messages
            and not self._global_cached_embed_paginators
        ):
            for _ in range(
                min(
                    100,
                    max(
                        len(self.cached_response_messages)
                        - self.cached_response_messages_maxsize,
                        0,
                    ),
                )
            ):
                _, response_message = self.cached_response_messages.popitem(last=False)
                paginator_tuple = self.cached_embed_paginators.get(response_message.id)
                if paginator_tuple is not None and paginator_tuple[0].is_running():  # type: ignore
                    paginator_tuple[1].cancel()  # type: ignore

        elif not self._global_cached_response_messages:
            for _ in range(
                min(
                    100,
                    max(
                        len(self.cached_response_messages)
                        - self.cached_response_messages_maxsize,
                        0,
                    ),
                )
            ):
                self.cached_response_messages.popitem(last=False)

        elif not self._global_cached_embed_paginators:
            for _ in range(
                min(
                    100,
                    max(
                        len(self.cached_embed_paginators)
                        - self.cached_embed_paginators_maxsize,
                        0,
                    ),
                )
            ):
                _, paginator_tuple = self.cached_embed_paginators.popitem(last=False)
                if paginator_tuple[0].is_running():  # type: ignore
                    paginator_tuple[1].cancel()  # type: ignore

    async def send_or_edit_response(
        self,
        ctx: commands.Context[BotT],
        content: str | None = UNSET,
        *,
        tts: bool = UNSET,
        embed: discord.Embed | None = UNSET,
        embeds: Sequence[discord.Embed] | None = UNSET,
        attachments: Sequence[discord.Attachment | discord.File] = UNSET,
        file: discord.File | None = UNSET,
        files: Sequence[discord.File] | None = UNSET,
        stickers: Sequence[discord.GuildSticker | discord.StickerItem] | None = None,
        delete_after: float | None = None,
        nonce: str | int | None = None,
        allowed_mentions: discord.AllowedMentions | None = UNSET,
        reference: discord.Message
        | discord.MessageReference
        | discord.PartialMessage
        | None = None,
        mention_author: bool | None = None,
        view: discord.ui.View | None = UNSET,
        suppress_embeds: bool = False,
        suppress: bool = False,
        destination: discord.abc.Messageable | None = None,
    ) -> discord.Message:  # type: ignore
        suppress_embeds = suppress or suppress_embeds
        destination = destination or ctx.channel

        send = False
        if response_message := self.cached_response_messages.get(ctx.message.id):
            try:
                return await response_message.edit(
                    content=MISSING if content is UNSET else content,
                    embed=MISSING if embed is UNSET else embed,
                    embeds=MISSING if embeds is UNSET else embeds,
                    attachments=MISSING if attachments is UNSET else attachments,
                    delete_after=MISSING if delete_after is UNSET else delete_after,
                    allowed_mentions=MISSING
                    if allowed_mentions is UNSET
                    else allowed_mentions,
                    suppress=MISSING if suppress_embeds is UNSET else suppress_embeds,
                    view=MISSING if view is UNSET else view,
                )  # type: ignore
            except discord.NotFound:
                send = True
        else:
            send = True

        if send:
            self.cached_response_messages[
                ctx.message.id
            ] = response_message = await destination.send(
                content=None if content is UNSET else content,
                tts=None if tts is UNSET else tts,
                embed=None if embed is UNSET else embed,
                embeds=None if embeds is UNSET else embeds,
                file=None if file is UNSET else file,
                files=None if files is UNSET else files,
                stickers=stickers,
                delete_after=delete_after,
                nonce=nonce,
                allowed_mentions=None
                if allowed_mentions is UNSET
                else allowed_mentions,
                reference=reference,
                mention_author=mention_author,
                view=None if view is UNSET else view,
                suppress_embeds=suppress_embeds,
            )  # type: ignore

            return response_message

    async def send_paginated_response_embeds(
        self,
        ctx: commands.Context[BotT],
        *embeds: discord.Embed,
        member: discord.Member | Sequence[discord.Member] | None = None,
        inactivity_timeout: int | None = 60,
        destination: discord.TextChannel
        | discord.VoiceChannel
        | discord.Thread
        | None = None,
    ):
        if not ctx.guild:
            raise ValueError(
                "invocation context 'ctx' must have a '.guild' associated with it."
            )

        assert isinstance(ctx.author, discord.Member) and isinstance(
            ctx.channel, (discord.TextChannel, discord.VoiceChannel, discord.Thread)
        )
        # this shouldn't normally be false
        paginator = None

        if not embeds:
            return

        destination = destination or ctx.channel

        if (
            response_message := self.cached_response_messages.get(ctx.message.id)
        ) is not None:
            try:
                if (
                    paginator_tuple := self.cached_embed_paginators.get(
                        response_message.id
                    )
                ) is not None:
                    if paginator_tuple[0].is_running():
                        await paginator_tuple[0].stop()

                if len(embeds) == 1:
                    await response_message.edit(embed=embeds[0])
                    return

                paginator = pgcbots.utils.pagination.EmbedPaginator(
                    (
                        response_message := await response_message.edit(
                            content="\u200b", embed=None
                        )
                    ),
                    *embeds,
                    member=member or ctx.author,
                    inactivity_timeout=60,
                    theme_color=int(self.theme_color),
                )
            except discord.NotFound:
                if len(embeds) == 1:
                    self.cached_response_messages[
                        ctx.message.id
                    ] = await destination.send(embed=embeds[0])
                    return

                paginator = pgcbots.utils.pagination.EmbedPaginator(
                    (response_message := await destination.send(content="\u200b")),
                    *embeds,
                    member=member or ctx.author,
                    inactivity_timeout=60,
                    theme_color=int(self.theme_color),
                )
        else:
            if len(embeds) == 1:  # don't use paginator for single embed
                self.cached_response_messages[ctx.message.id] = await destination.send(
                    embed=embeds[0]
                )
                return

            paginator = pgcbots.utils.pagination.EmbedPaginator(
                (response_message := await destination.send(content="\u200b")),
                *embeds,
                member=member or ctx.author,
                inactivity_timeout=inactivity_timeout
                if inactivity_timeout is not None
                else 60,
                theme_color=int(self.theme_color),
            )

        paginator_tuple = (
            paginator,
            asyncio.create_task(
                paginator.mainloop(client=ctx.bot),
                name=f"embed_paginator({response_message.jump_url})",
            ),
        )

        self.cached_response_messages[ctx.message.id] = response_message
        self.cached_embed_paginators[response_message.id] = paginator_tuple


class ExtensionManager:
    """A helper class for managing bot extensions."""

    def __init__(
        self,
        name: str,
        revisions: list[Revision],
        default_auto_migrate: bool,
        db_prefix: str,
    ) -> None:
        """Create an extension manager instance for a bot extension module.

        Parameters
        ----------
        name : str
            The extension name.
        migrations : list[Revision]
            The list of database revision dictionaries used for extension database
            object migration/rollback.
        default_auto_migrate : bool
            Whether automatic migration for extension database objects (upon extension
            loading at runtime) should be performed by default.
        db_prefix : str
            The name prefix to use for all database objects of this extension.
        """
        self.name = name
        self.revisions = revisions
        self.default_auto_migrate = default_auto_migrate
        self.db_prefix = db_prefix

    async def migrate(self, bot: PGCBotT, steps: int | None = None) -> int:
        """Perform a database migration for all database objects
        used by the managed extension.

        Parameters
        ----------
        bot : PGCBotT
            The bot object to use for database engine retrieval and extension
            metadata access. Must not be connected
            to Discord.
        steps : int, optional
            The maximum amount of migration steps to take (revision count
            to apply). None performs the maximum amount possible.
            Defaults to None.

        Returns
        -------
        int
            The total amount of migration steps that occured
            successfully.

        Raises
        ------
        RuntimeError
            Migration failed due to an error.
        DBAPIError
            DB-API related SQLAlchemyError.
        SQLAlchemyError
            Generic SQLAlchemyError.
        ValueError
            Invalid function arguments.
        """

        migration_count = 0
        db_engine = bot.get_database_engine()
        if not isinstance(db_engine, AsyncEngine):
            raise RuntimeError(
                "Failed to retrieve main database engine of type "
                "'sqlalchemy.ext.asyncio.AsyncEngine'"
            )
        elif db_engine.name not in ("sqlite", "postgresql"):
            raise RuntimeError(
                f"Unsupported database dialect '{db_engine.name}' for main database, "
                "must be 'sqlite' or 'postgresql'"
            )

        if steps and steps <= 0:
            raise ValueError("argument 'steps' must be None or > 0")

        if extension_data_exists := await bot.extension_data_exists(self.name):
            extension_data = await bot.read_extension_data(self.name)
        else:
            extension_data = dict(
                name=self.name,
                revision_number=-1,
                auto_migrate=self.default_auto_migrate,
                db_prefix=self.db_prefix,
            )

        _logger.info(
            f"Attempting migration for extension '{self.name}' "
            + (f"({steps} steps)..." if steps != None else "")
        )

        is_initial_migration = extension_data["revision_number"] == -1
        old_revision_number: int = extension_data["revision_number"]  # type: ignore
        revision_number = old_revision_number

        if old_revision_number >= len(self.revisions):
            raise RuntimeError(
                f"Stored revision number {old_revision_number} exceeds "
                f"highest available revision number {len(self.revisions)-1}"
            )

        conn: AsyncConnection
        async with db_engine.begin() as conn:
            for revision_number in range(
                old_revision_number + 1,
                (
                    len(self.revisions)
                    if not steps
                    else min(old_revision_number + 1 + steps, len(self.revisions))
                ),
            ):
                for stmt in self.revisions[revision_number]["migrate"][db_engine.name]:
                    await conn.execute(text(stmt))

                migration_count += 1

        if revision_number == old_revision_number:
            _logger.info(
                f"Stored revision number {revision_number} already matches the "
                "latest available revision, No migration was performed."
            )
            return migration_count

        extension_data["revision_number"] = revision_number
        if extension_data_exists:
            await bot.update_extension_data(**extension_data)  # type: ignore
        else:
            await bot.create_extension_data(**extension_data)  # type: ignore

        if is_initial_migration:
            _logger.info(
                f"Successfully performed initial migration for extension '{self.name}' "
                f"to revision number {revision_number}."
            )
        else:
            _logger.info(
                f"Successfully performed migration for extension '{self.name}' from "
                f"revision number {old_revision_number} to {revision_number}."
            )

        return migration_count

    async def rollback(self, bot: PGCBotT, steps: int) -> int:
        """Roll back the last 'steps' revisions that were applied
        by `.migrate`.

        Parameters
        ----------
        bot : PGCBotT
            The bot object to use for database engine retrieval.
        steps : int
            The maximum amount of rollback steps to take.

        Returns
        -------
        int
            The amount of rollbacks that occured.

        Raises
        ------
        RuntimeError
            Migration failed due to an error.
        DBAPIError
            DB-API related SQLAlchemyError.
        SQLAlchemyError
            Generic SQLAlchemyError.
        ValueError
            Invalid function arguments.
        """
        rollback_count = 0
        db_engine = bot.get_database_engine()
        if not isinstance(db_engine, AsyncEngine):
            raise RuntimeError(
                "Failed to retrieve main database engine of type "
                "'sqlalchemy.ext.asyncio.AsyncEngine'"
            )
        elif db_engine.name not in ("sqlite", "postgresql"):
            raise RuntimeError(
                f"Unsupported database dialect '{db_engine.name}' for main database, "
                "must be 'sqlite' or 'postgresql'"
            )

        if steps <= 0:
            raise ValueError("argument 'steps' must be greater than 0")

        if await bot.extension_data_exists(self.name):
            extension_data = await bot.read_extension_data(self.name)
        else:
            raise RuntimeError(
                f"No bot extension data found for extension '{self.name}'"
            )

        _logger.info(
            f"Attempting rollback for extension '{self.name}' "
            f"({min(steps, len(self.revisions))} steps)..."
        )

        old_revision_number: int = extension_data["revision_number"]  # type: ignore
        revision_number = old_revision_number

        if old_revision_number >= len(self.revisions):
            raise RuntimeError(
                f"Stored revision number {old_revision_number} exceeds "
                f"highest available revision number {len(self.revisions)-1}"
            )
        elif old_revision_number < 0:
            raise RuntimeError(
                f"Stored revision number {old_revision_number} must be >= 0 "
            )
        elif old_revision_number == 0:
            _logger.info(
                f"Stored revision number for extension '{self.name}' is 0 "
                "No rollback was performed."
            )
            return rollback_count

        conn: AsyncConnection
        async with db_engine.begin() as conn:
            for revision_number in range(
                old_revision_number, max(old_revision_number - steps, -1), -1
            ):
                for statement in self.revisions[revision_number]["rollback"][
                    db_engine.name
                ]:
                    await conn.execute(text(statement))

                rollback_count += 1

        revision_number = old_revision_number - steps

        # save new revision number determined by for-loop
        extension_data["revision_number"] = revision_number

        await bot.update_extension_data(**extension_data)  # type: ignore
        _logger.info(
            f"Successfully performed {rollback_count} rollbacks for extension "
            f"'{self.name}' from "
            f"revision number {old_revision_number} to "
            f"{revision_number}."
        )

        return rollback_count

    async def delete(self, bot: PGCBotT):
        """Delete all database information of the managed extension.

        Parameters
        ----------
        bot : PGCBotT
            The bot object to use for database engine retrieval.

        Raises
        ------
        RuntimeError
            Deletion failed due to an error.
        DBAPIError
            DB-API related SQLAlchemyError.
        SQLAlchemyError
            Generic SQLAlchemyError.
        ValueError
            Invalid function arguments.
        """
        db_engine = bot.get_database_engine()
        if not isinstance(db_engine, AsyncEngine):
            raise RuntimeError(
                "Failed to retrieve main database engine of type "
                "'sqlalchemy.ext.asyncio.AsyncEngine'"
            )
        elif db_engine.name not in ("sqlite", "postgresql"):
            raise RuntimeError(
                f"Unsupported database dialect '{db_engine.name}' for main database, "
                "must be 'sqlite' or 'postgresql'"
            )

        if await bot.extension_data_exists(self.name):
            extension_data = await bot.read_extension_data(self.name)
        else:
            raise RuntimeError(
                f"No bot extension data found for extension '{self.name}'"
            )

        _logger.info(f"Attempting data deletion for extension '{self.name}'...")

        old_revision_number: int = extension_data["revision_number"]  # type: ignore
        revision_number = old_revision_number

        if old_revision_number >= len(self.revisions):
            raise RuntimeError(
                f"Stored revision number {old_revision_number} exceeds "
                f"highest available revision number {len(self.revisions)-1}"
            )

        conn: AsyncConnection
        async with db_engine.begin() as conn:
            for revision_number in range(old_revision_number, -1, -1):
                if "delete" not in self.revisions[revision_number]:
                    continue

                for stmt in self.revisions[revision_number]["delete"][db_engine.name]:
                    await conn.execute(text(stmt))

                break

            else:
                raise RuntimeError(
                    f"Bot extension '{self.name}' does not define 'delete' fields"
                    " in its migration data"
                )

        await bot.delete_extension_data(self.name)  # type: ignore

        _logger.info(f"Successfully deleted all data of extension '{self.name}'.")

    async def prepare(self, bot: PGCBotT, initial_migration_steps: int = UNSET) -> int:
        """Helper method for running database-related boilerplate
        extension setup code.

        Parameters
        ----------
        bot : PGCBotT
            The bot loading this extension.

        initial_migration_steps : int, optional
            How many initial migration steps to perform if an initial
            migration was not yet performed for the bot extension.
            A value of ``0`` implies that no steps should be performed.
            Omission of this argument implies the maximum possible amount
            of steps.

        Returns
        -------
            The amount of initial migration steps performed successfully.

        Raises
        ------
        RuntimeError
            Setup preparation failed.
        """

        migration_count = 0
        db_engine = bot.get_database_engine()

        if not isinstance(db_engine, AsyncEngine):
            raise RuntimeError(
                "Failed to retrieve main database engine of type "
                "'sqlalchemy.ext.asyncio.AsyncEngine'"
            )
        elif db_engine.name not in ("sqlite", "postgresql"):
            raise RuntimeError(
                f"Unsupported database dialect '{db_engine.name}' for main database, "
                "must be 'sqlite' or 'postgresql'"
            )

        if extension_data_existed := await bot.extension_data_exists(self.name):
            extension_data = await bot.read_extension_data(self.name)
        else:
            extension_data = dict(
                name=self.name,
                revision_number=-1,
                auto_migrate=self.default_auto_migrate,
                db_prefix=self.db_prefix,
            )

        stored_revision_number: int = extension_data["revision_number"]  # type: ignore
        max_revision_number = len(self.revisions) - 1

        if stored_revision_number > max_revision_number:
            raise RuntimeError(
                f"Extension data found for '{self.name}' is incompatible: Stored "
                f"migration revision number {stored_revision_number} "
                f"exceeds highest available revision number {max_revision_number}"
            )
        elif stored_revision_number < max_revision_number:
            if not extension_data_existed and initial_migration_steps != 0:
                _logger.info(
                    f"No previous extension data found for extension '{self.name}', "
                    "performing migration..."
                )
                migration_count = await self.migrate(
                    bot, steps=initial_migration_steps or None
                )

            elif extension_data["auto_migrate"]:
                _logger.info(
                    f"Auto-migration is enabled for extension '{self.name}', "
                    "performing migration..."
                )
                await self.migrate(bot)
            else:
                raise RuntimeError(
                    "Extension setup preparation failed: "
                    "Auto-migration is disabled, and "
                    f"{max_revision_number-stored_revision_number} migrations are "
                    "unapplied "
                )

        async with db_engine.begin() as conn:
            if (
                await conn.execute(
                    text(
                        f"SELECT EXISTS(SELECT 1 FROM sqlite_master "
                        f"WHERE type='table' AND name='{self.db_prefix}bots')"
                        if db_engine.name == "sqlite"
                        else "SELECT EXISTS(SELECT 1 FROM "
                        "information_schema.tables "
                        f"WHERE table_name == '{self.db_prefix}bots')"
                    )
                )
            ).scalar():
                await conn.execute(  # register bot into extension-specific bots table (needed for bot-specific cascading data deletion per-extension)
                    text(
                        f"INSERT INTO {self.db_prefix}bots VALUES (:uid) "
                        "ON CONFLICT DO NOTHING"
                    ),
                    dict(uid=bot.uid),
                )

        return migration_count

    async def setup(self, bot: BotT):
        raise NotImplementedError("This method must be inherited in your bot extension")

    async def teardown(self, bot: BotT):
        raise NotImplementedError("This method must be inherited in your bot extension")
