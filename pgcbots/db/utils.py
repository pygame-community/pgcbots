import logging
from typing import Any, Iterable, Sequence
import sqlalchemy
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine
import sqlalchemy.exc
from pgcbots.constants import UNSET

from pgcbots.types import ConfigDatabase, Database, ExtensionData, Revision

_logger = logging.getLogger(__name__)


def validate_revisions(lst: list[Revision]) -> list[Revision]:
    for j, revision in enumerate(lst):
        for k in ("date", "description", "migrate", "rollback", "delete"):
            if k in revision:
                if not all(
                    isinstance(dct, dict)
                    and all(
                        isinstance(stmt_lst, list)
                        and all(isinstance(s, str) for s in stmt_lst)
                        for stmt_lst in dct.values()
                    )
                    for k, dct in revision.items()
                    if k not in ("date", "description")
                ):
                    raise ValueError(
                        f"Invalid structure for revision {j}: Must match "
                        "'dict[str, dict[str, str | list[str]]]'"
                    )

            elif k == "delete" and j == 0:
                raise ValueError(
                    f"Revision dictionary 0 (first revision) must define "
                    "field 'delete'"
                )
            else:
                raise ValueError(
                    f"Revision dictionary {j} does not define required field '{k}'"
                )

    return lst


async def load_databases(
    db_info_data: Sequence[ConfigDatabase],
    raise_exceptions: bool = True,
) -> list[Database]:
    dbs = []

    for db_info_dict in db_info_data:
        db_name = db_info_dict["name"]
        engine = None

        try:
            engine_kwargs = {}

            if "connect_args" in db_info_dict:
                engine_kwargs["connect_args"] = db_info_dict["connect_args"]

            engine = create_async_engine(db_info_dict["url"], **engine_kwargs)

            async with engine.connect():  # test if connection is possible
                pass

        except sqlalchemy.exc.SQLAlchemyError as exc:
            _logger.error(
                f"Failed to create engine and functioning connection "
                + (f"'{engine.name}+{engine.driver}' " if engine is not None else "")
                + f"for database '{db_name}'",
                exc_info=exc,
            )

            if raise_exceptions:
                raise
        else:
            dbs.append({"name": db_name, "engine": engine, "url": db_info_dict["url"]})

            if "connect_args" in db_info_dict:
                dbs[db_name]["connect_args"] = db_info_data["connect_args"]  # type: ignore

            _logger.info(
                f"Successfully configured engine '{engine.name}+{engine.driver}' "
                f"for database '{db_name}'"
            )

    return dbs


async def unload_databases(
    dbs: Iterable[Database],
    raise_exceptions: bool = True,
):
    for db_dict in dbs:
        db_name = db_dict["name"]
        if not isinstance(db_dict["engine"], AsyncEngine):
            raise TypeError(
                "Value for 'engine' must be instance of "
                "'sqlalchemy.ext.asyncio.AsyncEngine' for all dicts in param 'dbs'"
            )

        engine: AsyncEngine = db_dict["engine"]

        try:
            await engine.dispose()
        except sqlalchemy.exc.SQLAlchemyError as err:
            _logger.error(
                f"Failed to dispose connection pool of engine"
                f" '{engine.name}+{engine.driver}' of database '{db_name}'",
                exc_info=err,
            )

            if raise_exceptions:
                raise
        else:
            _logger.info(
                "Successfully disposed connection pool of engine "
                f"'{engine.name}+{engine.driver}' of database '{db_name}'"
            )


async def pgcbots_db_schema_is_defined(db: Database) -> bool:  # type: ignore
    engine = db["engine"]
    if engine.name not in ("sqlite", "postgresql"):
        raise RuntimeError(
            f"Unsupported database dialect '{engine.name}' for main database,"
            " must be 'sqlite' or 'postgresql'"
        )

    async with engine.connect() as conn:
        if engine.name == "sqlite":
            return bool(
                (
                    await conn.execute(
                        sqlalchemy.text(
                            "SELECT EXISTS(SELECT 1 FROM sqlite_schema "
                            f"WHERE type == 'table' "
                            "AND name == 'pgcbots_db_schema')"
                        )
                    )
                ).scalar()
            )

        elif engine.name == "postgresql":
            return bool(
                (
                    await conn.execute(
                        sqlalchemy.text(
                            "SELECT EXISTS(SELECT 1 FROM "
                            "information_schema.tables "
                            "WHERE table_name == 'pgcbots_db_schema')"
                        )
                    )
                ).scalar()
            )


async def initialize_pgcbots_db_schema(
    db: Database,
    revisions: list[Revision],
    bot_name: str,
    bot_uid: str,
    auto_migrate: bool = True,
) -> int:
    engine = db["engine"]
    if engine.name not in ("sqlite", "postgresql"):
        raise RuntimeError(
            f"Unsupported database dialect '{engine.name}' for main database,"
            " must be 'sqlite' or 'postgresql'"
        )

    should_migrate = False
    revision_number = -1
    migration_count = 0
    is_initial_migration = False

    if await pgcbots_db_schema_is_defined(db):
        async with engine.connect() as conn:
            result_row = (
                await conn.execute(
                    sqlalchemy.text(
                        "SELECT value FROM globals " f"WHERE key == 'revision_number'"
                    )
                )
            ).one_or_none()

            if result_row:
                revision_number = int(result_row.value)

        if revision_number == -1 or (
            revision_number < len(revisions) - 1 and auto_migrate
        ):
            is_initial_migration = True
            should_migrate = True
    else:
        should_migrate = True

    if should_migrate:
        _logger.info(
            f"Performing "
            + (
                "initial "
                if is_initial_migration
                else "automatic "
                if auto_migrate
                else ""
            )
            + "bot database migration..."
        )
        migration_count = await migrate_pgcbots_db_schema(
            db, revisions, None if auto_migrate else 1
        )

    async with engine.begin() as conn:
        if not (
            await conn.execute(
                sqlalchemy.text("SELECT EXISTS(SELECT 1 FROM bots WHERE uid == :uid)"),
                dict(uid=bot_uid),
            )
        ).scalar():
            await conn.execute(  # register bot application into database
                sqlalchemy.text(f"INSERT INTO bots VALUES (:uid, :name)"),
                dict(uid=bot_uid, name=bot_name),
            )

    return migration_count


async def get_pgcbots_db_schema_revision_number(db: Database) -> int:
    revision_number = -1
    engine = db["engine"]

    conn: AsyncConnection
    async with engine.begin() as conn:
        if engine.name not in ("sqlite", "postgresql"):
            raise RuntimeError(
                f"Unsupported database dialect '{engine.name}' for main database, "
                "must be 'sqlite' or 'postgresql'"
            )

        if await pgcbots_db_schema_is_defined(db):
            result_row = (
                await conn.execute(
                    sqlalchemy.text(
                        "SELECT value FROM globals WHERE key == 'revision_number'"
                    )
                )
            ).one_or_none()

            if result_row:
                revision_number = int(result_row.value)

    return revision_number


async def migrate_pgcbots_db_schema(
    db: Database, revisions: list[Revision], steps: int | None = None
) -> int:  #
    old_revision_number = revision_number = -1
    migration_count = 0
    is_initial_migration = False

    engine = db["engine"]

    if steps and steps <= 0:
        raise ValueError("argument 'steps' must be None or > 0")

    _logger.info(
        f"Attempting bot database migration"
        + (f" ({steps} steps)..." if steps else "...")
    )

    conn: AsyncConnection
    async with engine.begin() as conn:
        if engine.name not in ("sqlite", "postgresql"):
            raise RuntimeError(
                f"Unsupported database dialect '{engine.name}' for main database, "
                "must be 'sqlite' or 'postgresql'"
            )

        if await pgcbots_db_schema_is_defined(db):
            result_row = (
                await conn.execute(
                    sqlalchemy.text(
                        "SELECT value FROM globals WHERE key == 'revision_number'"
                    )
                )
            ).one_or_none()

            if result_row:
                old_revision_number = revision_number = int(result_row.value)

        else:
            is_initial_migration = True

        for revision_number in range(
            old_revision_number + 1,
            (
                len(revisions)
                if not steps
                else min(old_revision_number + 1 + steps, len(revisions))
            ),
        ):
            for statement in revisions[revision_number]["migrate"][engine.name]:
                await conn.execute(sqlalchemy.text(statement))

            migration_count += 1

        # only runs if for-loop above did not run at all
        if revision_number == old_revision_number:
            _logger.info(
                f"Stored revision number {revision_number} already matches the "
                f"latest available revision ({len(revisions)-1}). No migration "
                "was performed."
            )
            return migration_count
        elif revision_number == old_revision_number == -1:
            _logger.info(
                f"No revisions available for migration. No migration was performed."
            )
            return migration_count

        _logger.info(
            f"Successfully performed {'initial ' if is_initial_migration else ''}"
            "bot database migration from revision number "
            f"{old_revision_number} to {revision_number}."
        )

        await conn.execute(
            sqlalchemy.text(
                f"INSERT INTO globals "
                f"VALUES ('revision_number', :new_revision_number_str) "
                "ON CONFLICT DO UPDATE SET value = :new_revision_number_str "
                "WHERE key == 'revision_number'"
            ),
            dict(new_revision_number_str=revision_number),
        )

    return migration_count


async def rollback_pgcbots_db_schema(
    db: Database, revisions: list[Revision], steps: int
) -> int:  #
    old_revision_number = revision_number = -1
    rollback_count = 0

    engine = db["engine"]

    if steps < 0:
        raise ValueError("argument 'steps' must be > 0")

    _logger.info(
        f"Attempting bot database rollback"
        + (f" ({steps} steps)..." if steps != -1 else "...")
    )

    conn: AsyncConnection
    async with engine.begin() as conn:
        if engine.name not in ("sqlite", "postgresql"):
            raise RuntimeError(
                f"Unsupported database dialect '{engine.name}' for main database, "
                "must be 'sqlite' or 'postgresql'"
            )

        if await pgcbots_db_schema_is_defined(db):
            result_row = (
                await conn.execute(
                    sqlalchemy.text(
                        "SELECT value FROM globals WHERE key == 'revision_number'"
                    )
                )
            ).one_or_none()

            if result_row:
                old_revision_number = revision_number = int(result_row.value)

        else:
            raise RuntimeError(
                "Failed to perform bot database rollback: Database is not configured "
                "or has incorrect schema structure"
            )

        if old_revision_number >= len(revisions):
            raise RuntimeError(
                f"Stored revision number {old_revision_number} exceeds "
                f"highest available revision number {len(revisions)-1}"
            )
        elif old_revision_number < 0:
            raise RuntimeError(
                f"Stored revision number {old_revision_number} must be >= 0 "
            )
        elif old_revision_number == 0:
            _logger.info(
                f"Stored revision number is already at 0. " "No rollback was performed."
            )
            return rollback_count

        for revision_number in range(
            old_revision_number, max(old_revision_number - steps, -1), -1
        ):
            for statement in revisions[revision_number]["rollback"][engine.name]:
                await conn.execute(sqlalchemy.text(statement))

            rollback_count += 1

        revision_number = old_revision_number - steps

        await conn.execute(
            sqlalchemy.text(
                f"INSERT INTO globals "
                f"VALUES ('revision_number', :new_revision_number_str) "
                "ON CONFLICT DO UPDATE SET value = :new_revision_number_str "
                "WHERE key == 'revision_number'"
            ),
            dict(new_revision_number_str=revision_number),
        )

    _logger.info(
        f"Successfully performed "
        "bot database rollback from revision number "
        f"{old_revision_number} to {revision_number}."
    )
    return rollback_count


async def delete_pgcbots_db_schema(db: Database, revisions: list[Revision]):
    engine = db["engine"]
    conn: AsyncConnection
    async with engine.begin() as conn:
        if engine.name not in ("sqlite", "postgresql"):
            raise RuntimeError(
                f"Unsupported database dialect '{engine.name}' for main database, "
                "must be 'sqlite' or 'postgresql'"
            )

        for i in range(-1, -len(revisions) - 1, -1):
            if "delete" not in revisions[i]:
                continue

            for statement in revisions[i]["migrate"][engine.name]:
                await conn.execute(sqlalchemy.text(statement))


async def create_extension_data(
    db: Database,
    name: str,
    revision_number: int,
    auto_migrate: bool,
    db_prefix: str,
    data: bytes | None = None,
) -> None:
    if not isinstance(name, str):
        raise TypeError(
            f"argument 'name' must be a fully qualified extension "
            "name of type 'str', not "
            f"'{name.__class__.__name__}'"
        )
    elif not isinstance(revision_number, int):
        raise TypeError(
            f"argument 'revision_number' must be of type 'int', not "
            f"'{revision_number.__class__.__name__}'"
        )
    elif not isinstance(auto_migrate, bool):
        raise TypeError(
            f"argument 'auto_migrate' must be of type 'bool', not "
            f"'{auto_migrate.__class__.__name__}'"
        )
    elif not isinstance(db_prefix, str):
        raise TypeError(
            f"argument 'db_prefix' must be of type 'str', not "
            f"'{db_prefix.__class__.__name__}'"
        )
    elif data is not None and not isinstance(data, bytes):
        raise TypeError(
            f"argument 'data' must be 'None' or of type 'bytes', "
            f"not '{data.__class__.__name__}'"
        )

    engine: AsyncEngine = db["engine"]  # type: ignore
    conn: AsyncConnection

    async with engine.begin() as conn:
        await conn.execute(
            sqlalchemy.text(
                "INSERT INTO bot_extensions "
                "(name, revision_number, auto_migrate, db_prefix, "
                "data) VALUES (:name, :revision_number, :auto_migrate, "
                ":db_prefix, :data)"
            ),
            dict(
                name=name,
                revision_number=revision_number,
                auto_migrate=auto_migrate,
                db_prefix=db_prefix,
                data=data,
            ),
        )


async def get_extension_data_names(db: Database) -> tuple[str, ...]:
    if not await pgcbots_db_schema_is_defined(db):
        return ()

    engine: AsyncEngine = db["engine"]
    conn: AsyncConnection
    async with engine.connect() as conn:
        result: sqlalchemy.engine.Result = await conn.execute(
            sqlalchemy.text(f"SELECT name FROM bot_extensions"),
        )

        rows: Any = result.all()
        return tuple(row.name for row in rows)


async def read_extension_data(
    db: Database, name: str, data: bool = True
) -> ExtensionData:
    if not isinstance(name, str):
        raise TypeError(
            f"argument 'name' must be of type 'str', not "
            f"'{name.__class__.__name__}'"
        )

    engine: AsyncEngine = db["engine"]
    conn: AsyncConnection

    columns = "*"

    if not data:
        columns = "name, revision_number, auto_migrate, db_prefix"

    async with engine.connect() as conn:
        result: sqlalchemy.engine.Result = await conn.execute(
            sqlalchemy.text(
                f"SELECT {columns} FROM bot_extensions WHERE name == :name"
            ),
            dict(name=name),
        )

        row: Any = result.first()
        if row is None:
            raise LookupError(
                f"Could not find extension storage data for extension named "
                f"'{name}'"
            )

        return ExtensionData(  # type: ignore
            name=row.name,
            revision_number=row.revision_number,
            auto_migrate=bool(row.auto_migrate),
            db_prefix=row.db_prefix,
        ) | (dict(data=row.data) if data else {})


async def extension_data_exists(db: Database, name: str) -> bool:
    if not isinstance(name, str):
        raise TypeError(
            f"argument 'name' must be a fully qualified extension "
            "name of type 'str', not "
            f"'{name.__class__.__name__}'"
        )

    engine: AsyncEngine = db["engine"]
    conn: AsyncConnection

    async with engine.connect() as conn:
        storage_exists = (await pgcbots_db_schema_is_defined(db)) and bool(
            (
                await conn.execute(
                    sqlalchemy.text(
                        "SELECT EXISTS(SELECT 1 FROM bot_extensions WHERE name == :name)"
                    ),
                    dict(name=name),
                )
            ).scalar()
        )
    return storage_exists


async def update_extension_data(
    db: Database,
    name: str,
    revision_number: int | None = UNSET,
    auto_migrate: bool | None = UNSET,
    db_prefix: str | None = UNSET,
    data: bytes | None = UNSET,
) -> None:
    if not isinstance(name, str):
        raise TypeError(
            f"argument 'name' must be a fully qualified extension "
            "name of type 'str', not "
            f"'{name.__class__.__name__}'"
        )
    elif revision_number is not UNSET and not isinstance(revision_number, int):
        raise TypeError(
            f"argument 'revision_number' must be of type 'int', not "
            f"'{revision_number.__class__.__name__}'"
        )
    elif auto_migrate is not UNSET and not isinstance(auto_migrate, bool):
        raise TypeError(
            f"argument 'auto_migrate' must be of type 'bool', not "
            f"'{auto_migrate.__class__.__name__}'"
        )
    elif db_prefix is not UNSET and not isinstance(db_prefix, str):
        raise TypeError(
            f"argument 'db_prefix' must be of type 'str', not "
            f"'{db_prefix.__class__.__name__}'"
        )
    elif data is not UNSET and not isinstance(data, (bytes, type(None))):
        raise TypeError(
            f"argument 'data' must be 'None' or of type 'bytes', "
            f"not '{data.__class__.__name__}'"
        )

    if all(
        field is UNSET for field in (revision_number, auto_migrate, db_prefix, data)
    ):
        raise TypeError(
            f"arguments 'revision_number', 'auto_migrate', 'db_prefix' "
            "and 'data' cannot all be 'None'"
        )

    engine: AsyncEngine = db["engine"]
    conn: AsyncConnection

    async with engine.begin() as conn:
        if not bool(
            (
                await conn.execute(
                    sqlalchemy.text(
                        "SELECT EXISTS(SELECT 1 FROM bot_extensions WHERE name == :name)"
                    ),
                    dict(name=name),
                )
            ).scalar()
        ):
            raise LookupError(
                f"Could not find extension storage data for extension named "
                f"'{name}'"
            )

        params = {}
        params["name"] = name
        params |= (
            dict(revision_number=revision_number)
            if revision_number is not UNSET
            else {}
        )
        params |= dict(auto_migrate=auto_migrate) if auto_migrate is not UNSET else {}
        params |= dict(db_prefix=db_prefix) if db_prefix is not UNSET else {}
        params |= dict(data=data) if data is not UNSET else {}

        target_columns = ", ".join((f"{k} = :{k}" for k in params))

        await conn.execute(
            sqlalchemy.text(
                "UPDATE bot_extensions AS be"
                + f" SET {target_columns}"
                + " WHERE be.name == :name",
            ),
            parameters=params,
        )


async def delete_extension_data(db: Database, name: str) -> None:
    if not isinstance(name, str):
        raise TypeError(
            f"argument 'name' must be a fully qualified extension "
            "name of type 'str', not "
            f"'{name.__class__.__name__}'"
        )

    engine: AsyncEngine = db["engine"]
    conn: AsyncConnection

    async with engine.begin() as conn:
        await conn.execute(
            sqlalchemy.text("DELETE FROM bot_extensions WHERE name == :name"),
            dict(name=name),
        )
