from .utils import validate_revisions as _

INITIAL_REVISION = _(
    [
        {  # Revision: 0
            "date": "2023-06-27T00:30:30",
            "description": "Initial migration",
            "migrate": {
                "sqlite": [
                    "CREATE TABLE pgcbots_db_schema (id INTEGER)",  # dummy table for schema existence testing
                    ""
                    "CREATE TABLE globals ("
                    "   key VARCHAR (1000) PRIMARY KEY NOT NULL, "
                    "   value VARCHAR(1000000) NOT NULL)",
                    ""
                    "CREATE TABLE bots ("
                    "   uid VARCHAR(64) PRIMARY KEY NOT NULL, "
                    "   name VARCHAR(1000) NOT NULL) ",
                    ""
                    "CREATE TABLE bot_extensions ("
                    "   name VARCHAR(1000) PRIMARY KEY, "
                    "   revision_number INTEGER, "
                    "   auto_migrate INTEGER, "
                    "   db_prefix VARCHAR(10), "
                    "   data BLOB)",
                    ""
                    'INSERT INTO globals VALUES ("pgcbots_db_schema_version_number", "1")',
                ],
                "postgresql": [
                    "CREATE TABLE pgcbots_db_schema (id INTEGER)",
                    ""
                    "CREATE TABLE globals ("
                    "   key VARCHAR(1000) PRIMARY KEY NOT NULL, "
                    "   value VARCHAR(1000000) NOT NULL)",
                    ""
                    "CREATE TABLE bots ("
                    "   uid VARCHAR(64) PRIMARY KEY NOT NULL, "
                    "   name VARCHAR(1000) NOT NULL)",
                    ""
                    "CREATE TABLE bot_extensions ("
                    "   name VARCHAR(1000) PRIMARY KEY, "
                    "   revision_number INTEGER, "
                    "   auto_migrate SMALLINT, "
                    "   db_prefix VARCHAR(10), "
                    "   data BYTEA)"
                    ""
                    'INSERT INTO globals VALUES ("pgcbots_db_schema_version_number", "1")',
                ],
            },
            "rollback": {
                "sqlite": [
                    "DROP TABLE bot_extensions",
                    "DROP TABLE globals",
                    "DROP TABLE bots",
                ],
                "postgresql": [
                    "DROP TABLE bot_extensions",
                    "DROP TABLE globals",
                    "DROP TABLE bots",
                ],
            },
            "delete": {
                "sqlite": [
                    "DROP TABLE IF EXISTS bot_extensions",
                    "DROP TABLE IF EXISTS globals",
                    "DROP TABLE IF EXISTS bots",
                ],
                "postgresql": [
                    "DROP TABLE IF EXISTS bot_extensions",
                    "DROP TABLE IF EXISTS globals",
                    "DROP TABLE IF EXISTS bots",
                ],
            },
        }
    ]
)[0]
"""This revision defines the PGCBots Database Schema (v1.0) to be used by every bot application.
"""
