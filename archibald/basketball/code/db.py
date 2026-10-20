"""Write NBA frames into the local `nba` Postgres database."""

from __future__ import annotations

import math
import re

import pandas as pd
import psycopg
from psycopg import sql

DSN = "postgresql://nathanclark@127.0.0.1:5432/nba"


def connect():
    return psycopg.connect(DSN)


def ensure() -> None:
    with connect() as conn:
        conn.execute("CREATE SCHEMA IF NOT EXISTS league")
        conn.execute("CREATE SCHEMA IF NOT EXISTS games")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS league.loaded (
                name text PRIMARY KEY,
                n_rows bigint,
                loaded_at timestamptz DEFAULT now()
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS games.loaded (
                game_id text PRIMARY KEY,
                detail jsonb,
                loaded_at timestamptz DEFAULT now()
            )
            """
        )
        conn.commit()


def loaded(schema: str, name: str) -> bool:
    table = "loaded"
    with connect() as conn:
        row = conn.execute(
            sql.SQL("SELECT 1 FROM {}.{} WHERE {} = %s").format(
                sql.Identifier(schema),
                sql.Identifier(table),
                sql.Identifier("name" if schema == "league" else "game_id"),
            ),
            (name,),
        ).fetchone()
    return row is not None


def mark_loaded(schema: str, name: str, n_rows: int, detail: str | None = None) -> None:
    with connect() as conn:
        if schema == "league":
            conn.execute(
                """
                INSERT INTO league.loaded (name, n_rows)
                VALUES (%s, %s)
                ON CONFLICT (name) DO UPDATE SET n_rows = EXCLUDED.n_rows, loaded_at = now()
                """,
                (name, n_rows),
            )
        else:
            conn.execute(
                """
                INSERT INTO games.loaded (game_id, detail)
                VALUES (%s, %s::jsonb)
                ON CONFLICT (game_id) DO UPDATE SET detail = EXCLUDED.detail, loaded_at = now()
                """,
                (name, detail or "{}"),
            )
        conn.commit()


def _column_name(raw) -> str:
    text = str(raw).strip()
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    text = re.sub(r"[^0-9A-Za-z]+", "_", text).lower()
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "col"


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or (df.empty and len(df.columns) == 0):
        return pd.DataFrame()
    out = df.copy()
    out.columns = [_column_name(c) for c in out.columns]
    out = out.loc[:, ~out.columns.duplicated()]
    if "game_id" in out.columns:
        out["game_id"] = out["game_id"].map(_game_id)
    return out


def _game_id(value):
    if _missing(value):
        return None
    text = str(value).split(".")[0]
    if text.isdigit():
        return text.zfill(10)
    return text


def _missing(value) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        return False
    return False


def _pg_type(series: pd.Series, name: str) -> str:
    if name == "game_id":
        return "text"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_integer_dtype(series):
        return "bigint"
    if pd.api.types.is_float_dtype(series):
        return "double precision"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "timestamp"
    return "text"


def _python(value, pg_type: str):
    if _missing(value):
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass
    if pg_type == "boolean":
        if isinstance(value, str):
            return value.lower() in ("true", "t", "1")
        return bool(value)
    if pg_type == "bigint":
        return int(value)
    if pg_type == "double precision":
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    if pg_type == "timestamp":
        ts = pd.Timestamp(value)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        return ts.to_pydatetime()
    if pg_type == "text":
        return str(value)
    return value


def _existing_types(conn, schema: str, table: str) -> dict[str, str]:
    rows = conn.execute(
        """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
        """,
        (schema, table),
    ).fetchall()
    mapped = {}
    for name, data_type in rows:
        if data_type in ("bigint", "integer", "smallint"):
            mapped[name] = "bigint"
        elif data_type in ("double precision", "real", "numeric"):
            mapped[name] = "double precision"
        elif data_type == "boolean":
            mapped[name] = "boolean"
        elif "timestamp" in data_type:
            mapped[name] = "timestamp"
        else:
            mapped[name] = "text"
    return mapped


def write_frame(schema: str, table: str, df: pd.DataFrame, replace: bool = False) -> int:
    """Create or extend `schema.table` and copy rows into it. Returns rows written."""
    frame = normalize(df)
    if frame.empty and len(frame.columns) == 0:
        return 0
    with connect() as conn:
        types = {} if replace else _existing_types(conn, schema, table)
        if replace and types:
            conn.execute(
                sql.SQL("DROP TABLE {}.{}").format(sql.Identifier(schema), sql.Identifier(table))
            )
            types = {}
        if not types:
            col_defs = []
            for name in frame.columns:
                pg = _pg_type(frame[name], name)
                types[name] = pg
                col_defs.append(sql.SQL("{} {}").format(sql.Identifier(name), sql.SQL(pg)))
            conn.execute(
                sql.SQL("CREATE TABLE IF NOT EXISTS {}.{} ({})").format(
                    sql.Identifier(schema),
                    sql.Identifier(table),
                    sql.SQL(", ").join(col_defs),
                )
            )
        else:
            for name in frame.columns:
                if name not in types:
                    pg = _pg_type(frame[name], name)
                    types[name] = pg
                    conn.execute(
                        sql.SQL("ALTER TABLE {}.{} ADD COLUMN {} {}").format(
                            sql.Identifier(schema),
                            sql.Identifier(table),
                            sql.Identifier(name),
                            sql.SQL(pg),
                        )
                    )
        columns = list(types)
        for name in columns:
            if name not in frame.columns:
                frame[name] = None
        frame = frame[columns]
        copy_sql = sql.SQL("COPY {}.{} ({}) FROM STDIN").format(
            sql.Identifier(schema),
            sql.Identifier(table),
            sql.SQL(", ").join(sql.Identifier(c) for c in columns),
        )
        with conn.cursor() as cur:
            with cur.copy(copy_sql) as copy:
                for record in frame.itertuples(index=False, name=None):
                    copy.write_row(tuple(_python(value, types[col]) for col, value in zip(columns, record)))
        if "game_id" in columns and schema == "games" and table not in ("loaded", "games_index"):
            conn.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{} (game_id)").format(
                    sql.Identifier(f"{table}_game_id"),
                    sql.Identifier(schema),
                    sql.Identifier(table),
                )
            )
        conn.commit()
    return len(frame)


def clear_game(game_id: str) -> None:
    with connect() as conn:
        tables = conn.execute(
            """
            SELECT table_name FROM information_schema.columns
            WHERE table_schema = 'games' AND column_name = 'game_id'
              AND table_name NOT IN ('loaded', 'games_index')
            """
        ).fetchall()
        for (table,) in tables:
            conn.execute(
                sql.SQL("DELETE FROM games.{} WHERE game_id = %s").format(sql.Identifier(table)),
                (game_id,),
            )
        conn.commit()


def read_table(schema: str, table: str) -> pd.DataFrame:
    with connect() as conn:
        exists = conn.execute(
            """
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = %s AND table_name = %s
            """,
            (schema, table),
        ).fetchone()
        if not exists:
            return pd.DataFrame()
        rows = conn.execute(
            sql.SQL("SELECT * FROM {}.{}").format(sql.Identifier(schema), sql.Identifier(table))
        ).fetchall()
        cols = [desc.name for desc in conn.execute(
            sql.SQL("SELECT * FROM {}.{} LIMIT 0").format(sql.Identifier(schema), sql.Identifier(table))
        ).description]
    return pd.DataFrame(rows, columns=cols)
