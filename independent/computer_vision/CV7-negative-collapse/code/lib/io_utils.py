"""CSV output and the kill-test line every experiment prints."""

from __future__ import annotations

import csv
from pathlib import Path


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _parse(value: str):
    if value in ("True", "False"):
        return value == "True"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def load_csv(path: Path) -> list[dict]:
    """Inverse of save_csv with numeric and boolean fields restored. Missing file -> []."""
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return [{key: _parse(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def kill_test(supported: bool | None, text: str) -> str:
    tag = {True: "SUPPORTED", False: "KILLED", None: "INCONCLUSIVE"}[supported]
    line = f"KILL TEST: {tag}. {text}"
    print(line)
    return line
