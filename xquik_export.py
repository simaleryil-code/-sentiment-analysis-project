from __future__ import annotations

import csv
import io
from typing import Any


TEXT_COLUMNS = (
    "text",
    "tweet_text",
    "full_text",
    "content",
    "body",
    "message",
    "comment",
)

USER_COLUMNS = (
    "username",
    "user",
    "author_username",
    "screen_name",
    "handle",
    "name",
)

ID_COLUMNS = (
    "id",
    "tweet_id",
    "post_id",
    "status_id",
    "source_id",
    "url",
)

DATE_COLUMNS = (
    "created_at",
    "date",
    "timestamp",
    "time",
)


def _decode_csv(raw_csv: bytes | str) -> str:
    if isinstance(raw_csv, bytes):
        return raw_csv.decode("utf-8-sig")
    return raw_csv


def _first_value(row: dict[str, Any], aliases: tuple[str, ...]) -> str:
    lookup = {key.strip().lower(): key for key in row}
    for alias in aliases:
        original_key = lookup.get(alias)
        if original_key is None:
            continue
        value = row.get(original_key)
        if value is None:
            continue
        cleaned = str(value).strip()
        if cleaned and cleaned.lower() != "nan":
            return cleaned
    return ""


def normalize_xquik_export(raw_csv: bytes | str) -> list[dict[str, str]]:
    csv_text = _decode_csv(raw_csv)
    reader = csv.DictReader(io.StringIO(csv_text))
    if not reader.fieldnames:
        return []

    comments: list[dict[str, str]] = []
    for row_number, row in enumerate(reader, start=1):
        comment = _first_value(row, TEXT_COLUMNS)
        if not comment:
            continue

        comments.append(
            {
                "comment": comment,
                "username": _first_value(row, USER_COLUMNS) or "xquik_export",
                "source_id": _first_value(row, ID_COLUMNS) or str(row_number),
                "created_at": _first_value(row, DATE_COLUMNS),
            }
        )

    return comments


def build_xquik_metadata(comments: list[dict[str, str]]) -> dict[str, str | int]:
    return {
        "idVideo": "Xquik CSV",
        "uniqueId": "xquik-export",
        "nickname": "Xquik Export",
        "description": "Imported CSV export",
        "totalLike": "N/A",
        "totalComment": len(comments),
        "totalShare": "N/A",
        "createTime": "Imported",
        "duration": "N/A",
        "fetchStatus": "Imported from Xquik CSV",
    }
