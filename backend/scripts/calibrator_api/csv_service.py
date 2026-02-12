from __future__ import annotations

import base64
import binascii
import csv
import io
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple


ENCODINGS_TO_TRY = ["utf-8-sig", "utf-8", "latin-1"]
DELIMITERS = ",;\t|"
ID_COLUMN_CANDIDATES = ["StudyID", "study_id", "id", "patient_id", "record_id"]
REPORT_COLUMN_CANDIDATES = ["Report", "report_text", "report", "report_body", "narrative"]


class CsvServiceError(ValueError):
    """Raised for invalid CSV inputs."""


def normalize_column_name(value: str) -> str:
    return str(value).replace("\ufeff", "").strip()


def _decode_bytes(raw_bytes: bytes) -> Tuple[str, str]:
    last_error = "unknown decode error"
    for encoding in ENCODINGS_TO_TRY:
        try:
            return raw_bytes.decode(encoding), encoding
        except UnicodeDecodeError as exc:
            last_error = str(exc)
    raise CsvServiceError(f"Failed to decode CSV bytes with supported encodings. Last error: {last_error}")


def _detect_encoding_from_file(csv_path: Path) -> str:
    if not csv_path.exists():
        raise CsvServiceError(f"CSV file not found: {csv_path}")

    with csv_path.open("rb") as handle:
        sample_bytes = handle.read(65536)

    if not sample_bytes:
        raise CsvServiceError("CSV payload is empty.")

    _, encoding = _decode_bytes(sample_bytes)
    return encoding


def _detect_dialect(sample: str) -> csv.Dialect:
    try:
        return csv.Sniffer().sniff(sample, delimiters=DELIMITERS)
    except csv.Error:
        return csv.excel


def _read_csv_text(csv_text: str) -> Tuple[List[str], List[Dict[str, str]], str]:
    handle = io.StringIO(csv_text)
    sample = handle.read(16384)
    handle.seek(0)

    dialect = _detect_dialect(sample)
    reader = csv.DictReader(handle, dialect=dialect)

    raw_headers = list(reader.fieldnames or [])
    headers: List[str] = [normalize_column_name(header) for header in raw_headers if header is not None]
    if not headers:
        raise CsvServiceError("No header columns detected.")

    if len(set(headers)) != len(headers):
        raise CsvServiceError(
            f"Duplicate header names after normalization are not supported. Headers: {headers}"
        )

    rows: List[Dict[str, str]] = []
    for raw_row in reader:
        row: Dict[str, str] = {}
        for raw_key, raw_value in raw_row.items():
            key = normalize_column_name(raw_key or "")
            if not key:
                continue
            row[key] = str(raw_value or "")
        if row:
            rows.append(row)

    return headers, rows, str(getattr(dialect, "delimiter", ","))


def _decode_base64_payload(file_base64: str) -> bytes:
    try:
        return base64.b64decode(file_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise CsvServiceError(f"Invalid base64 CSV payload: {exc}") from exc


def _pick_existing_column(headers: Sequence[str], candidates: Sequence[str]) -> str:
    lowered_to_actual = {column.lower(): column for column in headers}
    for candidate in candidates:
        existing = lowered_to_actual.get(candidate.lower())
        if existing:
            return existing
    return ""


def _resolve_requested_column(headers: Sequence[str], requested: str) -> str:
    requested_clean = normalize_column_name(requested)
    if not requested_clean:
        return ""
    if requested_clean in headers:
        return requested_clean
    lowered = requested_clean.lower()
    for column in headers:
        if column.lower() == lowered:
            return column
    return requested_clean


def _build_preview_rows(rows: Sequence[Dict[str, str]], preview_rows: int) -> List[Dict[str, Any]]:
    limit = max(1, min(int(preview_rows or 5), 100))
    preview: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows[:limit], start=1):
        preview.append({"row_number": idx, "values": row})
    return preview


def _build_parse_response(
    *,
    source_name: str,
    headers: List[str],
    rows: List[Dict[str, str]],
    encoding: str,
    delimiter: str,
    preview_rows: int,
    requested_id_column: str,
    requested_report_column: str,
) -> Dict[str, Any]:
    resolved_id = _resolve_requested_column(headers, requested_id_column)
    if resolved_id and resolved_id not in headers:
        resolved_id = ""

    resolved_report = _resolve_requested_column(headers, requested_report_column)
    if resolved_report and resolved_report not in headers:
        resolved_report = ""

    inferred_id_column = resolved_id or _pick_existing_column(headers, ID_COLUMN_CANDIDATES)
    inferred_report_column = resolved_report or _pick_existing_column(headers, REPORT_COLUMN_CANDIDATES)

    return {
        "source": source_name,
        "columns": headers,
        "row_count": len(rows),
        "preview": _build_preview_rows(rows, preview_rows=preview_rows),
        "encoding": encoding,
        "delimiter": delimiter,
        "inferred_id_column": inferred_id_column,
        "inferred_report_column": inferred_report_column,
    }


def parse_csv_bytes(
    raw_bytes: bytes,
    *,
    preview_rows: int = 5,
    requested_id_column: str = "",
    requested_report_column: str = "",
    source_name: str = "uploaded.csv",
) -> Dict[str, Any]:
    rows_payload = read_csv_rows_from_bytes(raw_bytes, source_name=source_name)
    return _build_parse_response(
        source_name=rows_payload["source"],
        headers=rows_payload["columns"],
        rows=rows_payload["rows"],
        encoding=rows_payload["encoding"],
        delimiter=rows_payload["delimiter"],
        preview_rows=preview_rows,
        requested_id_column=requested_id_column,
        requested_report_column=requested_report_column,
    )


def parse_csv_file(
    csv_path: Path,
    *,
    preview_rows: int = 5,
    requested_id_column: str = "",
    requested_report_column: str = "",
) -> Dict[str, Any]:
    rows_payload = read_csv_rows_from_file(csv_path)
    return _build_parse_response(
        source_name=rows_payload["source"],
        headers=rows_payload["columns"],
        rows=rows_payload["rows"],
        encoding=rows_payload["encoding"],
        delimiter=rows_payload["delimiter"],
        preview_rows=preview_rows,
        requested_id_column=requested_id_column,
        requested_report_column=requested_report_column,
    )


def parse_csv_base64(
    file_base64: str,
    *,
    file_name: str = "uploaded.csv",
    preview_rows: int = 5,
    requested_id_column: str = "",
    requested_report_column: str = "",
) -> Dict[str, Any]:
    rows_payload = read_csv_rows_from_base64(file_base64, file_name=file_name)
    return _build_parse_response(
        source_name=rows_payload["source"],
        headers=rows_payload["columns"],
        rows=rows_payload["rows"],
        encoding=rows_payload["encoding"],
        delimiter=rows_payload["delimiter"],
        preview_rows=preview_rows,
        requested_id_column=requested_id_column,
        requested_report_column=requested_report_column,
    )


def read_csv_rows_from_bytes(raw_bytes: bytes, *, source_name: str = "uploaded.csv") -> Dict[str, Any]:
    if not raw_bytes:
        raise CsvServiceError("CSV payload is empty.")

    csv_text, encoding = _decode_bytes(raw_bytes)
    headers, rows, delimiter = _read_csv_text(csv_text)
    return {
        "source": source_name,
        "columns": headers,
        "rows": rows,
        "row_count": len(rows),
        "encoding": encoding,
        "delimiter": delimiter,
        "raw_bytes": raw_bytes,
    }


def read_csv_rows_from_file(csv_path: Path) -> Dict[str, Any]:
    if not csv_path.exists():
        raise CsvServiceError(f"CSV file not found: {csv_path}")

    raw_bytes = csv_path.read_bytes()
    return read_csv_rows_from_bytes(raw_bytes, source_name=str(csv_path))


def read_csv_rows_from_base64(file_base64: str, *, file_name: str = "uploaded.csv") -> Dict[str, Any]:
    raw_bytes = _decode_base64_payload(file_base64)
    return read_csv_rows_from_bytes(raw_bytes, source_name=file_name)


@contextmanager
def open_csv_stream(csv_path: Path) -> Iterator[Tuple[List[str], Iterator[Dict[str, str]], str, str]]:
    encoding = _detect_encoding_from_file(csv_path)
    handle = csv_path.open("r", encoding=encoding, newline="")
    try:
        sample = handle.read(16384)
        handle.seek(0)

        dialect = _detect_dialect(sample)
        reader = csv.DictReader(handle, dialect=dialect)
        raw_headers = list(reader.fieldnames or [])
        headers = [normalize_column_name(header) for header in raw_headers if header is not None]
        if not headers:
            raise CsvServiceError("No header columns detected.")
        if len(set(headers)) != len(headers):
            raise CsvServiceError(
                f"Duplicate header names after normalization are not supported. Headers: {headers}"
            )

        delimiter = str(getattr(dialect, "delimiter", ","))

        def row_iterator() -> Iterator[Dict[str, str]]:
            for raw_row in reader:
                row: Dict[str, str] = {}
                for raw_key, raw_value in raw_row.items():
                    key = normalize_column_name(raw_key or "")
                    if not key:
                        continue
                    row[key] = str(raw_value or "")
                if row:
                    yield row

        yield headers, row_iterator(), encoding, delimiter
    finally:
        handle.close()


def read_csv_headers(csv_path: Path) -> Dict[str, Any]:
    with open_csv_stream(csv_path) as (headers, _row_iterator, encoding, delimiter):
        return {
            "source": str(csv_path),
            "columns": headers,
            "encoding": encoding,
            "delimiter": delimiter,
        }
