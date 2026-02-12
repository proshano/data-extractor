from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .calibration_service import CalibrationOptions, run_feature_test
from .csv_service import open_csv_stream

PREVIEW_ROW_LIMIT = 25


def now_utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


@dataclass
class ExtractionJobRecord:
    job_id: str
    status: str
    total_rows: int
    id_column: str
    report_column: str
    output_csv_path: str
    completed_rows: int = 0
    ok_rows: int = 0
    error_rows: int = 0
    progress_percent: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    preview_rows: List[Dict[str, Any]] = field(default_factory=list)
    last_row: Optional[Dict[str, Any]] = None
    error: str = ""
    cancel_requested: bool = False


class ExtractionJobNotFoundError(KeyError):
    """Raised when an extraction job id is unknown."""


class ExtractionJobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, ExtractionJobRecord] = {}
        self._lock = threading.Lock()

    def create_job(
        self,
        *,
        features: Sequence[Dict[str, Any]],
        rows: Sequence[Dict[str, str]],
        id_column: str,
        report_column: str,
        options: CalibrationOptions,
        output_csv_path: Path,
    ) -> str:
        if not features:
            raise ValueError("At least one feature is required for extraction.")
        if not rows:
            raise ValueError("No input rows available for extraction.")
        if not id_column.strip():
            raise ValueError("id_column is required.")
        if not report_column.strip():
            raise ValueError("report_column is required.")

        output_csv_path.parent.mkdir(parents=True, exist_ok=True)

        job_id = uuid.uuid4().hex
        record = ExtractionJobRecord(
            job_id=job_id,
            status="pending",
            total_rows=len(rows),
            id_column=id_column,
            report_column=report_column,
            output_csv_path=str(output_csv_path),
        )

        with self._lock:
            self._jobs[job_id] = record

        thread = threading.Thread(
            target=self._run_job,
            args=(
                job_id,
                list(features),
                [dict(row) for row in rows],
                id_column,
                report_column,
                options,
                output_csv_path,
            ),
            daemon=True,
        )
        thread.start()
        return job_id

    def _run_job(
        self,
        job_id: str,
        features: List[Dict[str, Any]],
        rows: List[Dict[str, str]],
        id_column: str,
        report_column: str,
        options: CalibrationOptions,
        output_csv_path: Path,
    ) -> None:
        feature_names = [_safe_text(feature.get("name")) for feature in features]
        feature_defaults = {
            _safe_text(feature.get("name")): _safe_text(feature.get("missing_value_rule")) or "NA"
            for feature in features
        }
        include_study_id_alias = id_column.lower() != "study_id"
        fieldnames = [
            id_column,
            *(["study_id"] if include_study_id_alias else []),
            *feature_names,
            "_status",
            "_error_count",
            "_errors",
            "_processed_at",
            "_model",
            "_experiment_id",
            "_experiment_name",
            "_source_row_number",
        ]

        with self._lock:
            record = self._jobs[job_id]
            record.status = "running"
            record.started_at = time.time()
            record.updated_at = time.time()

        def should_cancel() -> bool:
            with self._lock:
                return self._jobs[job_id].cancel_requested

        ok_rows = 0
        error_rows = 0

        try:
            with output_csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                handle.flush()
                os.fsync(handle.fileno())

                total_rows = max(len(rows), 1)
                for row_index, source_row in enumerate(rows, start=1):
                    if should_cancel():
                        break

                    record_id = _safe_text(source_row.get(id_column)) or f"row_{row_index}"
                    report_text = _safe_text(source_row.get(report_column))
                    output_row: Dict[str, Any] = {id_column: record_id, **feature_defaults}
                    if include_study_id_alias:
                        output_row["study_id"] = record_id
                    output_row["_processed_at"] = now_utc_iso()
                    output_row["_experiment_id"] = _safe_text(options.experiment_id)
                    output_row["_experiment_name"] = _safe_text(options.experiment_name)
                    output_row["_source_row_number"] = row_index

                    row_errors: List[str] = []
                    model_value = ""

                    if not report_text:
                        row_errors.append(f"{report_column}:empty_report_text")
                    else:
                        for feature in features:
                            feature_name = _safe_text(feature.get("name"))
                            missing_value = feature_defaults.get(feature_name, "NA")
                            result = run_feature_test(
                                feature=feature,
                                report_text=report_text,
                                options=options,
                                should_cancel=should_cancel,
                            )

                            extracted_value = _safe_text(result.get("value")) or missing_value
                            output_row[feature_name] = extracted_value

                            result_model = _safe_text(result.get("model"))
                            if not model_value and result_model:
                                model_value = result_model

                            status = _safe_text(result.get("status"))
                            if status != "ok":
                                error_detail = _safe_text(result.get("error")) or status or "error"
                                row_errors.append(f"{feature_name}:{error_detail}")

                    output_row["_model"] = model_value
                    output_row["_error_count"] = len(row_errors)
                    output_row["_errors"] = " | ".join(row_errors)
                    if row_errors:
                        output_row["_status"] = "error"
                        error_rows += 1
                    else:
                        output_row["_status"] = "ok"
                        ok_rows += 1

                    writer.writerow(output_row)
                    handle.flush()
                    if row_index % 20 == 0:
                        os.fsync(handle.fileno())

                    last_row = {
                        "row_number": row_index,
                        id_column: record_id,
                        "study_id": record_id,
                        "_status": output_row["_status"],
                        "_error_count": output_row["_error_count"],
                    }

                    with self._lock:
                        current = self._jobs[job_id]
                        current.completed_rows = row_index
                        current.ok_rows = ok_rows
                        current.error_rows = error_rows
                        current.progress_percent = int((row_index / total_rows) * 100)
                        current.last_row = last_row
                        if len(current.preview_rows) < PREVIEW_ROW_LIMIT:
                            current.preview_rows.append(last_row)
                        current.updated_at = time.time()

                handle.flush()
                os.fsync(handle.fileno())

            with self._lock:
                current = self._jobs[job_id]
                current.finished_at = time.time()
                current.updated_at = time.time()
                if current.cancel_requested:
                    current.status = "cancelled"
                else:
                    current.status = "completed"
                    current.progress_percent = 100
        except Exception as exc:
            with self._lock:
                current = self._jobs[job_id]
                current.status = "failed"
                current.error = str(exc)
                current.finished_at = time.time()
                current.updated_at = time.time()

    def get_job(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise ExtractionJobNotFoundError(job_id)
            record = self._jobs[job_id]
            return {
                "job_id": record.job_id,
                "status": record.status,
                "total_rows": record.total_rows,
                "completed_rows": record.completed_rows,
                "ok_rows": record.ok_rows,
                "error_rows": record.error_rows,
                "progress_percent": record.progress_percent,
                "id_column": record.id_column,
                "report_column": record.report_column,
                "output_csv_path": record.output_csv_path,
                "created_at": record.created_at,
                "updated_at": record.updated_at,
                "started_at": record.started_at,
                "finished_at": record.finished_at,
                "preview_rows": list(record.preview_rows),
                "last_row": record.last_row,
                "error": record.error,
                "cancel_requested": record.cancel_requested,
            }

    def get_output_csv_path(self, job_id: str) -> Path:
        with self._lock:
            if job_id not in self._jobs:
                raise ExtractionJobNotFoundError(job_id)
            return Path(self._jobs[job_id].output_csv_path)

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise ExtractionJobNotFoundError(job_id)
            record = self._jobs[job_id]
            if record.status in {"completed", "failed", "cancelled"}:
                return {
                    "job_id": record.job_id,
                    "status": record.status,
                    "cancel_requested": record.cancel_requested,
                }
            record.cancel_requested = True
            record.updated_at = time.time()
            return {
                "job_id": record.job_id,
                "status": record.status,
                "cancel_requested": True,
            }


class ExtractionV2JobNotFoundError(KeyError):
    """Raised when a v2 extraction job id is unknown."""


class ExtractionV2ConflictError(ValueError):
    def __init__(
        self,
        *,
        code: str,
        message: str,
        reason: str,
        remediation: str,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = {
            "reason": reason,
            "remediation": remediation,
        }


@dataclass(frozen=True)
class ExtractionV2Context:
    input_signature: Dict[str, Any]
    schema_signature: str
    options_hash: str
    id_column: str
    report_column: str
    feature_columns: List[str]
    output_fieldnames: List[str]
    write_raw_response: bool


@dataclass(frozen=True)
class ExtractionV2RunSetup:
    resume_mode: str
    processed_rows_at_start: int
    ok_rows_at_start: int
    error_rows_at_start: int
    checkpoint_created_at: str
    state_path: Path
    output_mode: str
    needs_header: bool


@dataclass
class ExtractionV2JobRecord:
    job_id: str
    status: str
    id_column: str
    report_column: str
    output_csv_path: str
    resume_mode: str
    processed_rows_at_start: int
    processed_rows: int = 0
    ok_rows: int = 0
    error_rows: int = 0
    total_rows: Optional[int] = None
    progress_percent: Optional[int] = None
    last_source_row_number: Optional[int] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: str = ""
    cancel_requested: bool = False


class ExtractionJobStoreV2:
    def __init__(self) -> None:
        self._jobs: Dict[str, ExtractionV2JobRecord] = {}
        self._active_output_paths: Dict[str, str] = {}
        self._lock = threading.Lock()

    def create_job(
        self,
        *,
        features: Sequence[Dict[str, Any]],
        input_csv_path: Path,
        id_column: str,
        report_column: str,
        options: CalibrationOptions,
        output_csv_path: Path,
        resume: bool,
        overwrite_output: bool,
        write_raw_response: bool,
    ) -> Tuple[str, str, int]:
        if not features:
            raise ValueError("At least one feature is required for extraction.")
        if not id_column.strip():
            raise ValueError("id_column is required.")
        if not report_column.strip():
            raise ValueError("report_column is required.")
        if not input_csv_path.exists():
            raise ValueError(f"Input CSV not found: {input_csv_path}")

        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        state_path = self._build_state_path(output_csv_path)
        context = self._build_context(
            features=features,
            input_csv_path=input_csv_path,
            id_column=id_column,
            report_column=report_column,
            options=options,
            write_raw_response=write_raw_response,
        )

        output_key = str(output_csv_path.resolve())
        with self._lock:
            active_job_id = self._active_output_paths.get(output_key)
            if active_job_id:
                raise ExtractionV2ConflictError(
                    code="v2_output_locked",
                    message=(
                        f"Output is already being written by job {active_job_id}: {output_csv_path}"
                    ),
                    reason="output_locked",
                    remediation="Wait for the active job to finish or choose a different output_name.",
                )

            setup = self._prepare_run_setup(
                output_csv_path=output_csv_path,
                state_path=state_path,
                context=context,
                resume=resume,
                overwrite_output=overwrite_output,
            )

            job_id = uuid.uuid4().hex
            record = ExtractionV2JobRecord(
                job_id=job_id,
                status="pending",
                id_column=id_column,
                report_column=report_column,
                output_csv_path=str(output_csv_path),
                resume_mode=setup.resume_mode,
                processed_rows_at_start=setup.processed_rows_at_start,
                processed_rows=setup.processed_rows_at_start,
                ok_rows=setup.ok_rows_at_start,
                error_rows=setup.error_rows_at_start,
            )
            self._jobs[job_id] = record
            self._active_output_paths[output_key] = job_id

        thread = threading.Thread(
            target=self._run_job,
            args=(
                job_id,
                list(features),
                input_csv_path,
                id_column,
                report_column,
                options,
                output_csv_path,
                context,
                setup,
            ),
            daemon=True,
        )
        thread.start()
        return job_id, setup.resume_mode, setup.processed_rows_at_start

    def _run_job(
        self,
        job_id: str,
        features: List[Dict[str, Any]],
        input_csv_path: Path,
        id_column: str,
        report_column: str,
        options: CalibrationOptions,
        output_csv_path: Path,
        context: ExtractionV2Context,
        setup: ExtractionV2RunSetup,
    ) -> None:
        feature_defaults = {
            _safe_text(feature.get("name")): _safe_text(feature.get("missing_value_rule")) or "NA"
            for feature in features
        }

        with self._lock:
            record = self._jobs[job_id]
            record.status = "running"
            record.started_at = time.time()
            record.updated_at = time.time()

        def should_cancel() -> bool:
            with self._lock:
                return self._jobs[job_id].cancel_requested

        processed_rows = setup.processed_rows_at_start
        ok_rows = setup.ok_rows_at_start
        error_rows = setup.error_rows_at_start
        last_source_row_number: Optional[int] = None

        try:
            checkpoint_created_at = setup.checkpoint_created_at
            initial_checkpoint = self._build_checkpoint_payload(
                context=context,
                processed_rows=processed_rows,
                ok_rows=ok_rows,
                error_rows=error_rows,
                last_source_row_number=last_source_row_number,
                created_at=checkpoint_created_at,
            )
            self._write_checkpoint(setup.state_path, initial_checkpoint)

            with output_csv_path.open(setup.output_mode, encoding="utf-8", newline="") as output_handle:
                writer = csv.DictWriter(
                    output_handle,
                    fieldnames=context.output_fieldnames,
                    extrasaction="ignore",
                )

                if setup.needs_header:
                    writer.writeheader()
                    output_handle.flush()
                    os.fsync(output_handle.fileno())

                with open_csv_stream(input_csv_path) as (_headers, row_iterator, _encoding, _delimiter):
                    for source_row_number, source_row in enumerate(row_iterator, start=1):
                        if source_row_number <= setup.processed_rows_at_start:
                            continue

                        if should_cancel():
                            break

                        output_row = self._extract_output_row(
                            source_row=source_row,
                            source_row_number=source_row_number,
                            features=features,
                            feature_defaults=feature_defaults,
                            id_column=id_column,
                            report_column=report_column,
                            options=options,
                            write_raw_response=context.write_raw_response,
                            should_cancel=should_cancel,
                        )
                        writer.writerow(output_row)
                        output_handle.flush()
                        if source_row_number % 20 == 0:
                            os.fsync(output_handle.fileno())

                        processed_rows += 1
                        last_source_row_number = source_row_number
                        if output_row.get("_status") == "ok":
                            ok_rows += 1
                        else:
                            error_rows += 1

                        checkpoint = self._build_checkpoint_payload(
                            context=context,
                            processed_rows=processed_rows,
                            ok_rows=ok_rows,
                            error_rows=error_rows,
                            last_source_row_number=last_source_row_number,
                            created_at=checkpoint_created_at,
                        )
                        self._write_checkpoint(setup.state_path, checkpoint)

                        with self._lock:
                            current = self._jobs[job_id]
                            current.processed_rows = processed_rows
                            current.ok_rows = ok_rows
                            current.error_rows = error_rows
                            current.last_source_row_number = last_source_row_number
                            current.updated_at = time.time()

                output_handle.flush()
                os.fsync(output_handle.fileno())

            with self._lock:
                current = self._jobs[job_id]
                current.finished_at = time.time()
                current.updated_at = time.time()
                current.processed_rows = processed_rows
                current.ok_rows = ok_rows
                current.error_rows = error_rows
                current.last_source_row_number = last_source_row_number
                if current.cancel_requested:
                    current.status = "cancelled"
                else:
                    current.status = "completed"
                    current.total_rows = processed_rows
                    current.progress_percent = 100

            final_checkpoint = self._build_checkpoint_payload(
                context=context,
                processed_rows=processed_rows,
                ok_rows=ok_rows,
                error_rows=error_rows,
                last_source_row_number=last_source_row_number,
                created_at=setup.checkpoint_created_at,
            )
            self._write_checkpoint(setup.state_path, final_checkpoint)
        except Exception as exc:
            with self._lock:
                current = self._jobs[job_id]
                current.status = "failed"
                current.error = str(exc)
                current.finished_at = time.time()
                current.updated_at = time.time()
                current.processed_rows = processed_rows
                current.ok_rows = ok_rows
                current.error_rows = error_rows
                current.last_source_row_number = last_source_row_number
        finally:
            output_key = str(output_csv_path.resolve())
            with self._lock:
                self._active_output_paths.pop(output_key, None)

    def _extract_output_row(
        self,
        *,
        source_row: Dict[str, str],
        source_row_number: int,
        features: Sequence[Dict[str, Any]],
        feature_defaults: Dict[str, str],
        id_column: str,
        report_column: str,
        options: CalibrationOptions,
        write_raw_response: bool,
        should_cancel: Any,
    ) -> Dict[str, str]:
        record_id = _safe_text(source_row.get(id_column)) or f"row_{source_row_number}"
        report_text = _safe_text(source_row.get(report_column))
        include_study_id_alias = id_column.lower() != "study_id"

        output_row: Dict[str, str] = {id_column: record_id, **feature_defaults}
        if include_study_id_alias:
            output_row["study_id"] = record_id
        output_row["_processed_at"] = now_utc_iso()
        output_row["_model"] = ""
        if write_raw_response:
            output_row["_raw_response"] = ""

        if not report_text:
            output_row["_status"] = "llm_error"
            output_row["_error"] = f"{report_column}:empty_report_text"
            return output_row

        error_parts: List[str] = []
        raw_responses: Dict[str, str] = {}
        status_rank = {"ok": 0, "llm_error": 1, "parse_error": 2}
        row_status = "ok"

        for feature in features:
            feature_name = _safe_text(feature.get("name"))
            if not feature_name:
                continue

            result = run_feature_test(
                feature=feature,
                report_text=report_text,
                options=options,
                should_cancel=should_cancel,
            )
            missing_value = feature_defaults.get(feature_name, "NA")
            extracted_value = _safe_text(result.get("value")) or missing_value
            output_row[feature_name] = extracted_value

            model_value = _safe_text(result.get("model"))
            if model_value and not output_row["_model"]:
                output_row["_model"] = model_value

            feature_status = _safe_text(result.get("status")) or "llm_error"
            if feature_status != "ok":
                error_detail = _safe_text(result.get("error")) or feature_status
                error_parts.append(f"{feature_name}:{error_detail}")
                if status_rank.get(feature_status, 1) > status_rank.get(row_status, 0):
                    row_status = feature_status

            if write_raw_response:
                raw_response = str(result.get("raw_response") or "").strip()
                if raw_response:
                    raw_responses[feature_name] = raw_response

        if error_parts:
            output_row["_status"] = row_status
            output_row["_error"] = " | ".join(error_parts)
        else:
            output_row["_status"] = "ok"
            output_row["_error"] = ""

        if write_raw_response:
            output_row["_raw_response"] = (
                json.dumps(raw_responses, ensure_ascii=True) if raw_responses else ""
            )
        return output_row

    def _prepare_run_setup(
        self,
        *,
        output_csv_path: Path,
        state_path: Path,
        context: ExtractionV2Context,
        resume: bool,
        overwrite_output: bool,
    ) -> ExtractionV2RunSetup:
        output_exists = output_csv_path.exists()
        state_exists = state_path.exists()

        if resume:
            if output_exists and state_exists:
                checkpoint = self._read_checkpoint(state_path)
                self._validate_checkpoint_compatibility(
                    checkpoint=checkpoint,
                    context=context,
                    output_csv_path=output_csv_path,
                )

                processed_rows = self._as_non_negative_int(checkpoint.get("processed_rows", 0))
                ok_rows = self._as_non_negative_int(checkpoint.get("ok_rows", 0))
                error_rows = self._as_non_negative_int(checkpoint.get("error_rows", 0))
                resume_mode = "resumed" if processed_rows > 0 else "fresh"
                created_at = str(checkpoint.get("created_at") or now_utc_iso())
                return ExtractionV2RunSetup(
                    resume_mode=resume_mode,
                    processed_rows_at_start=processed_rows,
                    ok_rows_at_start=ok_rows,
                    error_rows_at_start=error_rows,
                    checkpoint_created_at=created_at,
                    state_path=state_path,
                    output_mode="a",
                    needs_header=False,
                )

            if output_exists and not state_exists:
                raise ExtractionV2ConflictError(
                    code="v2_resume_conflict",
                    message=(
                        "Resume requested but checkpoint file is missing for existing output: "
                        f"{output_csv_path}"
                    ),
                    reason="missing_checkpoint",
                    remediation="Use overwrite_output=true to start fresh or choose a different output_name.",
                )

            if not output_exists and state_exists:
                raise ExtractionV2ConflictError(
                    code="v2_resume_conflict",
                    message=(
                        "Resume requested but output CSV is missing while checkpoint exists: "
                        f"{output_csv_path}"
                    ),
                    reason="missing_output_csv",
                    remediation="Use overwrite_output=true to start fresh or choose a different output_name.",
                )

            return ExtractionV2RunSetup(
                resume_mode="fresh",
                processed_rows_at_start=0,
                ok_rows_at_start=0,
                error_rows_at_start=0,
                checkpoint_created_at=now_utc_iso(),
                state_path=state_path,
                output_mode="w",
                needs_header=True,
            )

        if output_exists and not overwrite_output:
            raise ExtractionV2ConflictError(
                code="v2_output_exists",
                message=f"Output CSV already exists: {output_csv_path}",
                reason="output_exists",
                remediation="Set overwrite_output=true or choose a different output_name.",
            )

        return ExtractionV2RunSetup(
            resume_mode="fresh",
            processed_rows_at_start=0,
            ok_rows_at_start=0,
            error_rows_at_start=0,
            checkpoint_created_at=now_utc_iso(),
            state_path=state_path,
            output_mode="w",
            needs_header=True,
        )

    def _validate_checkpoint_compatibility(
        self,
        *,
        checkpoint: Dict[str, Any],
        context: ExtractionV2Context,
        output_csv_path: Path,
    ) -> None:
        checkpoint_input_signature = checkpoint.get("input_signature", {})
        if checkpoint_input_signature != context.input_signature:
            raise ExtractionV2ConflictError(
                code="v2_resume_conflict",
                message="Resume checkpoint does not match current input CSV signature.",
                reason="input_signature_mismatch",
                remediation="Use overwrite_output=true to start fresh or choose a different output_name.",
            )

        if str(checkpoint.get("schema_signature", "")).strip() != context.schema_signature:
            raise ExtractionV2ConflictError(
                code="v2_resume_conflict",
                message="Resume checkpoint does not match current schema features.",
                reason="schema_mismatch",
                remediation="Use overwrite_output=true to start fresh or choose a different output_name.",
            )

        if str(checkpoint.get("options_hash", "")).strip() != context.options_hash:
            raise ExtractionV2ConflictError(
                code="v2_resume_conflict",
                message="Resume checkpoint does not match current extraction options.",
                reason="options_mismatch",
                remediation="Use overwrite_output=true to start fresh or choose a different output_name.",
            )

        columns = checkpoint.get("columns", {})
        if not isinstance(columns, dict):
            columns = {}

        if str(columns.get("id_column", "")).strip() != context.id_column:
            raise ExtractionV2ConflictError(
                code="v2_resume_conflict",
                message="Resume checkpoint id_column does not match request.",
                reason="id_column_mismatch",
                remediation="Use overwrite_output=true to start fresh or choose a different output_name.",
            )

        if str(columns.get("report_column", "")).strip() != context.report_column:
            raise ExtractionV2ConflictError(
                code="v2_resume_conflict",
                message="Resume checkpoint report_column does not match request.",
                reason="report_column_mismatch",
                remediation="Use overwrite_output=true to start fresh or choose a different output_name.",
            )

        feature_columns = columns.get("feature_columns", [])
        if list(feature_columns) != context.feature_columns:
            raise ExtractionV2ConflictError(
                code="v2_resume_conflict",
                message="Resume checkpoint feature columns do not match request.",
                reason="feature_columns_mismatch",
                remediation="Use overwrite_output=true to start fresh or choose a different output_name.",
            )

        output_fieldnames = columns.get("output_fieldnames", [])
        if list(output_fieldnames) != context.output_fieldnames:
            raise ExtractionV2ConflictError(
                code="v2_resume_conflict",
                message="Resume checkpoint output schema does not match request.",
                reason="output_schema_mismatch",
                remediation="Use overwrite_output=true to start fresh or choose a different output_name.",
            )

        header = self._read_csv_header(output_csv_path)
        if header != context.output_fieldnames:
            raise ExtractionV2ConflictError(
                code="v2_resume_conflict",
                message="Existing output CSV header does not match expected output schema.",
                reason="output_header_mismatch",
                remediation="Use overwrite_output=true to start fresh or choose a different output_name.",
            )

    def _build_context(
        self,
        *,
        features: Sequence[Dict[str, Any]],
        input_csv_path: Path,
        id_column: str,
        report_column: str,
        options: CalibrationOptions,
        write_raw_response: bool,
    ) -> ExtractionV2Context:
        feature_columns = [_safe_text(feature.get("name")) for feature in features if _safe_text(feature.get("name"))]
        output_fieldnames = [
            id_column,
            *(["study_id"] if id_column.lower() != "study_id" else []),
            *feature_columns,
            "_status",
            "_error",
            "_processed_at",
            "_model",
        ]
        if write_raw_response:
            output_fieldnames.append("_raw_response")

        return ExtractionV2Context(
            input_signature=self._build_input_signature(input_csv_path),
            schema_signature=self._stable_hash(features),
            options_hash=self._stable_hash(
                {
                    "llama_url": _safe_text(options.llama_url),
                    "temperature": float(options.temperature),
                    "max_retries": int(options.max_retries),
                    "model": _safe_text(options.model),
                    "experiment_id": _safe_text(options.experiment_id),
                    "experiment_name": _safe_text(options.experiment_name),
                    "system_instructions": _safe_text(options.system_instructions),
                    "extraction_instructions": _safe_text(options.extraction_instructions),
                    "reasoning_mode": _safe_text(options.reasoning_mode),
                    "reasoning_instructions": _safe_text(options.reasoning_instructions),
                    "output_instructions": _safe_text(options.output_instructions),
                    "judge_enabled": bool(options.judge_enabled),
                    "judge_model": _safe_text(options.judge_model),
                    "judge_instructions": _safe_text(options.judge_instructions),
                    "judge_acceptance_threshold": float(options.judge_acceptance_threshold),
                    "write_raw_response": bool(write_raw_response),
                }
            ),
            id_column=id_column,
            report_column=report_column,
            feature_columns=feature_columns,
            output_fieldnames=output_fieldnames,
            write_raw_response=write_raw_response,
        )

    def _build_input_signature(self, input_csv_path: Path) -> Dict[str, Any]:
        stat = input_csv_path.stat()
        mtime_ns = getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))
        return {
            "path": str(input_csv_path.resolve()),
            "size": int(stat.st_size),
            "mtime_ns": int(mtime_ns),
        }

    def _build_checkpoint_payload(
        self,
        *,
        context: ExtractionV2Context,
        processed_rows: int,
        ok_rows: int,
        error_rows: int,
        last_source_row_number: Optional[int],
        created_at: str,
    ) -> Dict[str, Any]:
        return {
            "version": 1,
            "input_signature": context.input_signature,
            "schema_signature": context.schema_signature,
            "options_hash": context.options_hash,
            "columns": {
                "id_column": context.id_column,
                "report_column": context.report_column,
                "feature_columns": context.feature_columns,
                "output_fieldnames": context.output_fieldnames,
            },
            "write_raw_response": context.write_raw_response,
            "processed_rows": processed_rows,
            "ok_rows": ok_rows,
            "error_rows": error_rows,
            "last_source_row_number": last_source_row_number,
            "created_at": created_at,
            "updated_at": now_utc_iso(),
        }

    def _read_checkpoint(self, state_path: Path) -> Dict[str, Any]:
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ExtractionV2ConflictError(
                code="v2_resume_conflict",
                message=f"Invalid checkpoint JSON at {state_path}: {exc}",
                reason="invalid_checkpoint_json",
                remediation="Use overwrite_output=true to start fresh or choose a different output_name.",
            ) from exc
        if not isinstance(payload, dict):
            raise ExtractionV2ConflictError(
                code="v2_resume_conflict",
                message=f"Invalid checkpoint payload at {state_path}.",
                reason="invalid_checkpoint_payload",
                remediation="Use overwrite_output=true to start fresh or choose a different output_name.",
            )
        return payload

    def _write_checkpoint(self, state_path: Path, payload: Dict[str, Any]) -> None:
        tmp_path = state_path.with_suffix(f"{state_path.suffix}.tmp")
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp_path, state_path)

    def _build_state_path(self, output_csv_path: Path) -> Path:
        return Path(f"{output_csv_path}.state.json")

    def _read_csv_header(self, csv_path: Path) -> List[str]:
        if not csv_path.exists():
            return []
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
        return list(header or [])

    def _as_non_negative_int(self, value: Any) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return 0
        return max(parsed, 0)

    def _stable_hash(self, payload: Any) -> str:
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def get_job(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise ExtractionV2JobNotFoundError(job_id)
            record = self._jobs[job_id]

            now = record.finished_at or time.time()
            elapsed_seconds = 0.0
            if record.started_at:
                elapsed_seconds = max(0.0, now - record.started_at)
            processed_in_run = max(0, record.processed_rows - record.processed_rows_at_start)
            rows_per_minute = 0.0
            if elapsed_seconds > 0:
                rows_per_minute = (processed_in_run / elapsed_seconds) * 60.0

            return {
                "job_id": record.job_id,
                "status": record.status,
                "processed_rows": record.processed_rows,
                "ok_rows": record.ok_rows,
                "error_rows": record.error_rows,
                "elapsed_seconds": round(elapsed_seconds, 3),
                "rows_per_minute": round(rows_per_minute, 3),
                "last_source_row_number": record.last_source_row_number,
                "cancel_requested": record.cancel_requested,
                "output_csv_path": record.output_csv_path,
                "error": record.error,
                "total_rows": record.total_rows,
                "progress_percent": record.progress_percent,
                "id_column": record.id_column,
                "report_column": record.report_column,
                "resume_mode": record.resume_mode,
                "processed_rows_at_start": record.processed_rows_at_start,
                "created_at": record.created_at,
                "updated_at": record.updated_at,
                "started_at": record.started_at,
                "finished_at": record.finished_at,
            }

    def get_output_csv_path(self, job_id: str) -> Path:
        with self._lock:
            if job_id not in self._jobs:
                raise ExtractionV2JobNotFoundError(job_id)
            return Path(self._jobs[job_id].output_csv_path)

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise ExtractionV2JobNotFoundError(job_id)
            record = self._jobs[job_id]
            if record.status in {"completed", "failed", "cancelled"}:
                return {
                    "job_id": record.job_id,
                    "status": record.status,
                    "cancel_requested": record.cancel_requested,
                }
            record.cancel_requested = True
            record.updated_at = time.time()
            return {
                "job_id": record.job_id,
                "status": record.status,
                "cancel_requested": True,
            }
