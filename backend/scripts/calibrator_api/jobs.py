from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .calibration_service import CalibrationOptions, run_all_feature_tests


@dataclass
class JobRecord:
    job_id: str
    status: str
    total: int
    reports_total: int = 1
    reports_completed: int = 0
    active_report_index: Optional[int] = None
    completed: int = 0
    progress_percent: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    results: List[Dict[str, Any]] = field(default_factory=list)
    report_results: List[Dict[str, Any]] = field(default_factory=list)
    last_result: Optional[Dict[str, Any]] = None
    error: str = ""
    cancel_requested: bool = False


class JobNotFoundError(KeyError):
    """Raised when a job id is unknown."""


class CalibrationJobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def create_test_all_job(
        self,
        *,
        features: Sequence[Dict[str, Any]],
        report_text: str,
        options: CalibrationOptions,
    ) -> str:
        job_id = uuid.uuid4().hex
        record = JobRecord(job_id=job_id, status="pending", total=len(features), reports_total=1)

        with self._lock:
            self._jobs[job_id] = record

        thread = threading.Thread(
            target=self._run_job,
            args=(job_id, list(features), report_text, options),
            daemon=True,
        )
        thread.start()
        return job_id

    def create_test_batch_job(
        self,
        *,
        features: Sequence[Dict[str, Any]],
        reports: Sequence[Dict[str, Any]],
        options_list: Sequence[CalibrationOptions],
    ) -> str:
        clean_reports: List[Dict[str, Any]] = []
        for item in reports:
            report_text = str(item.get("report_text", "")).strip()
            if not report_text:
                continue

            row_number_raw = item.get("row_number")
            row_number: Optional[int] = None
            if row_number_raw is not None:
                try:
                    parsed = int(row_number_raw)
                except (TypeError, ValueError):
                    parsed = 0
                if parsed > 0:
                    row_number = parsed

            clean_reports.append(
                {
                    "report_text": report_text,
                    "row_number": row_number,
                }
            )

        if not clean_reports:
            raise ValueError("At least one non-empty report is required.")

        normalized_options = [option for option in options_list if option is not None]
        if not normalized_options:
            raise ValueError("At least one experiment setup is required.")

        job_id = uuid.uuid4().hex
        total_items = len(features) * len(clean_reports) * len(normalized_options)
        record = JobRecord(
            job_id=job_id,
            status="pending",
            total=total_items,
            reports_total=len(clean_reports) * len(normalized_options),
        )

        with self._lock:
            self._jobs[job_id] = record

        thread = threading.Thread(
            target=self._run_batch_job,
            args=(job_id, list(features), clean_reports, normalized_options),
            daemon=True,
        )
        thread.start()
        return job_id

    def _run_job(
        self,
        job_id: str,
        features: List[Dict[str, Any]],
        report_text: str,
        options: CalibrationOptions,
    ) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.status = "running"
            record.started_at = time.time()
            record.active_report_index = 1
            record.updated_at = time.time()

        def should_cancel() -> bool:
            with self._lock:
                current = self._jobs[job_id]
                return current.cancel_requested

        def progress_callback(completed: int, total: int, result: Optional[Dict[str, Any]]) -> None:
            result_payload: Optional[Dict[str, Any]] = None
            if result is not None:
                result_payload = dict(result)
                result_payload["report_index"] = 1
                result_payload["row_number"] = None

            with self._lock:
                current = self._jobs[job_id]
                current.completed = completed
                current.progress_percent = int((completed / max(total, 1)) * 100)
                if result_payload is not None:
                    current.last_result = result_payload
                    current.results.append(result_payload)
                current.updated_at = time.time()

        try:
            run_all_feature_tests(
                features=features,
                report_text=report_text,
                options=options,
                progress_callback=progress_callback,
                should_cancel=should_cancel,
            )

            with self._lock:
                current = self._jobs[job_id]
                current.finished_at = time.time()
                current.updated_at = time.time()
                current.reports_completed = 1
                current.active_report_index = None
                current.report_results = [
                    {
                        "run_index": 1,
                        "report_index": 1,
                        "row_number": None,
                        "report_text": report_text,
                        "experiment_id": str(options.experiment_id or "").strip(),
                        "experiment_name": str(options.experiment_name or "").strip(),
                        "results": list(current.results),
                    }
                ]
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
                current.active_report_index = None
                current.updated_at = time.time()

    def _run_batch_job(
        self,
        job_id: str,
        features: List[Dict[str, Any]],
        reports: List[Dict[str, Any]],
        options_list: List[CalibrationOptions],
    ) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.status = "running"
            record.started_at = time.time()
            record.active_report_index = 1 if reports and options_list else None
            record.updated_at = time.time()

        completed_count = 0
        total_count = max(len(features) * len(reports) * len(options_list), 1)
        run_index = 0

        def should_cancel() -> bool:
            with self._lock:
                current = self._jobs[job_id]
                return current.cancel_requested

        try:
            for report_index, report_item in enumerate(reports, start=1):
                if should_cancel():
                    break

                report_text = str(report_item.get("report_text", "") or "")
                row_number = report_item.get("row_number")
                for options in options_list:
                    if should_cancel():
                        break

                    run_index += 1
                    report_feature_results: List[Dict[str, Any]] = []

                    with self._lock:
                        current = self._jobs[job_id]
                        current.active_report_index = run_index
                        current.updated_at = time.time()

                    def progress_callback(
                        _completed_for_report: int, _total_for_report: int, result: Optional[Dict[str, Any]]
                    ) -> None:
                        nonlocal completed_count
                        if result is None:
                            return

                        completed_count += 1
                        result_payload = dict(result)
                        result_payload["run_index"] = run_index
                        result_payload["report_index"] = report_index
                        result_payload["row_number"] = row_number
                        result_payload["experiment_id"] = (
                            str(result_payload.get("experiment_id", "")).strip()
                            or str(options.experiment_id or "").strip()
                        )
                        result_payload["experiment_name"] = (
                            str(result_payload.get("experiment_name", "")).strip()
                            or str(options.experiment_name or "").strip()
                        )
                        report_feature_results.append(result_payload)

                        with self._lock:
                            current = self._jobs[job_id]
                            current.completed = completed_count
                            current.progress_percent = int((completed_count / total_count) * 100)
                            current.last_result = result_payload
                            current.results.append(result_payload)
                            current.updated_at = time.time()

                    run_all_feature_tests(
                        features=features,
                        report_text=report_text,
                        options=options,
                        progress_callback=progress_callback,
                        should_cancel=should_cancel,
                    )

                    with self._lock:
                        current = self._jobs[job_id]
                        current.report_results.append(
                            {
                                "run_index": run_index,
                                "report_index": report_index,
                                "row_number": row_number,
                                "report_text": report_text,
                                "experiment_id": str(options.experiment_id or "").strip(),
                                "experiment_name": str(options.experiment_name or "").strip(),
                                "results": list(report_feature_results),
                            }
                        )
                        current.reports_completed = len(current.report_results)
                        current.updated_at = time.time()

            with self._lock:
                current = self._jobs[job_id]
                current.finished_at = time.time()
                current.updated_at = time.time()
                current.active_report_index = None
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
                current.active_report_index = None
                current.updated_at = time.time()

    def get_job(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise JobNotFoundError(job_id)
            record = self._jobs[job_id]
            return {
                "job_id": record.job_id,
                "status": record.status,
                "total": record.total,
                "reports_total": record.reports_total,
                "reports_completed": record.reports_completed,
                "active_report_index": record.active_report_index,
                "completed": record.completed,
                "progress_percent": record.progress_percent,
                "created_at": record.created_at,
                "updated_at": record.updated_at,
                "started_at": record.started_at,
                "finished_at": record.finished_at,
                "results": list(record.results),
                "report_results": list(record.report_results),
                "last_result": record.last_result,
                "error": record.error,
                "cancel_requested": record.cancel_requested,
            }

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise JobNotFoundError(job_id)
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
