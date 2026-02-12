from __future__ import annotations

import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .calibration_service import CalibrationOptions, run_feature_test
from .csv_service import (
    CsvServiceError,
    read_csv_headers,
    parse_csv_base64,
    parse_csv_file,
    read_csv_rows_from_base64,
    read_csv_rows_from_file,
)
from .extraction_jobs import (
    ExtractionJobNotFoundError,
    ExtractionJobStore,
    ExtractionJobStoreV2,
    ExtractionV2ConflictError,
    ExtractionV2JobNotFoundError,
)
from .hf_service import (
    HuggingFaceServiceError,
    download_gguf_model,
    list_gguf_files,
    search_gguf_models,
)
from .jobs import CalibrationJobStore, JobNotFoundError
from .llama_service import LlamaServiceError, fetch_server_models, llama_server_manager
from .schema_service import (
    SchemaServiceError,
    load_schema,
    load_session_state,
    save_schema,
    save_session_state,
    validate_features,
)


class ApiError(Exception):
    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}


class CsvLoadRequest(BaseModel):
    path: Optional[str] = None
    file_name: Optional[str] = None
    file_base64: Optional[str] = None
    preview_rows: int = Field(default=8, ge=1, le=100)
    id_column: str = ""
    report_column: str = ""


class SchemaLoadRequest(BaseModel):
    path: str


class SchemaSaveRequest(BaseModel):
    path: str
    features: List[Dict[str, Any]]
    schema_name: str = "data_extraction_calibrated"
    missing_default: str = "NA"


class SessionSaveRequest(BaseModel):
    state: Dict[str, Any]


class JudgeConfigRequest(BaseModel):
    enabled: bool = False
    model: str = ""
    instructions: str = ""
    acceptance_threshold: float = Field(default=0.6, ge=0.0, le=1.0)


class FeatureTestRequest(BaseModel):
    feature: Dict[str, Any]
    report_text: str
    llama_url: str = "http://127.0.0.1:8080"
    temperature: float = 0.0
    max_retries: int = Field(default=5, ge=1, le=20)
    model: str = ""
    experiment_id: str = ""
    experiment_name: str = ""
    system_instructions: str = ""
    extraction_instructions: str = ""
    reasoning_mode: str = "direct"
    reasoning_instructions: str = ""
    output_instructions: str = ""
    judge: JudgeConfigRequest = Field(default_factory=JudgeConfigRequest)


class TestAllRequest(BaseModel):
    features: List[Dict[str, Any]]
    report_text: str
    llama_url: str = "http://127.0.0.1:8080"
    temperature: float = 0.0
    max_retries: int = Field(default=5, ge=1, le=20)
    model: str = ""
    experiment_id: str = ""
    experiment_name: str = ""
    system_instructions: str = ""
    extraction_instructions: str = ""
    reasoning_mode: str = "direct"
    reasoning_instructions: str = ""
    output_instructions: str = ""
    judge: JudgeConfigRequest = Field(default_factory=JudgeConfigRequest)


class TestBatchReportRequest(BaseModel):
    row_number: Optional[int] = Field(default=None, ge=1)
    study_id: str = ""
    report_text: str


class ExperimentRunRequest(BaseModel):
    experiment_id: str = ""
    experiment_name: str = ""
    system_instructions: str = ""
    extraction_instructions: str = ""
    reasoning_mode: str = "direct"
    reasoning_instructions: str = ""
    output_instructions: str = ""
    judge: JudgeConfigRequest = Field(default_factory=JudgeConfigRequest)


class TestBatchRequest(BaseModel):
    features: List[Dict[str, Any]]
    reports: List[TestBatchReportRequest] = Field(default_factory=list)
    llama_url: str = "http://127.0.0.1:8080"
    temperature: float = 0.0
    max_retries: int = Field(default=5, ge=1, le=20)
    model: str = ""
    experiment_id: str = ""
    experiment_name: str = ""
    system_instructions: str = ""
    extraction_instructions: str = ""
    reasoning_mode: str = "direct"
    reasoning_instructions: str = ""
    output_instructions: str = ""
    judge: JudgeConfigRequest = Field(default_factory=JudgeConfigRequest)
    experiments: List[ExperimentRunRequest] = Field(default_factory=list)


class ExtractionRunRequest(BaseModel):
    features: List[Dict[str, Any]]
    id_column: str
    report_column: str
    path: str = ""
    file_name: str = ""
    file_base64: str = ""
    output_csv_path: str = ""
    overwrite_output: bool = False
    llama_url: str = "http://127.0.0.1:8080"
    temperature: float = 0.0
    max_retries: int = Field(default=5, ge=1, le=20)
    model: str = ""
    experiment_id: str = ""
    experiment_name: str = ""
    system_instructions: str = ""
    extraction_instructions: str = ""
    reasoning_mode: str = "direct"
    reasoning_instructions: str = ""
    output_instructions: str = ""
    judge: JudgeConfigRequest = Field(default_factory=JudgeConfigRequest)


class ExtractionV2RunRequest(BaseModel):
    features: List[Dict[str, Any]]
    input_csv_path: str
    id_column: str
    report_column: str
    output_name: str = ""
    resume: bool = True
    overwrite_output: bool = False
    write_raw_response: bool = False
    file_name: str = ""
    file_base64: str = ""
    path: str = ""
    llama_url: str = "http://127.0.0.1:8080"
    temperature: float = 0.0
    max_retries: int = Field(default=5, ge=1, le=20)
    model: str = ""
    experiment_id: str = ""
    experiment_name: str = ""
    system_instructions: str = ""
    extraction_instructions: str = ""
    reasoning_mode: str = "direct"
    reasoning_instructions: str = ""
    output_instructions: str = ""
    judge: JudgeConfigRequest = Field(default_factory=JudgeConfigRequest)


class ModelListRequest(BaseModel):
    llama_url: str = "http://127.0.0.1:8080"


class LlamaServerStatusRequest(BaseModel):
    llama_url: str = "http://127.0.0.1:8080"


class LlamaServerStartRequest(BaseModel):
    model_path: str
    port: int = Field(default=8080, ge=1, le=65535)
    ctx_size: int = Field(default=8192, ge=256, le=131072)


class LlamaServerEnsureRequest(BaseModel):
    model_path: str
    port: int = Field(default=8080, ge=1, le=65535)
    ctx_size: int = Field(default=8192, ge=256, le=131072)


class HuggingFaceSearchRequest(BaseModel):
    query: str = ""
    limit: int = Field(default=20, ge=1, le=50)


class HuggingFaceFilesRequest(BaseModel):
    repo_id: str


class HuggingFaceDownloadRequest(BaseModel):
    repo_id: str
    file_name: str
    destination_dir: str = ""
    hf_token: str = ""


def _read_source_value(source: Any, key: str, default: Any = None) -> Any:
    if source is None:
        return default
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


def build_calibration_options(request: Any, source_override: Any = None) -> CalibrationOptions:
    source = source_override if source_override is not None else request

    def read(key: str, default: Any = "") -> Any:
        raw_value = _read_source_value(source, key, None)
        if raw_value is None and source is not request:
            raw_value = _read_source_value(request, key, default)
        if raw_value is None:
            return default
        return raw_value

    judge = read("judge", None)
    return CalibrationOptions(
        llama_url=str(read("llama_url", "http://127.0.0.1:8080")).strip(),
        temperature=float(read("temperature", 0.0)),
        max_retries=int(read("max_retries", 5)),
        model=str(read("model", "")).strip(),
        experiment_id=str(read("experiment_id", "")).strip(),
        experiment_name=str(read("experiment_name", "")).strip(),
        system_instructions=str(read("system_instructions", "")).strip(),
        extraction_instructions=str(read("extraction_instructions", "")).strip(),
        reasoning_mode=str(read("reasoning_mode", "direct") or "direct").strip() or "direct",
        reasoning_instructions=str(read("reasoning_instructions", "")).strip(),
        output_instructions=str(read("output_instructions", "")).strip(),
        judge_enabled=bool(_read_source_value(judge, "enabled", False)),
        judge_model=str(_read_source_value(judge, "model", "") or "").strip(),
        judge_instructions=str(_read_source_value(judge, "instructions", "") or "").strip(),
        judge_acceptance_threshold=float(_read_source_value(judge, "acceptance_threshold", 0.6)),
    )


def resolve_existing_column(headers: List[str], requested_name: str) -> str:
    requested = str(requested_name or "").strip()
    if not requested:
        return ""
    if requested in headers:
        return requested
    lowered = requested.lower()
    for header in headers:
        if header.lower() == lowered:
            return header
    return ""


def timestamp_suffix() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def build_default_output_csv_path(source_name: str) -> Path:
    base_name = Path(str(source_name or "extraction_input.csv")).name
    stem = Path(base_name).stem or "extraction_input"
    exports_dir = Path.home() / ".data_prompt_calibrator" / "exports"
    return exports_dir / f"{stem}_extracted_{timestamp_suffix()}.csv"


def build_unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 1000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise ValueError(f"Unable to generate unique path for: {path}")


DEFAULT_EXPORT_ROOT = Path.home() / ".data_prompt_calibrator" / "exports"
EXPORT_ROOT_ENV = "CALIBRATOR_EXPORT_ROOT"
SAFE_OUTPUT_NAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def get_export_root() -> Path:
    root_value = str(os.getenv(EXPORT_ROOT_ENV, "")).strip()
    if root_value:
        return Path(root_value).expanduser().resolve()
    return DEFAULT_EXPORT_ROOT.expanduser().resolve()


def _sanitize_name_stem(value: str, *, fallback: str) -> str:
    sanitized = SAFE_OUTPUT_NAME_PATTERN.sub("_", str(value or "").strip()).strip("._-")
    if sanitized:
        return sanitized
    return fallback


def resolve_v2_output_csv_path(input_csv_path: Path, output_name: str) -> Path:
    cleaned_name = str(output_name or "").strip()
    if cleaned_name:
        parsed = Path(cleaned_name)
        if parsed.is_absolute():
            raise ValueError("output_name must be a file name, not an absolute path.")
        if len(parsed.parts) != 1:
            raise ValueError("output_name must not include path separators.")
        if parsed.name in {".", ".."}:
            raise ValueError("output_name is invalid.")
        suffix = parsed.suffix.lower()
        stem_source = parsed.stem if suffix == ".csv" else parsed.name
        stem = _sanitize_name_stem(stem_source, fallback="extraction_output")
        file_name = f"{stem}.csv"
    else:
        input_stem = _sanitize_name_stem(input_csv_path.stem or "input", fallback="input")
        file_name = f"{input_stem}_extracted.csv"

    export_root = get_export_root()
    export_root.mkdir(parents=True, exist_ok=True)
    output_path = (export_root / file_name).resolve()
    if not _is_relative_to(output_path, export_root):
        raise ValueError("Resolved output path must stay inside managed export directory.")
    return output_path


job_store = CalibrationJobStore()
extraction_job_store = ExtractionJobStore()
extraction_job_store_v2 = ExtractionJobStoreV2()


@asynccontextmanager
async def app_lifespan(_: FastAPI):
    try:
        yield
    finally:
        try:
            llama_server_manager.stop()
        except Exception:
            pass


app = FastAPI(title="Data Calibrator API", version="0.1.0", lifespan=app_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173", "null"],
    allow_origin_regex=r"^https?://(127\.0\.0\.1|localhost)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ApiError)
async def api_error_handler(_: Any, exc: ApiError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message, "details": exc.details}},
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(_: Any, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "internal_error",
                "message": "Unexpected server error.",
                "details": {"reason": str(exc)},
            }
        },
    )


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "calibrator_api",
        "timestamp_unix": int(time.time()),
    }


@app.post("/api/csv/load")
async def load_csv_endpoint(request: CsvLoadRequest) -> Dict[str, Any]:
    if not request.path and not request.file_base64:
        raise ApiError(
            status_code=400,
            code="csv_missing_source",
            message="Provide either 'path' or 'file_base64'.",
        )

    try:
        if request.path:
            result = parse_csv_file(
                Path(request.path),
                preview_rows=request.preview_rows,
                requested_id_column=request.id_column,
                requested_report_column=request.report_column,
            )
        else:
            result = parse_csv_base64(
                request.file_base64 or "",
                file_name=request.file_name or "uploaded.csv",
                preview_rows=request.preview_rows,
                requested_id_column=request.id_column,
                requested_report_column=request.report_column,
            )
        return result
    except CsvServiceError as exc:
        raise ApiError(status_code=400, code="csv_load_failed", message=str(exc)) from exc


@app.post("/api/schema/load")
async def load_schema_endpoint(request: SchemaLoadRequest) -> Dict[str, Any]:
    try:
        return load_schema(Path(request.path))
    except SchemaServiceError as exc:
        raise ApiError(status_code=400, code="schema_load_failed", message=str(exc)) from exc


@app.post("/api/schema/save")
async def save_schema_endpoint(request: SchemaSaveRequest) -> Dict[str, Any]:
    try:
        return save_schema(
            Path(request.path),
            features=request.features,
            schema_name=request.schema_name,
            missing_default=request.missing_default,
        )
    except SchemaServiceError as exc:
        raise ApiError(status_code=400, code="schema_save_failed", message=str(exc)) from exc


@app.get("/api/session/load")
async def load_session_endpoint() -> Dict[str, Any]:
    try:
        return {"state": load_session_state()}
    except SchemaServiceError as exc:
        raise ApiError(status_code=400, code="session_load_failed", message=str(exc)) from exc


@app.post("/api/session/save")
async def save_session_endpoint(request: SessionSaveRequest) -> Dict[str, Any]:
    try:
        saved = save_session_state(request.state)
        return {"state": saved}
    except (SchemaServiceError, ValueError, TypeError) as exc:
        raise ApiError(status_code=400, code="session_save_failed", message=str(exc)) from exc


@app.post("/api/models/list")
async def list_models_endpoint(request: ModelListRequest) -> Dict[str, Any]:
    try:
        models = fetch_server_models(request.llama_url)
        return {"llama_url": request.llama_url.strip().rstrip("/"), "models": models}
    except LlamaServiceError as exc:
        raise ApiError(status_code=400, code="model_list_failed", message=str(exc)) from exc


@app.post("/api/hf/gguf/search")
async def hf_search_gguf_endpoint(request: HuggingFaceSearchRequest) -> Dict[str, Any]:
    try:
        models = search_gguf_models(query=request.query, limit=request.limit)
        return {"models": models}
    except HuggingFaceServiceError as exc:
        raise ApiError(status_code=400, code="hf_search_failed", message=str(exc)) from exc


@app.post("/api/hf/gguf/files")
async def hf_list_gguf_files_endpoint(request: HuggingFaceFilesRequest) -> Dict[str, Any]:
    try:
        files = list_gguf_files(repo_id=request.repo_id)
        return {"repo_id": request.repo_id.strip(), "files": files}
    except HuggingFaceServiceError as exc:
        raise ApiError(status_code=400, code="hf_files_failed", message=str(exc)) from exc


@app.post("/api/hf/gguf/download")
async def hf_download_gguf_endpoint(request: HuggingFaceDownloadRequest) -> Dict[str, Any]:
    try:
        return download_gguf_model(
            repo_id=request.repo_id,
            file_name=request.file_name,
            destination_dir=request.destination_dir,
            hf_token=request.hf_token,
        )
    except HuggingFaceServiceError as exc:
        raise ApiError(status_code=400, code="hf_download_failed", message=str(exc)) from exc


@app.get("/api/llama/local-models")
async def list_local_models_endpoint() -> Dict[str, Any]:
    models, timed_out = llama_server_manager.list_local_models(include_slow_dirs=False)
    return {
        "models": models,
        "timed_out": timed_out,
        "binary_path": llama_server_manager.get_binary_path(),
    }


@app.post("/api/llama/server/status")
async def llama_server_status_endpoint(request: LlamaServerStatusRequest) -> Dict[str, Any]:
    try:
        return llama_server_manager.status(llama_url=request.llama_url)
    except LlamaServiceError as exc:
        raise ApiError(status_code=400, code="llama_status_failed", message=str(exc)) from exc


@app.post("/api/llama/server/start")
async def llama_server_start_endpoint(request: LlamaServerStartRequest) -> Dict[str, Any]:
    try:
        started = llama_server_manager.start(
            model_path=request.model_path,
            port=request.port,
            ctx_size=request.ctx_size,
        )
        status = llama_server_manager.status(llama_url=f"http://127.0.0.1:{request.port}")
        status.update(started)
        return status
    except LlamaServiceError as exc:
        raise ApiError(status_code=400, code="llama_start_failed", message=str(exc)) from exc


@app.post("/api/llama/server/ensure")
async def llama_server_ensure_endpoint(request: LlamaServerEnsureRequest) -> Dict[str, Any]:
    try:
        ensured = llama_server_manager.ensure_running(
            model_path=request.model_path,
            port=request.port,
            ctx_size=request.ctx_size,
        )
        status = llama_server_manager.status(llama_url=f"http://127.0.0.1:{request.port}")
        status.update(ensured)
        return status
    except LlamaServiceError as exc:
        raise ApiError(status_code=400, code="llama_ensure_failed", message=str(exc)) from exc


@app.post("/api/llama/server/stop")
async def llama_server_stop_endpoint(request: LlamaServerStatusRequest) -> Dict[str, Any]:
    llama_server_manager.stop()
    try:
        return llama_server_manager.status(llama_url=request.llama_url)
    except LlamaServiceError as exc:
        raise ApiError(status_code=400, code="llama_stop_failed", message=str(exc)) from exc


@app.post("/api/llama/server/stop-now")
async def llama_server_stop_now_endpoint() -> Dict[str, Any]:
    llama_server_manager.stop()
    return {"stopped": True}


@app.post("/api/test/feature")
async def test_feature_endpoint(request: FeatureTestRequest) -> Dict[str, Any]:
    try:
        validated_feature = validate_features([request.feature])[0]
    except SchemaServiceError as exc:
        raise ApiError(status_code=400, code="invalid_feature", message=str(exc)) from exc

    options = build_calibration_options(request)

    result = run_feature_test(
        feature=validated_feature,
        report_text=request.report_text,
        options=options,
    )
    return result


@app.post("/api/test/all")
async def test_all_endpoint(request: TestAllRequest) -> Dict[str, Any]:
    try:
        validated_features = validate_features(request.features)
    except SchemaServiceError as exc:
        raise ApiError(status_code=400, code="invalid_features", message=str(exc)) from exc

    if not request.report_text.strip():
        raise ApiError(
            status_code=400,
            code="empty_report_text",
            message="report_text must be non-empty.",
        )

    options = build_calibration_options(request)

    job_id = job_store.create_test_all_job(
        features=validated_features,
        report_text=request.report_text,
        options=options,
    )
    return {"job_id": job_id}


@app.post("/api/test/batch")
async def test_batch_endpoint(request: TestBatchRequest) -> Dict[str, Any]:
    try:
        validated_features = validate_features(request.features)
    except SchemaServiceError as exc:
        raise ApiError(status_code=400, code="invalid_features", message=str(exc)) from exc

    if len(request.reports) == 0:
        raise ApiError(
            status_code=400,
            code="empty_reports",
            message="At least one report is required.",
        )
    if len(request.reports) > 20:
        raise ApiError(
            status_code=400,
            code="too_many_reports",
            message="You can test at most 20 reports at a time.",
        )

    report_items: List[Dict[str, Any]] = []
    for report in request.reports:
        report_text = str(report.report_text or "").strip()
        if not report_text:
            continue
        report_items.append(
            {
                "row_number": report.row_number,
                "study_id": str(report.study_id or "").strip(),
                "report_text": report_text,
            }
        )

    if not report_items:
        raise ApiError(
            status_code=400,
            code="empty_reports",
            message="At least one report_text must be non-empty.",
        )

    if request.experiments:
        options_list = [build_calibration_options(request, source_override=experiment) for experiment in request.experiments]
    else:
        options_list = [build_calibration_options(request)]

    try:
        job_id = job_store.create_test_batch_job(
            features=validated_features,
            reports=report_items,
            options_list=options_list,
        )
    except ValueError as exc:
        raise ApiError(status_code=400, code="invalid_reports", message=str(exc)) from exc

    return {"job_id": job_id}


@app.post("/api/extract/run")
async def run_extraction_endpoint(request: ExtractionRunRequest) -> Dict[str, Any]:
    try:
        validated_features = validate_features(request.features)
    except SchemaServiceError as exc:
        raise ApiError(status_code=400, code="invalid_features", message=str(exc)) from exc

    requested_id_column = str(request.id_column or "").strip()
    if not requested_id_column:
        raise ApiError(
            status_code=400,
            code="missing_id_column",
            message="id_column is required for extraction.",
        )

    requested_report_column = str(request.report_column or "").strip()
    if not requested_report_column:
        raise ApiError(
            status_code=400,
            code="missing_report_column",
            message="report_column is required for extraction.",
        )

    has_path = bool(str(request.path or "").strip())
    has_upload = bool(str(request.file_base64 or "").strip())
    if not has_path and not has_upload:
        raise ApiError(
            status_code=400,
            code="csv_missing_source",
            message="Provide either 'path' or 'file_base64' for extraction.",
        )

    try:
        if has_path:
            csv_source = read_csv_rows_from_file(Path(str(request.path).strip()).expanduser())
        else:
            csv_source = read_csv_rows_from_base64(
                str(request.file_base64 or ""),
                file_name=str(request.file_name or "uploaded.csv"),
            )
    except CsvServiceError as exc:
        raise ApiError(status_code=400, code="csv_load_failed", message=str(exc)) from exc

    id_column = resolve_existing_column(csv_source["columns"], requested_id_column)
    if not id_column:
        raise ApiError(
            status_code=400,
            code="invalid_id_column",
            message=f"ID column '{requested_id_column}' was not found in CSV headers.",
            details={"columns": csv_source["columns"]},
        )

    report_column = resolve_existing_column(csv_source["columns"], requested_report_column)
    if not report_column:
        raise ApiError(
            status_code=400,
            code="invalid_report_column",
            message=f"Report column '{requested_report_column}' was not found in CSV headers.",
            details={"columns": csv_source["columns"]},
        )

    requested_output_csv_path = str(request.output_csv_path or "").strip()
    output_csv_path = (
        Path(requested_output_csv_path).expanduser()
        if requested_output_csv_path
        else build_default_output_csv_path(csv_source["source"])
    )

    if requested_output_csv_path and output_csv_path.exists() and not request.overwrite_output:
        raise ApiError(
            status_code=400,
            code="output_exists",
            message=(
                f"Output CSV already exists: {output_csv_path}. "
                "Set overwrite_output=true or choose a different path."
            ),
        )

    if not requested_output_csv_path:
        output_csv_path = build_unique_path(output_csv_path)

    options = build_calibration_options(request)

    try:
        job_id = extraction_job_store.create_job(
            features=validated_features,
            rows=csv_source["rows"],
            id_column=id_column,
            report_column=report_column,
            options=options,
            output_csv_path=output_csv_path,
        )
    except ValueError as exc:
        raise ApiError(status_code=400, code="invalid_extraction_request", message=str(exc)) from exc

    return {"job_id": job_id, "output_csv_path": str(output_csv_path)}


@app.post("/api/extract/v2/run")
async def run_extraction_v2_endpoint(request: ExtractionV2RunRequest) -> Dict[str, Any]:
    try:
        validated_features = validate_features(request.features)
    except SchemaServiceError as exc:
        raise ApiError(status_code=400, code="invalid_features", message=str(exc)) from exc

    requested_id_column = str(request.id_column or "").strip()
    if not requested_id_column:
        raise ApiError(
            status_code=400,
            code="missing_id_column",
            message="id_column is required for extraction.",
        )

    requested_report_column = str(request.report_column or "").strip()
    if not requested_report_column:
        raise ApiError(
            status_code=400,
            code="missing_report_column",
            message="report_column is required for extraction.",
        )

    if str(request.file_base64 or "").strip() or str(request.file_name or "").strip() or str(request.path or "").strip():
        raise ApiError(
            status_code=400,
            code="v2_unsupported_input_source",
            message="v2 extraction accepts disk CSV paths only. Use input_csv_path.",
        )

    input_csv_path_value = str(request.input_csv_path or "").strip()
    if not input_csv_path_value:
        raise ApiError(
            status_code=400,
            code="missing_input_csv_path",
            message="input_csv_path is required for v2 extraction.",
        )

    input_csv_path = Path(input_csv_path_value).expanduser()
    try:
        headers_payload = read_csv_headers(input_csv_path)
    except CsvServiceError as exc:
        raise ApiError(status_code=400, code="csv_load_failed", message=str(exc)) from exc

    id_column = resolve_existing_column(headers_payload["columns"], requested_id_column)
    if not id_column:
        raise ApiError(
            status_code=400,
            code="invalid_id_column",
            message=f"ID column '{requested_id_column}' was not found in CSV headers.",
            details={"columns": headers_payload["columns"]},
        )

    report_column = resolve_existing_column(headers_payload["columns"], requested_report_column)
    if not report_column:
        raise ApiError(
            status_code=400,
            code="invalid_report_column",
            message=f"Report column '{requested_report_column}' was not found in CSV headers.",
            details={"columns": headers_payload["columns"]},
        )

    try:
        output_csv_path = resolve_v2_output_csv_path(
            input_csv_path=input_csv_path,
            output_name=str(request.output_name or ""),
        )
    except ValueError as exc:
        raise ApiError(status_code=400, code="invalid_output_name", message=str(exc)) from exc

    options = build_calibration_options(request)

    try:
        job_id, resume_mode, processed_rows_at_start = extraction_job_store_v2.create_job(
            features=validated_features,
            input_csv_path=input_csv_path,
            id_column=id_column,
            report_column=report_column,
            options=options,
            output_csv_path=output_csv_path,
            resume=bool(request.resume),
            overwrite_output=bool(request.overwrite_output),
            write_raw_response=bool(request.write_raw_response),
        )
    except ExtractionV2ConflictError as exc:
        raise ApiError(
            status_code=409,
            code=exc.code,
            message=exc.message,
            details=exc.details,
        ) from exc
    except ValueError as exc:
        raise ApiError(status_code=400, code="invalid_extraction_request", message=str(exc)) from exc

    return {
        "job_id": job_id,
        "output_csv_path": str(output_csv_path),
        "resume_mode": resume_mode,
        "processed_rows_at_start": processed_rows_at_start,
    }


@app.get("/api/extract/v2/jobs/{job_id}")
async def get_extraction_v2_job_endpoint(job_id: str) -> Dict[str, Any]:
    try:
        return extraction_job_store_v2.get_job(job_id)
    except ExtractionV2JobNotFoundError as exc:
        raise ApiError(
            status_code=404,
            code="extraction_v2_job_not_found",
            message=f"Unknown extraction v2 job_id: {job_id}",
        ) from exc


@app.post("/api/extract/v2/jobs/{job_id}/cancel")
async def cancel_extraction_v2_job_endpoint(job_id: str) -> Dict[str, Any]:
    try:
        return extraction_job_store_v2.cancel_job(job_id)
    except ExtractionV2JobNotFoundError as exc:
        raise ApiError(
            status_code=404,
            code="extraction_v2_job_not_found",
            message=f"Unknown extraction v2 job_id: {job_id}",
        ) from exc


@app.get("/api/extract/v2/jobs/{job_id}/download")
async def download_extraction_v2_job_endpoint(job_id: str) -> FileResponse:
    try:
        output_csv_path = extraction_job_store_v2.get_output_csv_path(job_id)
    except ExtractionV2JobNotFoundError as exc:
        raise ApiError(
            status_code=404,
            code="extraction_v2_job_not_found",
            message=f"Unknown extraction v2 job_id: {job_id}",
        ) from exc

    if not output_csv_path.exists():
        raise ApiError(
            status_code=404,
            code="extraction_output_not_found",
            message=f"Extraction output CSV not found for job_id: {job_id}",
        )

    return FileResponse(
        path=output_csv_path,
        media_type="text/csv",
        filename=output_csv_path.name,
    )


@app.get("/api/extract/jobs/{job_id}")
async def get_extraction_job_endpoint(job_id: str) -> Dict[str, Any]:
    try:
        return extraction_job_store.get_job(job_id)
    except ExtractionJobNotFoundError as exc:
        raise ApiError(
            status_code=404,
            code="extraction_job_not_found",
            message=f"Unknown extraction job_id: {job_id}",
        ) from exc


@app.post("/api/extract/jobs/{job_id}/cancel")
async def cancel_extraction_job_endpoint(job_id: str) -> Dict[str, Any]:
    try:
        return extraction_job_store.cancel_job(job_id)
    except ExtractionJobNotFoundError as exc:
        raise ApiError(
            status_code=404,
            code="extraction_job_not_found",
            message=f"Unknown extraction job_id: {job_id}",
        ) from exc


@app.get("/api/extract/jobs/{job_id}/download")
async def download_extraction_job_endpoint(job_id: str) -> FileResponse:
    try:
        output_csv_path = extraction_job_store.get_output_csv_path(job_id)
    except ExtractionJobNotFoundError as exc:
        raise ApiError(
            status_code=404,
            code="extraction_job_not_found",
            message=f"Unknown extraction job_id: {job_id}",
        ) from exc

    if not output_csv_path.exists():
        raise ApiError(
            status_code=404,
            code="extraction_output_not_found",
            message=f"Extraction output CSV not found for job_id: {job_id}",
        )

    return FileResponse(
        path=output_csv_path,
        media_type="text/csv",
        filename=output_csv_path.name,
    )


@app.get("/api/jobs/{job_id}")
async def get_job_endpoint(job_id: str) -> Dict[str, Any]:
    try:
        return job_store.get_job(job_id)
    except JobNotFoundError as exc:
        raise ApiError(
            status_code=404,
            code="job_not_found",
            message=f"Unknown job_id: {job_id}",
        ) from exc


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job_endpoint(job_id: str) -> Dict[str, Any]:
    try:
        return job_store.cancel_job(job_id)
    except JobNotFoundError as exc:
        raise ApiError(
            status_code=404,
            code="job_not_found",
            message=f"Unknown job_id: {job_id}",
        ) from exc
