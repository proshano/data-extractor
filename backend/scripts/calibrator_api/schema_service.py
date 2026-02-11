from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set


DEFAULT_MISSING_VALUE = "NA"
STATE_DIR = Path.home() / ".data_prompt_calibrator"
STATE_FILE = STATE_DIR / "calibrator_web_state.json"
DEFAULT_EXPERIMENT_ID = "baseline_experiment"
VALID_REASONING_MODES = {"direct", "plan_then_extract", "react_style", "custom"}


def _default_experiment_profile() -> Dict[str, Any]:
    return {
        "id": DEFAULT_EXPERIMENT_ID,
        "name": "Baseline",
        "system_instructions": "",
        "extraction_instructions": "",
        "reasoning_mode": "direct",
        "reasoning_instructions": "",
        "output_instructions": "",
        "judge": {
            "enabled": False,
            "model": "",
            "instructions": "",
            "acceptance_threshold": 0.6,
        },
    }


class SchemaServiceError(ValueError):
    """Raised for invalid schema or session data."""


def _normalize_judge_config(raw_judge: Any) -> Dict[str, Any]:
    if not isinstance(raw_judge, dict):
        raw_judge = {}
    try:
        threshold = float(raw_judge.get("acceptance_threshold", 0.6))
    except (TypeError, ValueError):
        threshold = 0.6
    threshold = max(0.0, min(1.0, threshold))
    return {
        "enabled": bool(raw_judge.get("enabled", False)),
        "model": str(raw_judge.get("model", "")).strip(),
        "instructions": str(raw_judge.get("instructions", "")).strip(),
        "acceptance_threshold": threshold,
    }


def _normalize_experiment_profile(raw_profile: Any, index: int) -> Dict[str, Any]:
    fallback = _default_experiment_profile()
    if not isinstance(raw_profile, dict):
        raw_profile = {}

    profile_id = str(raw_profile.get("id", "")).strip() or f"experiment_{index + 1}"
    reasoning_mode = str(raw_profile.get("reasoning_mode", "direct")).strip()
    if reasoning_mode not in VALID_REASONING_MODES:
        reasoning_mode = "direct"

    return {
        "id": profile_id,
        "name": str(raw_profile.get("name", "")).strip() or f"Experiment {index + 1}",
        "system_instructions": str(raw_profile.get("system_instructions", "")).strip(),
        "extraction_instructions": str(raw_profile.get("extraction_instructions", "")).strip(),
        "reasoning_mode": reasoning_mode,
        "reasoning_instructions": str(raw_profile.get("reasoning_instructions", "")).strip(),
        "output_instructions": str(raw_profile.get("output_instructions", "")).strip(),
        "judge": _normalize_judge_config(raw_profile.get("judge", fallback["judge"])),
    }


def _normalize_experiment_profiles(raw_profiles: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_profiles, list) or not raw_profiles:
        return [_default_experiment_profile()]

    normalized: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()
    for index, raw_profile in enumerate(raw_profiles):
        profile = _normalize_experiment_profile(raw_profile, index)
        if profile["id"] in seen_ids:
            continue
        seen_ids.add(profile["id"])
        normalized.append(profile)

    if not normalized:
        return [_default_experiment_profile()]
    return normalized


def _normalize_active_experiment_ids(raw_ids: Any, known_ids: Set[str]) -> List[str]:
    if not isinstance(raw_ids, list):
        return []

    active_ids: List[str] = []
    seen_ids: Set[str] = set()
    for raw_id in raw_ids:
        experiment_id = str(raw_id).strip()
        if not experiment_id or experiment_id not in known_ids or experiment_id in seen_ids:
            continue
        seen_ids.add(experiment_id)
        active_ids.append(experiment_id)
    return active_ids


def _normalize_csv_cache(raw_cache: Any) -> Any:
    if not isinstance(raw_cache, dict):
        return None

    columns = raw_cache.get("columns")
    preview = raw_cache.get("preview")
    if not isinstance(columns, list) or not isinstance(preview, list):
        return None

    safe_preview: List[Dict[str, Any]] = []
    for row in preview:
        if not isinstance(row, dict):
            continue
        row_number = row.get("row_number")
        values = row.get("values")
        if not isinstance(row_number, int) or row_number < 1:
            continue
        if not isinstance(values, dict):
            values = {}
        safe_preview.append(
            {
                "row_number": row_number,
                "values": {str(key): str(value) for key, value in values.items()},
            }
        )

    return {
        "source": str(raw_cache.get("source", "")),
        "columns": [str(column) for column in columns],
        "row_count": int(raw_cache.get("row_count", 0)),
        "preview": safe_preview,
        "encoding": str(raw_cache.get("encoding", "")),
        "delimiter": str(raw_cache.get("delimiter", ",")),
        "inferred_id_column": str(raw_cache.get("inferred_id_column", "")),
        "inferred_report_column": str(raw_cache.get("inferred_report_column", "")),
    }


def _normalize_feature(feature: Dict[str, Any], index: int, names_seen: Set[str]) -> Dict[str, Any]:
    name = str(feature.get("name", "")).strip()
    description = str(feature.get("description", "")).strip()
    missing_rule = str(feature.get("missing_value_rule", DEFAULT_MISSING_VALUE)).strip()
    prompt = str(feature.get("prompt", "")).strip()

    if not name:
        raise SchemaServiceError(f"Schema feature #{index} is missing 'name'.")
    if name in names_seen:
        raise SchemaServiceError(f"Duplicate feature name in schema: {name}")
    if not description:
        raise SchemaServiceError(f"Schema feature '{name}' is missing 'description'.")
    if not missing_rule:
        raise SchemaServiceError(f"Schema feature '{name}' has empty 'missing_value_rule'.")

    has_allowed_values = "allowed_values" in feature and feature.get("allowed_values") is not None
    has_type_hint = "type_hint" in feature and str(feature.get("type_hint", "")).strip()
    if not has_allowed_values and not has_type_hint:
        raise SchemaServiceError(
            f"Schema feature '{name}' must include either 'allowed_values' or 'type_hint'."
        )

    normalized: Dict[str, Any] = {
        "name": name,
        "description": description,
        "missing_value_rule": missing_rule,
        "prompt": prompt,
    }

    if has_allowed_values:
        raw_allowed = feature.get("allowed_values")
        if not isinstance(raw_allowed, list):
            raise SchemaServiceError(f"Schema feature '{name}' has non-list allowed_values.")
        allowed = [str(item).strip() for item in raw_allowed if str(item).strip()]
        if not allowed:
            raise SchemaServiceError(f"Schema feature '{name}' has empty allowed_values.")
        normalized["allowed_values"] = allowed
    else:
        normalized["type_hint"] = str(feature.get("type_hint", "string")).strip() or "string"

    return normalized


def validate_features(features: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(features, (list, tuple)) or not features:
        raise SchemaServiceError("Schema must include a non-empty 'features' list.")

    normalized: List[Dict[str, Any]] = []
    names_seen: Set[str] = set()
    for index, item in enumerate(features):
        if not isinstance(item, dict):
            raise SchemaServiceError(f"Schema feature #{index} must be an object.")
        normalized.append(_normalize_feature(item, index=index, names_seen=names_seen))
        names_seen.add(normalized[-1]["name"])

    return normalized


def load_schema(schema_path: Path) -> Dict[str, Any]:
    if not schema_path.exists():
        raise SchemaServiceError(f"Schema file not found: {schema_path}")

    try:
        payload = json.loads(schema_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SchemaServiceError(f"Invalid schema JSON in {schema_path}: {exc}") from exc

    features = validate_features(payload.get("features", []))
    schema_name = str(payload.get("schema_name", "data_extraction_calibrated")).strip() or "data_extraction_calibrated"
    missing_default = str(payload.get("missing_default", DEFAULT_MISSING_VALUE)).strip() or DEFAULT_MISSING_VALUE

    return {
        "schema_path": str(schema_path),
        "schema_name": schema_name,
        "missing_default": missing_default,
        "features": features,
    }


def save_schema(
    schema_path: Path,
    *,
    features: Sequence[Dict[str, Any]],
    schema_name: str = "data_extraction_calibrated",
    missing_default: str = DEFAULT_MISSING_VALUE,
) -> Dict[str, Any]:
    normalized_features = validate_features(features)
    payload = {
        "schema_name": schema_name.strip() or "data_extraction_calibrated",
        "missing_default": missing_default.strip() or DEFAULT_MISSING_VALUE,
        "features": normalized_features,
    }

    schema_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = schema_path.with_suffix(f"{schema_path.suffix}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(schema_path)

    return {
        "schema_path": str(schema_path),
        "schema_name": payload["schema_name"],
        "missing_default": payload["missing_default"],
        "features": normalized_features,
    }


def load_session_state() -> Dict[str, Any]:
    if not STATE_FILE.exists():
        default_experiments = [_default_experiment_profile()]
        return {
            "version": 1,
            "csv_path": "",
            "csv_cache": None,
            "schema_path": "",
            "llama_url": "http://127.0.0.1:8080",
            "temperature": 0.0,
            "max_retries": 5,
            "id_column": "",
            "report_column": "",
            "sample_index": 1,
            "server_model": "",
            "features": [],
            "experiment_profiles": default_experiments,
            "active_experiment_id": default_experiments[0]["id"],
            "active_experiment_ids": [default_experiments[0]["id"]],
        }

    try:
        payload = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SchemaServiceError(f"Invalid session state JSON in {STATE_FILE}: {exc}") from exc

    features = payload.get("features", [])
    safe_features: List[Dict[str, Any]] = []
    if isinstance(features, list):
        for feature in features:
            if isinstance(feature, dict):
                safe_features.append(feature)

    experiment_profiles = _normalize_experiment_profiles(payload.get("experiment_profiles", []))
    raw_active_experiment_id = str(payload.get("active_experiment_id", "")).strip()
    known_ids = {profile["id"] for profile in experiment_profiles}
    active_experiment_id = raw_active_experiment_id if raw_active_experiment_id in known_ids else experiment_profiles[0]["id"]
    active_experiment_ids = _normalize_active_experiment_ids(payload.get("active_experiment_ids"), known_ids)
    if not active_experiment_ids:
        active_experiment_ids = [active_experiment_id]
    csv_cache = _normalize_csv_cache(payload.get("csv_cache"))

    return {
        "version": 1,
        "csv_path": str(payload.get("csv_path", "")),
        "csv_cache": csv_cache,
        "schema_path": str(payload.get("schema_path", "")),
        "llama_url": str(payload.get("llama_url", "http://127.0.0.1:8080")),
        "temperature": float(payload.get("temperature", 0.0)),
        "max_retries": int(payload.get("max_retries", 5)),
        "id_column": str(payload.get("id_column", "")),
        "report_column": str(payload.get("report_column", "")),
        "sample_index": int(payload.get("sample_index", 1)),
        "server_model": str(payload.get("server_model", "")),
        "features": safe_features,
        "experiment_profiles": experiment_profiles,
        "active_experiment_id": active_experiment_id,
        "active_experiment_ids": active_experiment_ids,
    }


def save_session_state(state: Dict[str, Any]) -> Dict[str, Any]:
    experiment_profiles = _normalize_experiment_profiles(state.get("experiment_profiles", []))
    raw_active_experiment_id = str(state.get("active_experiment_id", "")).strip()
    known_ids = {profile["id"] for profile in experiment_profiles}
    active_experiment_id = raw_active_experiment_id if raw_active_experiment_id in known_ids else experiment_profiles[0]["id"]
    active_experiment_ids = _normalize_active_experiment_ids(state.get("active_experiment_ids"), known_ids)
    if not active_experiment_ids:
        active_experiment_ids = [active_experiment_id]

    payload = {
        "version": 1,
        "csv_path": str(state.get("csv_path", "")),
        "csv_cache": _normalize_csv_cache(state.get("csv_cache")),
        "schema_path": str(state.get("schema_path", "")),
        "llama_url": str(state.get("llama_url", "http://127.0.0.1:8080")),
        "temperature": float(state.get("temperature", 0.0)),
        "max_retries": int(state.get("max_retries", 5)),
        "id_column": str(state.get("id_column", "")),
        "report_column": str(state.get("report_column", "")),
        "sample_index": int(state.get("sample_index", 1)),
        "server_model": str(state.get("server_model", "")),
        "features": [feature for feature in state.get("features", []) if isinstance(feature, dict)],
        "experiment_profiles": experiment_profiles,
        "active_experiment_id": active_experiment_id,
        "active_experiment_ids": active_experiment_ids,
    }

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = STATE_FILE.with_suffix(".tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(STATE_FILE)
    return payload
