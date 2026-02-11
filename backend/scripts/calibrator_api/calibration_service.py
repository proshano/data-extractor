from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from .. import extract_pipeline

DEFAULT_SYSTEM_INSTRUCTIONS = (
    "You are a clinical data extraction assistant. Follow the schema exactly and avoid unsupported claims."
)
DEFAULT_EXTRACTION_INSTRUCTIONS = (
    "Use only explicitly documented evidence from the report. "
    "If evidence is absent or ambiguous, return the missing value."
)
DEFAULT_OUTPUT_INSTRUCTIONS = (
    'Return exactly one JSON object with one key: "value". '
    "Do not include markdown, prose, or extra keys."
)
DEFAULT_JUDGE_INSTRUCTIONS = (
    "Judge whether the extracted value is supported by report evidence for the requested feature."
)

REASONING_MODE_INSTRUCTIONS: Dict[str, str] = {
    "direct": (
        "Use concise internal reasoning, then provide only the final structured answer."
    ),
    "plan_then_extract": (
        "Internally map feature criteria to evidence first, then decide the value. "
        "Keep internal planning hidden from output."
    ),
    "react_style": (
        "Use an internal ReAct-style loop: observe evidence, reason about criteria, "
        "verify support, then finalize. Keep internal steps hidden from output."
    ),
}


@dataclass(frozen=True)
class CalibrationOptions:
    llama_url: str
    temperature: float = 0.0
    max_retries: int = 5
    model: str = ""
    experiment_id: str = ""
    experiment_name: str = ""
    system_instructions: str = ""
    extraction_instructions: str = ""
    reasoning_mode: str = "direct"
    reasoning_instructions: str = ""
    output_instructions: str = ""
    judge_enabled: bool = False
    judge_model: str = ""
    judge_instructions: str = ""
    judge_acceptance_threshold: float = 0.6


def normalize_reasoning_mode(value: str) -> str:
    mode = str(value or "").strip()
    if mode in REASONING_MODE_INSTRUCTIONS or mode == "custom":
        return mode
    return "direct"


def resolve_reasoning_instructions(options: CalibrationOptions) -> str:
    custom = str(options.reasoning_instructions or "").strip()
    if custom:
        return custom

    mode = normalize_reasoning_mode(options.reasoning_mode)
    if mode == "custom":
        return REASONING_MODE_INSTRUCTIONS["direct"]
    return REASONING_MODE_INSTRUCTIONS.get(mode, REASONING_MODE_INSTRUCTIONS["direct"])


def build_single_feature_prompt(
    feature: Dict[str, Any],
    report_text: str,
    options: CalibrationOptions,
) -> str:
    name = str(feature.get("name", "")).strip()
    description = str(feature.get("description", "")).strip()
    missing_value = str(feature.get("missing_value_rule", "NA")).strip() or "NA"
    feature_prompt = str(feature.get("prompt", "")).strip()
    if "allowed_values" in feature and feature.get("allowed_values") is not None:
        allowed = json.dumps(feature.get("allowed_values", []), ensure_ascii=True)
        value_rule = f"Allowed values: {allowed}."
    else:
        value_rule = f"Type hint: {json.dumps(feature.get('type_hint', 'string'))}."

    system_instructions = str(options.system_instructions or "").strip() or DEFAULT_SYSTEM_INSTRUCTIONS
    extraction_instructions = (
        str(options.extraction_instructions or "").strip() or DEFAULT_EXTRACTION_INSTRUCTIONS
    )
    output_instructions = str(options.output_instructions or "").strip() or DEFAULT_OUTPUT_INSTRUCTIONS
    reasoning_mode = normalize_reasoning_mode(options.reasoning_mode)
    reasoning_instructions = resolve_reasoning_instructions(options)

    return (
        "Task: extract one structured clinical feature from one report.\n\n"
        "System instructions:\n"
        f"{system_instructions}\n\n"
        "Extraction instructions:\n"
        f"{extraction_instructions}\n\n"
        f"Reasoning mode: {reasoning_mode}\n"
        "Reasoning instructions:\n"
        f"{reasoning_instructions}\n"
        "Keep reasoning internal. Output only the requested JSON object.\n\n"
        "Output instructions:\n"
        f"{output_instructions}\n\n"
        "Feature definition:\n"
        f"Name: {name}\n"
        f"Description: {description}\n"
        f"{value_rule}\n"
        f'Missing value rule: return "{missing_value}" if not documented.\n'
        f"Feature-specific guidance: {feature_prompt or 'None'}\n\n"
        "Report text:\n"
        f"{report_text}\n"
    )


def clamp_score(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return 0.0
    if parsed > 1:
        return 1.0
    return parsed


def normalize_judge_verdict(value: Any) -> str:
    verdict = str(value or "").strip().lower()
    if verdict in {"accepted", "accept", "pass", "supported"}:
        return "accepted"
    if verdict in {"rejected", "reject", "fail", "unsupported"}:
        return "rejected"
    return "uncertain"


def build_judge_prompt(
    feature: Dict[str, Any],
    report_text: str,
    extracted_value: str,
    extraction_prompt: str,
    options: CalibrationOptions,
) -> str:
    name = str(feature.get("name", "")).strip()
    description = str(feature.get("description", "")).strip()
    missing_value = str(feature.get("missing_value_rule", "NA")).strip() or "NA"
    feature_prompt = str(feature.get("prompt", "")).strip()
    if "allowed_values" in feature and feature.get("allowed_values") is not None:
        allowed = json.dumps(feature.get("allowed_values", []), ensure_ascii=True)
        value_rule = f"Allowed values: {allowed}."
    else:
        value_rule = f"Type hint: {json.dumps(feature.get('type_hint', 'string'))}."
    custom_judge_instructions = str(options.judge_instructions or "").strip()
    judge_instructions = custom_judge_instructions or DEFAULT_JUDGE_INSTRUCTIONS
    threshold = clamp_score(options.judge_acceptance_threshold)
    threshold_value = 0.6 if threshold is None else threshold
    system_instructions = str(options.system_instructions or "").strip() or DEFAULT_SYSTEM_INSTRUCTIONS
    extraction_instructions = (
        str(options.extraction_instructions or "").strip() or DEFAULT_EXTRACTION_INSTRUCTIONS
    )
    output_instructions = str(options.output_instructions or "").strip() or DEFAULT_OUTPUT_INSTRUCTIONS
    reasoning_mode = normalize_reasoning_mode(options.reasoning_mode)
    reasoning_instructions = resolve_reasoning_instructions(options)
    prompt_used = str(extraction_prompt or "").strip()
    prompt_without_report = prompt_used
    report_marker = "\n\nReport text:\n"
    if report_marker in prompt_used:
        prompt_without_report = prompt_used.split(report_marker, maxsplit=1)[0].strip()

    return (
        "Task: judge whether an extracted value is supported by the report evidence.\n"
        'Return exactly one JSON object with keys: "verdict", "score", "rationale".\n'
        'Verdict must be one of: "accepted", "rejected", "uncertain".\n'
        "Score must be a number from 0 to 1.\n"
        f"Treat score >= {threshold_value:.2f} as accepted and below that as rejected unless uncertain.\n"
        "Keep rationale short (one to two sentences).\n\n"
        "Judge instructions:\n"
        f"{judge_instructions}\n\n"
        "Extraction run context:\n"
        "System instructions:\n"
        f"{system_instructions}\n\n"
        "Extraction instructions:\n"
        f"{extraction_instructions}\n\n"
        f"Reasoning mode: {reasoning_mode}\n"
        "Reasoning instructions:\n"
        f"{reasoning_instructions}\n\n"
        "Output instructions:\n"
        f"{output_instructions}\n\n"
        "Feature definition:\n"
        f"Name: {name}\n"
        f"Description: {description}\n"
        f"{value_rule}\n"
        f'Missing value: "{missing_value}"\n\n'
        f"Feature-specific guidance: {feature_prompt or 'None'}\n\n"
        "Original extraction prompt sent to the first model (without report body):\n"
        f"{prompt_without_report or 'Unavailable'}\n\n"
        f'Candidate extracted value: "{extracted_value}"\n\n'
        "Report text:\n"
        f"{report_text}\n"
    )


def run_judge_test(
    feature: Dict[str, Any],
    report_text: str,
    extracted_value: str,
    extraction_prompt: str,
    options: CalibrationOptions,
    *,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    started = time.perf_counter()
    if should_cancel and should_cancel():
        return {
            "status": "judge_error",
            "verdict": "",
            "score": None,
            "rationale": "",
            "raw_response": "",
            "error": "Cancelled.",
            "model": "",
            "duration_ms": int((time.perf_counter() - started) * 1000),
        }
    prompt = build_judge_prompt(
        feature=feature,
        report_text=report_text,
        extracted_value=extracted_value,
        extraction_prompt=extraction_prompt,
        options=options,
    )

    judge_model = str(options.judge_model or "").strip() or str(options.model or "").strip()
    judge_response = extract_pipeline.call_llamacpp_with_retries(
        base_url=options.llama_url,
        prompt=prompt,
        temperature=0.0,
        max_retries=int(options.max_retries),
        model=judge_model,
        should_cancel=should_cancel,
    )

    raw_response = str(judge_response.get("content", "") or "")
    response_model = str(judge_response.get("model", "") or "")
    duration_ms = int((time.perf_counter() - started) * 1000)

    if not judge_response.get("success"):
        return {
            "status": "judge_error",
            "verdict": "",
            "score": None,
            "rationale": "",
            "raw_response": raw_response,
            "error": str(judge_response.get("error", "Unknown judge error.")),
            "model": response_model,
            "duration_ms": duration_ms,
        }

    parsed, parse_error = extract_pipeline.parse_llm_json_response(raw_response)
    if parse_error:
        return {
            "status": "judge_error",
            "verdict": "",
            "score": None,
            "rationale": "",
            "raw_response": raw_response,
            "error": parse_error,
            "model": response_model,
            "duration_ms": duration_ms,
        }

    parsed_payload: Dict[str, Any] = parsed if isinstance(parsed, dict) else {}
    verdict = normalize_judge_verdict(parsed_payload.get("verdict", ""))
    score = clamp_score(parsed_payload.get("score"))
    rationale = str(parsed_payload.get("rationale", "")).strip()
    threshold = clamp_score(options.judge_acceptance_threshold)
    threshold_value = 0.6 if threshold is None else threshold

    status = verdict
    if score is not None:
        status = "accepted" if score >= threshold_value else "rejected"
    if verdict == "uncertain":
        status = "uncertain"

    return {
        "status": status,
        "verdict": verdict,
        "score": score,
        "rationale": rationale,
        "raw_response": raw_response,
        "error": "",
        "model": response_model,
        "duration_ms": duration_ms,
    }


def run_feature_test(
    feature: Dict[str, Any],
    report_text: str,
    options: CalibrationOptions,
    *,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    started = time.perf_counter()
    if should_cancel and should_cancel():
        feature_name = str(feature.get("name", "")).strip()
        missing_value = str(feature.get("missing_value_rule", "NA")).strip() or "NA"
        return {
            "feature_name": feature_name,
            "status": "llm_error",
            "value": missing_value,
            "parsed": None,
            "raw_response": "",
            "error": "Cancelled.",
            "model": "",
            "duration_ms": int((time.perf_counter() - started) * 1000),
            "experiment_id": str(options.experiment_id or "").strip(),
            "experiment_name": str(options.experiment_name or "").strip(),
            "judge_result": None,
        }
    prompt = build_single_feature_prompt(feature, report_text, options)

    llm_result = extract_pipeline.call_llamacpp_with_retries(
        base_url=options.llama_url,
        prompt=prompt,
        temperature=float(options.temperature),
        max_retries=int(options.max_retries),
        model=options.model,
        should_cancel=should_cancel,
    )

    raw_response = str(llm_result.get("content", "") or "")
    response_model = str(llm_result.get("model", "") or "")
    feature_name = str(feature.get("name", "")).strip()
    missing_value = str(feature.get("missing_value_rule", "NA")).strip() or "NA"
    experiment_id = str(options.experiment_id or "").strip()
    experiment_name = str(options.experiment_name or "").strip()

    if not llm_result.get("success"):
        duration_ms = int((time.perf_counter() - started) * 1000)
        return {
            "feature_name": feature_name,
            "status": "llm_error",
            "value": missing_value,
            "parsed": None,
            "raw_response": raw_response,
            "error": str(llm_result.get("error", "Unknown LLM error.")),
            "model": response_model,
            "duration_ms": duration_ms,
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "judge_result": None,
        }

    parsed, parse_error = extract_pipeline.parse_llm_json_response(raw_response)
    if parse_error:
        duration_ms = int((time.perf_counter() - started) * 1000)
        return {
            "feature_name": feature_name,
            "status": "parse_error",
            "value": missing_value,
            "parsed": None,
            "raw_response": raw_response,
            "error": parse_error,
            "model": response_model,
            "duration_ms": duration_ms,
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "judge_result": None,
        }

    value = missing_value
    if isinstance(parsed, dict):
        raw_value = parsed.get("value", missing_value)
        value = str(raw_value).strip() if raw_value is not None else missing_value
        if not value:
            value = missing_value

    judge_result: Optional[Dict[str, Any]] = None
    if options.judge_enabled and not (should_cancel and should_cancel()):
        judge_result = run_judge_test(
            feature=feature,
            report_text=report_text,
            extracted_value=value,
            extraction_prompt=prompt,
            options=options,
            should_cancel=should_cancel,
        )

    duration_ms = int((time.perf_counter() - started) * 1000)
    return {
        "feature_name": feature_name,
        "status": "ok",
        "value": value,
        "parsed": parsed,
        "raw_response": raw_response,
        "error": "",
        "model": response_model,
        "duration_ms": duration_ms,
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "judge_result": judge_result,
    }


def run_all_feature_tests(
    features: Sequence[Dict[str, Any]],
    report_text: str,
    options: CalibrationOptions,
    *,
    progress_callback: Optional[Callable[[int, int, Optional[Dict[str, Any]]], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    total = len(features)

    for index, feature in enumerate(features, start=1):
        if should_cancel and should_cancel():
            break

        result = run_feature_test(
            feature=feature,
            report_text=report_text,
            options=options,
            should_cancel=should_cancel,
        )
        results.append(result)

        if progress_callback:
            progress_callback(index, total, result)

    return results
