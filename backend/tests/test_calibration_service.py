from __future__ import annotations

from unittest.mock import patch

from backend.scripts.calibrator_api.calibration_service import CalibrationOptions, run_feature_test


def build_feature() -> dict:
    return {
        "name": "rv_function",
        "description": "RV systolic function",
        "allowed_values": ["normal", "reduced", "NA"],
        "missing_value_rule": "NA",
        "prompt": "Use explicit evidence only.",
    }


def test_run_feature_test_with_judge_result():
    options = CalibrationOptions(
        llama_url="http://127.0.0.1:8080",
        experiment_id="exp_react",
        experiment_name="ReAct candidate",
        reasoning_mode="react_style",
        judge_enabled=True,
        judge_model="judge-model",
        judge_acceptance_threshold=0.8,
    )
    feature = build_feature()

    with patch(
        "backend.scripts.extract_pipeline.call_llamacpp_with_retries",
        side_effect=[
            {
                "success": True,
                "content": '{"value":"normal"}',
                "error": "",
                "model": "extract-model",
            },
            {
                "success": True,
                "content": '{"verdict":"accepted","score":0.95,"rationale":"Explicitly documented."}',
                "error": "",
                "model": "judge-model",
            },
        ],
    ):
        result = run_feature_test(feature=feature, report_text="RV function appears normal.", options=options)

    assert result["status"] == "ok"
    assert result["value"] == "normal"
    assert result["experiment_id"] == "exp_react"
    assert result["experiment_name"] == "ReAct candidate"
    assert result["judge_result"]["status"] == "accepted"
    assert result["judge_result"]["model"] == "judge-model"


def test_run_feature_test_without_judge_result():
    options = CalibrationOptions(
        llama_url="http://127.0.0.1:8080",
        experiment_id="baseline",
        experiment_name="Baseline",
        reasoning_mode="direct",
        judge_enabled=False,
    )
    feature = build_feature()

    with patch(
        "backend.scripts.extract_pipeline.call_llamacpp_with_retries",
        return_value={
            "success": True,
            "content": '{"value":"reduced"}',
            "error": "",
            "model": "extract-model",
        },
    ):
        result = run_feature_test(feature=feature, report_text="RV function mildly reduced.", options=options)

    assert result["status"] == "ok"
    assert result["value"] == "reduced"
    assert result["judge_result"] is None
