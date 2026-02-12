import base64
import csv
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.scripts.calibrator_api import csv_service, schema_service

try:
    from fastapi.testclient import TestClient

    from backend.scripts.calibrator_api.app import app

    FASTAPI_AVAILABLE = True
except Exception:
    FASTAPI_AVAILABLE = False

if FASTAPI_AVAILABLE:
    try:
        _probe_client = TestClient(app)
        _probe_client.close()
    except Exception:
        FASTAPI_AVAILABLE = False


class CsvServiceTests(unittest.TestCase):
    def test_parse_csv_base64_infers_columns(self):
        csv_text = "StudyID,Report,Other\n1,EF 55,foo\n2,EF 40,bar\n"
        encoded = base64.b64encode(csv_text.encode("utf-8")).decode("ascii")

        result = csv_service.parse_csv_base64(encoded, file_name="sample.csv", preview_rows=2)

        self.assertEqual(result["source"], "sample.csv")
        self.assertEqual(result["row_count"], 2)
        self.assertEqual(result["inferred_id_column"], "StudyID")
        self.assertEqual(result["inferred_report_column"], "Report")
        self.assertEqual(len(result["preview"]), 2)

    def test_open_csv_stream_reads_headers_without_materializing_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "input.csv"
            csv_path.write_text(
                "StudyID,Report\n1,First report\n2,Second report\n",
                encoding="utf-8",
            )

            with csv_service.open_csv_stream(csv_path) as (headers, row_iterator, encoding, delimiter):
                self.assertEqual(headers, ["StudyID", "Report"])
                self.assertEqual(encoding, "utf-8-sig")
                self.assertEqual(delimiter, ",")
                self.assertFalse(isinstance(row_iterator, list))
                first_row = next(row_iterator)
                self.assertEqual(first_row["StudyID"], "1")
                self.assertEqual(first_row["Report"], "First report")

    def test_read_csv_headers_returns_columns_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "sample.csv"
            csv_path.write_text(
                "record_id;report_text\n1;ok\n",
                encoding="utf-8",
            )

            payload = csv_service.read_csv_headers(csv_path)

            self.assertEqual(payload["columns"], ["record_id", "report_text"])
            self.assertEqual(payload["delimiter"], ";")


class SchemaServiceTests(unittest.TestCase):
    def test_save_and_load_schema_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            schema_path = Path(temp_dir) / "schema.json"
            features = [
                {
                    "name": "lvef_percent",
                    "description": "LVEF value",
                    "type_hint": "numeric_or_NA",
                    "missing_value_rule": "NA",
                    "prompt": "Use explicit numeric EF value only.",
                }
            ]

            saved = schema_service.save_schema(schema_path, features=features)
            loaded = schema_service.load_schema(schema_path)

            self.assertEqual(saved["schema_path"], str(schema_path))
            self.assertEqual(loaded["features"][0]["name"], "lvef_percent")

    def test_save_and_load_session_with_experiment_profiles(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state_dir = Path(temp_dir)
            state_file = state_dir / "calibrator_web_state.json"
            with patch.object(schema_service, "STATE_DIR", state_dir), patch.object(
                schema_service, "STATE_FILE", state_file
            ):
                schema_service.save_session_state(
                    {
                        "csv_path": "/tmp/reports.csv",
                        "csv_cache": {
                            "source": "reports.csv",
                            "columns": ["StudyID", "Report"],
                            "row_count": 2,
                            "preview": [
                                {"row_number": 1, "values": {"StudyID": "1", "Report": "EF 55"}},
                                {"row_number": 2, "values": {"StudyID": "2", "Report": "EF 45"}},
                            ],
                            "encoding": "utf-8",
                            "delimiter": ",",
                            "inferred_id_column": "StudyID",
                            "inferred_report_column": "Report",
                        },
                        "features": [],
                        "experiment_profiles": [
                            {
                                "id": "baseline_experiment",
                                "name": "Baseline",
                                "system_instructions": "Use strict evidence.",
                                "extraction_instructions": "",
                                "reasoning_mode": "react_style",
                                "reasoning_instructions": "Observe, reason, verify.",
                                "output_instructions": 'Return {"value": "..."} only.',
                                "judge": {
                                    "enabled": True,
                                    "model": "judge-model",
                                    "instructions": "Reject unsupported values.",
                                    "acceptance_threshold": 0.7,
                                },
                            }
                        ],
                        "active_experiment_id": "baseline_experiment",
                    }
                )
                loaded = schema_service.load_session_state()

        self.assertEqual(loaded["active_experiment_id"], "baseline_experiment")
        self.assertEqual(len(loaded["experiment_profiles"]), 1)
        profile = loaded["experiment_profiles"][0]
        self.assertEqual(profile["reasoning_mode"], "react_style")
        self.assertEqual(profile["judge"]["enabled"], True)
        self.assertEqual(profile["judge"]["acceptance_threshold"], 0.7)
        self.assertEqual(loaded["csv_cache"]["source"], "reports.csv")
        self.assertEqual(loaded["csv_cache"]["row_count"], 2)


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi not installed")
class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")

    def test_csv_load_endpoint_with_base64(self):
        csv_text = "record_id,report_text\n1,EF 50\n"
        encoded = base64.b64encode(csv_text.encode("utf-8")).decode("ascii")

        response = self.client.post(
            "/api/csv/load",
            json={"file_name": "inline.csv", "file_base64": encoded, "preview_rows": 3},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["row_count"], 1)
        self.assertEqual(payload["inferred_id_column"], "record_id")

    def test_feature_test_endpoint(self):
        with patch(
            "backend.scripts.extract_pipeline.call_llamacpp_with_retries",
            return_value={
                "success": True,
                "content": '{"value":"normal"}',
                "error": "",
                "model": "fake-model",
            },
        ):
            response = self.client.post(
                "/api/test/feature",
                json={
                    "feature": {
                        "name": "rv_function",
                        "description": "RV function",
                        "allowed_values": ["normal", "reduced", "NA"],
                        "missing_value_rule": "NA",
                        "prompt": "Use explicit words only.",
                    },
                    "report_text": "RV function appears normal.",
                    "llama_url": "http://127.0.0.1:8080",
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["value"], "normal")

    def test_feature_test_endpoint_with_judge(self):
        with patch(
            "backend.scripts.extract_pipeline.call_llamacpp_with_retries",
            side_effect=[
                {
                    "success": True,
                    "content": '{"value":"normal"}',
                    "error": "",
                    "model": "extractor-model",
                },
                {
                    "success": True,
                    "content": '{"verdict":"accepted","score":0.91,"rationale":"Supported by explicit mention."}',
                    "error": "",
                    "model": "judge-model",
                },
            ],
        ):
            response = self.client.post(
                "/api/test/feature",
                json={
                    "feature": {
                        "name": "rv_function",
                        "description": "RV function",
                        "allowed_values": ["normal", "reduced", "NA"],
                        "missing_value_rule": "NA",
                        "prompt": "Use explicit words only.",
                    },
                    "report_text": "RV function appears normal.",
                    "llama_url": "http://127.0.0.1:8080",
                    "experiment_id": "exp_a",
                    "experiment_name": "ReAct candidate",
                    "reasoning_mode": "react_style",
                    "judge": {
                        "enabled": True,
                        "model": "judge-model",
                        "instructions": "Accept only explicit support.",
                        "acceptance_threshold": 0.8,
                    },
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["experiment_name"], "ReAct candidate")
        self.assertEqual(payload["judge_result"]["status"], "accepted")
        self.assertEqual(payload["judge_result"]["model"], "judge-model")

    def test_models_list_endpoint(self):
        with patch(
            "backend.scripts.calibrator_api.app.fetch_server_models",
            return_value=["model-a", "model-b"],
        ):
            response = self.client.post(
                "/api/models/list",
                json={"llama_url": "http://127.0.0.1:8080"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["models"], ["model-a", "model-b"])
        self.assertEqual(payload["llama_url"], "http://127.0.0.1:8080")

    def test_hf_search_endpoint(self):
        with patch(
            "backend.scripts.calibrator_api.app.search_gguf_models",
            return_value=[
                {
                    "repo_id": "foo/bar-gguf",
                    "downloads": 123,
                    "likes": 10,
                    "last_modified": "2026-01-01T00:00:00.000Z",
                    "gguf_files": [
                        {
                            "file_name": "model.Q4_K_M.gguf",
                            "size_bytes": 4321000000,
                            "size_gb": 4.32,
                        }
                    ],
                }
            ],
        ):
            response = self.client.post(
                "/api/hf/gguf/search",
                json={"query": "phi", "limit": 10},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload["models"]), 1)
        self.assertEqual(payload["models"][0]["repo_id"], "foo/bar-gguf")

    def test_hf_files_endpoint(self):
        with patch(
            "backend.scripts.calibrator_api.app.list_gguf_files",
            return_value=[
                {"file_name": "a.Q4_K_M.gguf", "size_bytes": 3900000000, "size_gb": 3.9},
                {"file_name": "a.Q8_0.gguf", "size_bytes": 7200000000, "size_gb": 7.2},
            ],
        ):
            response = self.client.post(
                "/api/hf/gguf/files",
                json={"repo_id": "foo/bar-gguf"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["repo_id"], "foo/bar-gguf")
        self.assertEqual(payload["files"][0]["file_name"], "a.Q4_K_M.gguf")
        self.assertEqual(payload["files"][0]["size_gb"], 3.9)

    def test_hf_download_endpoint(self):
        with patch(
            "backend.scripts.calibrator_api.app.download_gguf_model",
            return_value={
                "repo_id": "foo/bar-gguf",
                "file_name": "a.Q4_K_M.gguf",
                "destination_dir": "/Users/test/models",
                "downloaded_path": "/Users/test/models/a.Q4_K_M.gguf",
            },
        ):
            response = self.client.post(
                "/api/hf/gguf/download",
                json={
                    "repo_id": "foo/bar-gguf",
                    "file_name": "a.Q4_K_M.gguf",
                    "destination_dir": "/Users/test/models",
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["downloaded_path"], "/Users/test/models/a.Q4_K_M.gguf")

    def test_llama_local_models_endpoint(self):
        with patch(
            "backend.scripts.calibrator_api.app.llama_server_manager.list_local_models",
            return_value=(["/tmp/sample.gguf"], False),
        ), patch(
            "backend.scripts.calibrator_api.app.llama_server_manager.get_binary_path",
            return_value="/usr/local/bin/llama-server",
        ):
            response = self.client.get("/api/llama/local-models")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["models"], ["/tmp/sample.gguf"])
        self.assertEqual(payload["timed_out"], False)

    def test_llama_server_status_endpoint(self):
        mocked_status = {
            "process_running": False,
            "managed_model_path": "",
            "managed_port": None,
            "managed_ctx_size": None,
            "started_at_unix": None,
            "llama_url": "http://127.0.0.1:8080",
            "binary_path": "/usr/local/bin/llama-server",
            "reachable": False,
            "server_models": [],
            "connect_error": "unreachable",
            "logs_tail": [],
        }
        with patch(
            "backend.scripts.calibrator_api.app.llama_server_manager.status",
            return_value=mocked_status,
        ):
            response = self.client.post(
                "/api/llama/server/status",
                json={"llama_url": "http://127.0.0.1:8080"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["reachable"], False)
        self.assertEqual(payload["llama_url"], "http://127.0.0.1:8080")

    def test_llama_server_start_endpoint(self):
        with patch(
            "backend.scripts.calibrator_api.app.llama_server_manager.start",
            return_value={
                "process_running": True,
                "managed_model_path": "/tmp/sample.gguf",
                "managed_port": 8080,
                "managed_ctx_size": 8192,
                "started_at_unix": 123,
            },
        ), patch(
            "backend.scripts.calibrator_api.app.llama_server_manager.status",
            return_value={
                "process_running": True,
                "managed_model_path": "/tmp/sample.gguf",
                "managed_port": 8080,
                "managed_ctx_size": 8192,
                "started_at_unix": 123,
                "llama_url": "http://127.0.0.1:8080",
                "binary_path": "/usr/local/bin/llama-server",
                "reachable": True,
                "server_models": ["model-a"],
                "connect_error": "",
                "logs_tail": [],
            },
        ):
            response = self.client.post(
                "/api/llama/server/start",
                json={"model_path": "/tmp/sample.gguf", "port": 8080, "ctx_size": 8192},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["process_running"], True)
        self.assertEqual(payload["managed_port"], 8080)

    def test_llama_server_ensure_endpoint(self):
        with patch(
            "backend.scripts.calibrator_api.app.llama_server_manager.ensure_running",
            return_value={
                "process_running": True,
                "managed_model_path": "/tmp/sample.gguf",
                "managed_port": 8080,
                "managed_ctx_size": 8192,
                "started_at_unix": 123,
                "changed": False,
            },
        ), patch(
            "backend.scripts.calibrator_api.app.llama_server_manager.status",
            return_value={
                "process_running": True,
                "managed_model_path": "/tmp/sample.gguf",
                "managed_port": 8080,
                "managed_ctx_size": 8192,
                "started_at_unix": 123,
                "llama_url": "http://127.0.0.1:8080",
                "binary_path": "/usr/local/bin/llama-server",
                "reachable": True,
                "server_models": ["model-a"],
                "connect_error": "",
                "logs_tail": [],
            },
        ):
            response = self.client.post(
                "/api/llama/server/ensure",
                json={"model_path": "/tmp/sample.gguf", "port": 8080, "ctx_size": 8192},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["process_running"], True)
        self.assertEqual(payload["changed"], False)

    def test_llama_server_stop_endpoint(self):
        with patch(
            "backend.scripts.calibrator_api.app.llama_server_manager.stop",
            return_value={"process_running": False},
        ), patch(
            "backend.scripts.calibrator_api.app.llama_server_manager.status",
            return_value={
                "process_running": False,
                "managed_model_path": "/tmp/sample.gguf",
                "managed_port": 8080,
                "managed_ctx_size": 8192,
                "started_at_unix": 123,
                "llama_url": "http://127.0.0.1:8080",
                "binary_path": "/usr/local/bin/llama-server",
                "reachable": False,
                "server_models": [],
                "connect_error": "not reachable",
                "logs_tail": ["[managed] llama-server stopped."],
            },
        ):
            response = self.client.post(
                "/api/llama/server/stop",
                json={"llama_url": "http://127.0.0.1:8080"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["process_running"], False)
        self.assertEqual(payload["logs_tail"], ["[managed] llama-server stopped."])

    def test_llama_server_stop_now_endpoint(self):
        with patch(
            "backend.scripts.calibrator_api.app.llama_server_manager.stop",
            return_value={"process_running": False},
        ) as mocked_stop:
            response = self.client.post("/api/llama/server/stop-now")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["stopped"], True)
        mocked_stop.assert_called_once()

    def test_test_all_job_lifecycle(self):
        with patch(
            "backend.scripts.extract_pipeline.call_llamacpp_with_retries",
            return_value={
                "success": True,
                "content": '{"value":"55"}',
                "error": "",
                "model": "fake-model",
            },
        ):
            start_response = self.client.post(
                "/api/test/all",
                json={
                    "features": [
                        {
                            "name": "lvef_percent",
                            "description": "LVEF value",
                            "type_hint": "numeric_or_NA",
                            "missing_value_rule": "NA",
                            "prompt": "Return a number only.",
                        }
                    ],
                    "report_text": "EF 55%.",
                    "llama_url": "http://127.0.0.1:8080",
                },
            )

            self.assertEqual(start_response.status_code, 200)
            job_id = start_response.json()["job_id"]

            deadline = time.time() + 3
            last_payload = {}
            while time.time() < deadline:
                poll_response = self.client.get(f"/api/jobs/{job_id}")
                self.assertEqual(poll_response.status_code, 200)
                last_payload = poll_response.json()
                if last_payload["status"] in {"completed", "failed", "cancelled"}:
                    break
                time.sleep(0.05)

        self.assertEqual(last_payload.get("status"), "completed")
        self.assertEqual(last_payload.get("completed"), 1)

    def test_test_batch_job_lifecycle(self):
        with patch(
            "backend.scripts.extract_pipeline.call_llamacpp_with_retries",
            return_value={
                "success": True,
                "content": '{"value":"normal"}',
                "error": "",
                "model": "fake-model",
            },
        ):
            start_response = self.client.post(
                "/api/test/batch",
                json={
                    "features": [
                        {
                            "name": "rv_function",
                            "description": "RV function",
                            "allowed_values": ["normal", "reduced", "NA"],
                            "missing_value_rule": "NA",
                            "prompt": "Use explicit words only.",
                        }
                    ],
                    "reports": [
                        {"row_number": 1, "study_id": "S1", "report_text": "RV function appears normal."},
                        {"row_number": 2, "study_id": "S2", "report_text": "RV function is mildly reduced."},
                    ],
                    "llama_url": "http://127.0.0.1:8080",
                },
            )

            self.assertEqual(start_response.status_code, 200)
            job_id = start_response.json()["job_id"]

            deadline = time.time() + 3
            last_payload = {}
            while time.time() < deadline:
                poll_response = self.client.get(f"/api/jobs/{job_id}")
                self.assertEqual(poll_response.status_code, 200)
                last_payload = poll_response.json()
                if last_payload["status"] in {"completed", "failed", "cancelled"}:
                    break
                time.sleep(0.05)

        self.assertEqual(last_payload.get("status"), "completed")
        self.assertEqual(last_payload.get("reports_total"), 2)
        self.assertEqual(last_payload.get("reports_completed"), 2)
        self.assertEqual(len(last_payload.get("report_results", [])), 2)
        self.assertEqual(last_payload["report_results"][0]["study_id"], "S1")
        self.assertEqual(last_payload["report_results"][1]["study_id"], "S2")
        self.assertEqual(last_payload["report_results"][0]["results"][0]["study_id"], "S1")
        self.assertEqual(last_payload["report_results"][1]["results"][0]["study_id"], "S2")

    def test_extract_run_job_lifecycle_and_download(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_csv_path = Path(temp_dir) / "reports.csv"
            output_csv_path = Path(temp_dir) / "extracted.csv"
            input_csv_path.write_text(
                "StudyID,Report\n1,RV function appears normal.\n2,RV function appears normal.\n",
                encoding="utf-8",
            )

            with patch(
                "backend.scripts.extract_pipeline.call_llamacpp_with_retries",
                return_value={
                    "success": True,
                    "content": '{"value":"normal"}',
                    "error": "",
                    "model": "fake-model",
                },
            ):
                start_response = self.client.post(
                    "/api/extract/run",
                    json={
                        "features": [
                            {
                                "name": "rv_function",
                                "description": "RV function",
                                "allowed_values": ["normal", "reduced", "NA"],
                                "missing_value_rule": "NA",
                                "prompt": "Use explicit words only.",
                            }
                        ],
                        "id_column": "StudyID",
                        "report_column": "Report",
                        "path": str(input_csv_path),
                        "output_csv_path": str(output_csv_path),
                        "llama_url": "http://127.0.0.1:8080",
                    },
                )

                self.assertEqual(start_response.status_code, 200)
                payload = start_response.json()
                job_id = payload["job_id"]
                self.assertEqual(payload["output_csv_path"], str(output_csv_path))

                deadline = time.time() + 5
                job_payload = {}
                while time.time() < deadline:
                    poll_response = self.client.get(f"/api/extract/jobs/{job_id}")
                    self.assertEqual(poll_response.status_code, 200)
                    job_payload = poll_response.json()
                    if job_payload["status"] in {"completed", "failed", "cancelled"}:
                        break
                    time.sleep(0.05)

                self.assertEqual(job_payload.get("status"), "completed")
                self.assertEqual(job_payload.get("total_rows"), 2)
                self.assertEqual(job_payload.get("completed_rows"), 2)
                self.assertEqual(job_payload.get("ok_rows"), 2)
                self.assertTrue(output_csv_path.exists())

                download_response = self.client.get(f"/api/extract/jobs/{job_id}/download")
                self.assertEqual(download_response.status_code, 200)
                downloaded_text = download_response.content.decode("utf-8")
                self.assertIn("rv_function", downloaded_text)
                self.assertIn("normal", downloaded_text)

    def test_extract_v2_run_job_lifecycle_and_download(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_csv_path = Path(temp_dir) / "reports.csv"
            export_root = Path(temp_dir) / "exports"
            input_csv_path.write_text(
                "StudyID,Report\n1,RV function appears normal.\n2,RV function appears normal.\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"CALIBRATOR_EXPORT_ROOT": str(export_root)}), patch(
                "backend.scripts.extract_pipeline.call_llamacpp_with_retries",
                return_value={
                    "success": True,
                    "content": '{"value":"normal"}',
                    "error": "",
                    "model": "fake-model",
                },
            ):
                start_response = self.client.post(
                    "/api/extract/v2/run",
                    json={
                        "features": [
                            {
                                "name": "rv_function",
                                "description": "RV function",
                                "allowed_values": ["normal", "reduced", "NA"],
                                "missing_value_rule": "NA",
                                "prompt": "Use explicit words only.",
                            }
                        ],
                        "input_csv_path": str(input_csv_path),
                        "id_column": "StudyID",
                        "report_column": "Report",
                        "output_name": "v2_run_output",
                        "resume": True,
                        "overwrite_output": False,
                        "llama_url": "http://127.0.0.1:8080",
                    },
                )

                self.assertEqual(start_response.status_code, 200)
                payload = start_response.json()
                job_id = payload["job_id"]
                self.assertEqual(payload["resume_mode"], "fresh")
                self.assertEqual(payload["processed_rows_at_start"], 0)
                self.assertTrue(str(payload["output_csv_path"]).startswith(str(export_root)))

                deadline = time.time() + 5
                job_payload = {}
                while time.time() < deadline:
                    poll_response = self.client.get(f"/api/extract/v2/jobs/{job_id}")
                    self.assertEqual(poll_response.status_code, 200)
                    job_payload = poll_response.json()
                    if job_payload["status"] in {"completed", "failed", "cancelled"}:
                        break
                    time.sleep(0.05)

                self.assertEqual(job_payload.get("status"), "completed")
                self.assertEqual(job_payload.get("processed_rows"), 2)
                self.assertEqual(job_payload.get("ok_rows"), 2)
                self.assertEqual(job_payload.get("total_rows"), 2)
                self.assertEqual(job_payload.get("progress_percent"), 100)

                download_response = self.client.get(f"/api/extract/v2/jobs/{job_id}/download")
                self.assertEqual(download_response.status_code, 200)
                downloaded_text = download_response.content.decode("utf-8")
                self.assertIn("rv_function", downloaded_text)
                self.assertIn("normal", downloaded_text)
                self.assertIn("study_id", downloaded_text)
                self.assertIn("\n1,1,", downloaded_text)
                self.assertIn("\n2,2,", downloaded_text)

    def test_extract_v2_cancel_mid_run_keeps_partial_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_csv_path = Path(temp_dir) / "reports.csv"
            export_root = Path(temp_dir) / "exports"
            rows = ["StudyID,Report"]
            for idx in range(1, 81):
                rows.append(f"{idx},RV function appears normal.")
            input_csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

            def slow_llm(*_args, **_kwargs):
                time.sleep(0.02)
                return {
                    "success": True,
                    "content": '{"value":"normal"}',
                    "error": "",
                    "model": "fake-model",
                }

            with patch.dict(os.environ, {"CALIBRATOR_EXPORT_ROOT": str(export_root)}), patch(
                "backend.scripts.extract_pipeline.call_llamacpp_with_retries",
                side_effect=slow_llm,
            ):
                start_response = self.client.post(
                    "/api/extract/v2/run",
                    json={
                        "features": [
                            {
                                "name": "rv_function",
                                "description": "RV function",
                                "allowed_values": ["normal", "reduced", "NA"],
                                "missing_value_rule": "NA",
                                "prompt": "Use explicit words only.",
                            }
                        ],
                        "input_csv_path": str(input_csv_path),
                        "id_column": "StudyID",
                        "report_column": "Report",
                        "output_name": "v2_cancel_output",
                        "resume": True,
                        "overwrite_output": False,
                        "llama_url": "http://127.0.0.1:8080",
                    },
                )
                self.assertEqual(start_response.status_code, 200)
                payload = start_response.json()
                job_id = payload["job_id"]
                output_csv_path = Path(payload["output_csv_path"])

                cancel_sent = False
                deadline = time.time() + 10
                final_payload = {}
                while time.time() < deadline:
                    poll_response = self.client.get(f"/api/extract/v2/jobs/{job_id}")
                    self.assertEqual(poll_response.status_code, 200)
                    final_payload = poll_response.json()

                    if (
                        not cancel_sent
                        and final_payload["status"] == "running"
                        and int(final_payload.get("processed_rows", 0)) >= 5
                    ):
                        cancel_response = self.client.post(f"/api/extract/v2/jobs/{job_id}/cancel")
                        self.assertEqual(cancel_response.status_code, 200)
                        cancel_sent = True

                    if final_payload["status"] in {"completed", "failed", "cancelled"}:
                        break
                    time.sleep(0.05)

                self.assertTrue(cancel_sent)
                self.assertEqual(final_payload.get("status"), "cancelled")
                processed_rows = int(final_payload.get("processed_rows", 0))
                self.assertGreater(processed_rows, 0)
                self.assertLess(processed_rows, 80)
                self.assertTrue(output_csv_path.exists())
                with output_csv_path.open("r", encoding="utf-8", newline="") as handle:
                    output_rows = list(csv.DictReader(handle))
                self.assertEqual(len(output_rows), processed_rows)

    def test_extract_v2_resume_from_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_csv_path = Path(temp_dir) / "reports.csv"
            export_root = Path(temp_dir) / "exports"
            rows = ["StudyID,Report"]
            for idx in range(1, 13):
                rows.append(f"{idx},RV function appears normal.")
            input_csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

            def slow_llm(*_args, **_kwargs):
                time.sleep(0.02)
                return {
                    "success": True,
                    "content": '{"value":"normal"}',
                    "error": "",
                    "model": "fake-model",
                }

            with patch.dict(os.environ, {"CALIBRATOR_EXPORT_ROOT": str(export_root)}), patch(
                "backend.scripts.extract_pipeline.call_llamacpp_with_retries",
                side_effect=slow_llm,
            ):
                start_response = self.client.post(
                    "/api/extract/v2/run",
                    json={
                        "features": [
                            {
                                "name": "rv_function",
                                "description": "RV function",
                                "allowed_values": ["normal", "reduced", "NA"],
                                "missing_value_rule": "NA",
                                "prompt": "Use explicit words only.",
                            }
                        ],
                        "input_csv_path": str(input_csv_path),
                        "id_column": "StudyID",
                        "report_column": "Report",
                        "output_name": "v2_resume_output",
                        "resume": True,
                        "overwrite_output": False,
                        "llama_url": "http://127.0.0.1:8080",
                    },
                )
                self.assertEqual(start_response.status_code, 200)
                first_job_id = start_response.json()["job_id"]

                partial_processed = 0
                cancel_sent = False
                deadline = time.time() + 10
                while time.time() < deadline:
                    poll_response = self.client.get(f"/api/extract/v2/jobs/{first_job_id}")
                    self.assertEqual(poll_response.status_code, 200)
                    first_payload = poll_response.json()

                    partial_processed = int(first_payload.get("processed_rows", 0))
                    if not cancel_sent and first_payload["status"] == "running" and partial_processed >= 3:
                        cancel_response = self.client.post(f"/api/extract/v2/jobs/{first_job_id}/cancel")
                        self.assertEqual(cancel_response.status_code, 200)
                        cancel_sent = True

                    if first_payload["status"] in {"completed", "failed", "cancelled"}:
                        break
                    time.sleep(0.05)

                self.assertTrue(cancel_sent)

                resume_response = self.client.post(
                    "/api/extract/v2/run",
                    json={
                        "features": [
                            {
                                "name": "rv_function",
                                "description": "RV function",
                                "allowed_values": ["normal", "reduced", "NA"],
                                "missing_value_rule": "NA",
                                "prompt": "Use explicit words only.",
                            }
                        ],
                        "input_csv_path": str(input_csv_path),
                        "id_column": "StudyID",
                        "report_column": "Report",
                        "output_name": "v2_resume_output",
                        "resume": True,
                        "overwrite_output": False,
                        "llama_url": "http://127.0.0.1:8080",
                    },
                )
                self.assertEqual(resume_response.status_code, 200)
                resume_payload = resume_response.json()
                self.assertEqual(resume_payload["resume_mode"], "resumed")
                self.assertEqual(resume_payload["processed_rows_at_start"], partial_processed)
                second_job_id = resume_payload["job_id"]
                output_csv_path = Path(resume_payload["output_csv_path"])

                deadline = time.time() + 10
                second_payload = {}
                while time.time() < deadline:
                    poll_response = self.client.get(f"/api/extract/v2/jobs/{second_job_id}")
                    self.assertEqual(poll_response.status_code, 200)
                    second_payload = poll_response.json()
                    if second_payload["status"] in {"completed", "failed", "cancelled"}:
                        break
                    time.sleep(0.05)

                self.assertEqual(second_payload.get("status"), "completed")
                self.assertEqual(second_payload.get("processed_rows"), 12)
                self.assertEqual(second_payload.get("total_rows"), 12)
                with output_csv_path.open("r", encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 12)

    def test_extract_v2_conflict_on_incompatible_resume_context(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_csv_path = Path(temp_dir) / "reports.csv"
            export_root = Path(temp_dir) / "exports"
            input_csv_path.write_text(
                "StudyID,Report\n1,RV function appears normal.\n2,RV function appears normal.\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"CALIBRATOR_EXPORT_ROOT": str(export_root)}), patch(
                "backend.scripts.extract_pipeline.call_llamacpp_with_retries",
                return_value={
                    "success": True,
                    "content": '{"value":"normal"}',
                    "error": "",
                    "model": "fake-model",
                },
            ):
                first_response = self.client.post(
                    "/api/extract/v2/run",
                    json={
                        "features": [
                            {
                                "name": "rv_function",
                                "description": "RV function",
                                "allowed_values": ["normal", "reduced", "NA"],
                                "missing_value_rule": "NA",
                                "prompt": "Use explicit words only.",
                            }
                        ],
                        "input_csv_path": str(input_csv_path),
                        "id_column": "StudyID",
                        "report_column": "Report",
                        "output_name": "v2_conflict_output",
                        "resume": True,
                        "overwrite_output": False,
                        "llama_url": "http://127.0.0.1:8080",
                    },
                )
                self.assertEqual(first_response.status_code, 200)
                first_job_id = first_response.json()["job_id"]

                deadline = time.time() + 5
                while time.time() < deadline:
                    poll_response = self.client.get(f"/api/extract/v2/jobs/{first_job_id}")
                    self.assertEqual(poll_response.status_code, 200)
                    if poll_response.json()["status"] in {"completed", "failed", "cancelled"}:
                        break
                    time.sleep(0.05)

                second_response = self.client.post(
                    "/api/extract/v2/run",
                    json={
                        "features": [
                            {
                                "name": "different_feature",
                                "description": "Another feature",
                                "allowed_values": ["yes", "no", "NA"],
                                "missing_value_rule": "NA",
                                "prompt": "Return yes/no only.",
                            }
                        ],
                        "input_csv_path": str(input_csv_path),
                        "id_column": "StudyID",
                        "report_column": "Report",
                        "output_name": "v2_conflict_output",
                        "resume": True,
                        "overwrite_output": False,
                        "llama_url": "http://127.0.0.1:8080",
                    },
                )

                self.assertEqual(second_response.status_code, 409)
                payload = second_response.json()
                self.assertEqual(payload["error"]["code"], "v2_resume_conflict")
                self.assertIn("remediation", payload["error"]["details"])

    def test_extract_v2_rejects_output_name_traversal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_csv_path = Path(temp_dir) / "reports.csv"
            export_root = Path(temp_dir) / "exports"
            input_csv_path.write_text(
                "StudyID,Report\n1,RV function appears normal.\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"CALIBRATOR_EXPORT_ROOT": str(export_root)}):
                response = self.client.post(
                    "/api/extract/v2/run",
                    json={
                        "features": [
                            {
                                "name": "rv_function",
                                "description": "RV function",
                                "allowed_values": ["normal", "reduced", "NA"],
                                "missing_value_rule": "NA",
                                "prompt": "Use explicit words only.",
                            }
                        ],
                        "input_csv_path": str(input_csv_path),
                        "id_column": "StudyID",
                        "report_column": "Report",
                        "output_name": "../escape",
                        "resume": True,
                        "overwrite_output": False,
                        "llama_url": "http://127.0.0.1:8080",
                    },
                )

            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.json()["error"]["code"], "invalid_output_name")


if __name__ == "__main__":
    unittest.main()
