import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.scripts.calibrator_api.calibration_service import CalibrationOptions
from backend.scripts.calibrator_api.extraction_jobs import (
    ExtractionJobStore,
    ExtractionJobStoreV2,
    ExtractionV2ConflictError,
)


class ExtractionJobStoreTests(unittest.TestCase):
    def test_create_job_writes_output_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_csv_path = Path(temp_dir) / "extracted.csv"
            store = ExtractionJobStore()

            with patch(
                "backend.scripts.extract_pipeline.call_llamacpp_with_retries",
                return_value={
                    "success": True,
                    "content": '{"value":"normal"}',
                    "error": "",
                    "model": "fake-model",
                },
            ):
                job_id = store.create_job(
                    features=[
                        {
                            "name": "rv_function",
                            "description": "RV function",
                            "allowed_values": ["normal", "reduced", "NA"],
                            "missing_value_rule": "NA",
                            "prompt": "Use explicit words only.",
                        }
                    ],
                    rows=[
                        {"StudyID": "1", "Report": "RV function appears normal."},
                        {"StudyID": "2", "Report": "RV function appears normal."},
                    ],
                    id_column="StudyID",
                    report_column="Report",
                    options=CalibrationOptions(llama_url="http://127.0.0.1:8080"),
                    output_csv_path=output_csv_path,
                )

                deadline = time.time() + 5
                payload = {}
                while time.time() < deadline:
                    payload = store.get_job(job_id)
                    if payload["status"] in {"completed", "failed", "cancelled"}:
                        break
                    time.sleep(0.05)

            self.assertEqual(payload.get("status"), "completed")
            self.assertEqual(payload.get("completed_rows"), 2)
            self.assertEqual(payload.get("ok_rows"), 2)
            self.assertTrue(output_csv_path.exists())
            output_text = output_csv_path.read_text(encoding="utf-8")
            self.assertIn("rv_function", output_text)
            self.assertIn("normal", output_text)


class ExtractionJobStoreV2Tests(unittest.TestCase):
    def test_resume_conflict_detected_on_schema_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_csv_path = Path(temp_dir) / "input.csv"
            output_csv_path = Path(temp_dir) / "output.csv"
            input_csv_path.write_text(
                "StudyID,Report\n1,RV function appears normal.\n",
                encoding="utf-8",
            )

            store = ExtractionJobStoreV2()

            with patch(
                "backend.scripts.extract_pipeline.call_llamacpp_with_retries",
                return_value={
                    "success": True,
                    "content": '{"value":"normal"}',
                    "error": "",
                    "model": "fake-model",
                },
            ):
                job_id, _resume_mode, _processed = store.create_job(
                    features=[
                        {
                            "name": "rv_function",
                            "description": "RV function",
                            "allowed_values": ["normal", "reduced", "NA"],
                            "missing_value_rule": "NA",
                            "prompt": "Use explicit words only.",
                        }
                    ],
                    input_csv_path=input_csv_path,
                    id_column="StudyID",
                    report_column="Report",
                    options=CalibrationOptions(llama_url="http://127.0.0.1:8080"),
                    output_csv_path=output_csv_path,
                    resume=True,
                    overwrite_output=False,
                    write_raw_response=False,
                )

                deadline = time.time() + 5
                while time.time() < deadline:
                    payload = store.get_job(job_id)
                    if payload["status"] in {"completed", "failed", "cancelled"}:
                        break
                    time.sleep(0.05)

            self.assertEqual(payload.get("status"), "completed")
            output_text = output_csv_path.read_text(encoding="utf-8")
            self.assertIn("study_id", output_text)
            self.assertIn("\n1,1,", output_text)

            with self.assertRaises(ExtractionV2ConflictError):
                store.create_job(
                    features=[
                        {
                            "name": "different_feature",
                            "description": "Different feature",
                            "allowed_values": ["yes", "no", "NA"],
                            "missing_value_rule": "NA",
                            "prompt": "Return yes/no only.",
                        }
                    ],
                    input_csv_path=input_csv_path,
                    id_column="StudyID",
                    report_column="Report",
                    options=CalibrationOptions(llama_url="http://127.0.0.1:8080"),
                    output_csv_path=output_csv_path,
                    resume=True,
                    overwrite_output=False,
                    write_raw_response=False,
                )


if __name__ == "__main__":
    unittest.main()
