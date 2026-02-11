# Data Extraction Pipeline (Standalone)

This folder contains a resume-safe CLI pipeline for extracting structured features from free-text reports using a local `llama.cpp` server (`/v1/chat/completions` OpenAI-compatible API).

It also includes a simple desktop calibration UI so non-engineers can tune per-feature prompts before running large-scale extraction.

## 1) Prerequisites

- Python 3.9+
- Installed Python packages:
  - `requests`
  - `fastapi`
  - `uvicorn`
  - `pandas` (optional for downstream analysis, not required by extraction runtime)
- A local `llama-server` binary from `llama.cpp`
- A downloaded `.gguf` model

## 2) Download a GGUF model (Hugging Face)

Example models (choose one):
- `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF`
- `bartowski/Llama-3.2-3B-Instruct-GGUF`
- `Qwen/Qwen2.5-7B-Instruct-GGUF`

Example with `huggingface-cli`:

```bash
huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --local-dir ~/models
```

## 3) Start llama-server manually

Example:

```bash
llama-server -m ~/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --port 8080 --ctx-size 8192
```

Health check:

```bash
curl http://127.0.0.1:8080/v1/models
```

## 4) Run extraction

Schema example is provided at:
- `backend/scripts/feature_schema.example.json`

Example extraction command:

```bash
python backend/scripts/extract_pipeline.py \
  --input-csv data/reports.csv \
  --output-csv data/extracted.csv \
  --schema-file backend/scripts/feature_schema.example.json \
  --llamacpp-base-url http://127.0.0.1:8080 \
  --temperature 0.0 \
  --max-retries 5 \
  --model-label llama-local-8b
```

Default column mapping:
- ID column: `StudyID`
- Report text column: `Report`

If needed, you can override with `--id-column` and `--report-column`.
Optional server model targeting: `--llamacpp-model <model_id>`.

## 4b) Simple calibration UI (recommended before full run)

Launch a local desktop UI (no frontend build):

```bash
python backend/scripts/launch_calibrator.py
```

Create a desktop launcher icon (macOS `.command` file):

```bash
python backend/scripts/create_desktop_launcher.py
```

Optional preload:

```bash
python backend/scripts/launch_calibrator.py \
  --input-csv data/reports.csv \
  --schema-file backend/scripts/feature_schema.example.json
```

In the UI, you can:
- Select a CSV and it auto-loads immediately
- Define each feature and its exact prompt text
- Test one feature or all features on a sample row
- Preview the combined extraction prompt
- Save a calibrated schema JSON for production extraction
- Discover local GGUF models and select one
- Refresh server models from `/v1/models` and choose one for calibration calls
- Use setup helpers to install `llama.cpp` (macOS/Homebrew), download a recommended model, and start/stop `llama-server`

Feature persistence:
- The calibrator now auto-saves your workspace (including feature definitions/prompts) and auto-loads it on next launch.
- Saved state path: `~/.data_prompt_calibrator/calibrator_state.json`

## 4c) Web calibration UI (new default for faster iteration)

The web calibrator keeps the extraction logic in Python and adds a responsive local UI.

Run both backend API and frontend in one command:

```bash
./scripts/run_calibrator_web.sh
```

What this command now does automatically:
- Creates a local Python virtual environment at `.venv` (first run only)
- Installs backend packages from `backend/requirements-web.txt`
- Installs frontend npm packages in `frontend/calibrator-ui`
- Starts both services

This starts:
- FastAPI backend at `http://127.0.0.1:8000`
- React UI at `http://127.0.0.1:5173`

Optional (development hot-reload for backend):

```bash
ENABLE_BACKEND_RELOAD=1 ./scripts/run_calibrator_web.sh
```

If needed, run backend manually:

```bash
python -m uvicorn backend.scripts.calibrator_api.app:app --host 127.0.0.1 --port 8000 --reload
```

Core API endpoints:
- `POST /api/csv/load`
- `POST /api/schema/load`
- `POST /api/schema/save`
- `POST /api/models/list`
- `POST /api/hf/gguf/search`
- `POST /api/hf/gguf/files` (returns per-file size metadata: `size_bytes`, `size_gb`)
- `POST /api/hf/gguf/download`
- `GET /api/llama/local-models`
- `POST /api/llama/server/status`
- `POST /api/llama/server/start`
- `POST /api/llama/server/ensure`
- `POST /api/llama/server/stop`
- `POST /api/test/feature`
- `POST /api/test/all`
- `POST /api/test/batch`
- `GET /api/jobs/{job_id}`
- `POST /api/jobs/{job_id}/cancel`
- `GET /api/health`

Optional session endpoints:
- `GET /api/session/load`
- `POST /api/session/save`

## 5) Resume an interrupted run

Resume is enabled by default. If `--output-csv` already exists, previously processed IDs are skipped.

```bash
python backend/scripts/extract_pipeline.py \
  --input-csv data/reports.csv \
  --output-csv data/extracted.csv \
  --schema-file backend/scripts/feature_schema.example.json \
  --resume
```

To force restart from scratch:

```bash
python backend/scripts/extract_pipeline.py \
  --input-csv data/reports.csv \
  --output-csv data/extracted.csv \
  --schema-file backend/scripts/feature_schema.example.json \
  --no-resume
```

## 6) Evaluate against a gold standard

```bash
python backend/scripts/evaluate_extraction.py \
  --pred-csv data/extracted.csv \
  --gold-csv data/gold.csv \
  --output-summary-csv data/evaluation_summary.csv \
  --schema-file backend/scripts/feature_schema.example.json
```

Metrics:
- Per-feature exact-match accuracy
- Row-level all-features-correct rate

## 7) Troubleshooting

- **Connection refused / timeout**
  - Verify `llama-server` is running.
  - Confirm URL and port (`--llamacpp-base-url`).
- **Slow inference**
  - Use a smaller GGUF quantization/model.
  - Lower context size if memory is constrained.
- **Malformed model output**
  - Keep temperature low (`0.0`).
  - Ensure server supports `response_format={"type":"json_object"}`.
- **UI will not open**
  - Ensure your Python installation includes `tkinter` (standard on most macOS/Linux Python builds).
- **Resume mismatch**
  - If output schema changed between runs, start with `--no-resume`.
- **Column validation errors**
  - Confirm input contains the selected `--id-column` and `--report-column`.
