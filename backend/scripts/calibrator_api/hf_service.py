from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import error as urllib_error
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


HF_API_BASE = "https://huggingface.co/api"
DEFAULT_DOWNLOAD_DIR = Path.home() / "models"
BYTES_PER_GB = 1_000_000_000


class HuggingFaceServiceError(ValueError):
    """Raised when Hugging Face GGUF operations fail."""


def _request_json(url: str, *, timeout: float = 20.0, token: str = "") -> Any:
    headers = {
        "Accept": "application/json",
        "User-Agent": "data-calibrator/0.1",
    }
    if token.strip():
        headers["Authorization"] = f"Bearer {token.strip()}"

    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            payload_raw = response.read()
    except urllib_error.HTTPError as exc:
        raise HuggingFaceServiceError(f"Hugging Face API returned HTTP {exc.code}.") from exc
    except urllib_error.URLError as exc:
        raise HuggingFaceServiceError(f"Could not reach Hugging Face API: {exc.reason}.") from exc
    except TimeoutError as exc:
        raise HuggingFaceServiceError("Timed out while calling Hugging Face API.") from exc

    try:
        return json.loads(payload_raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        raise HuggingFaceServiceError("Hugging Face API returned invalid JSON.") from exc


def _coerce_size_bytes(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return None
    return parsed


def _size_gb_from_bytes(size_bytes: Optional[int]) -> Optional[float]:
    if size_bytes is None:
        return None
    return round(size_bytes / BYTES_PER_GB, 2)


def _extract_gguf_files(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    siblings = payload.get("siblings", [])
    if not isinstance(siblings, list):
        return []

    files: List[Dict[str, Any]] = []
    seen = set()
    for sibling in siblings:
        if not isinstance(sibling, dict):
            continue
        filename = str(sibling.get("rfilename", "")).strip()
        if not filename or not filename.lower().endswith(".gguf"):
            continue
        if filename in seen:
            continue
        seen.add(filename)

        size_bytes = _coerce_size_bytes(sibling.get("size"))
        if size_bytes is None:
            lfs = sibling.get("lfs")
            if isinstance(lfs, dict):
                size_bytes = _coerce_size_bytes(lfs.get("size"))

        files.append(
            {
                "file_name": filename,
                "size_bytes": size_bytes,
                "size_gb": _size_gb_from_bytes(size_bytes),
            }
        )

    files.sort(key=lambda item: str(item["file_name"]))
    return files


def search_gguf_models(*, query: str = "", limit: int = 20) -> List[Dict[str, Any]]:
    safe_limit = max(1, min(int(limit), 50))
    params = {
        "limit": str(safe_limit),
        "full": "true",
        "sort": "downloads",
        "direction": "-1",
    }
    query_value = query.strip()
    if query_value:
        params["search"] = query_value

    url = f"{HF_API_BASE}/models?{urlencode(params)}"
    payload = _request_json(url)
    if not isinstance(payload, list):
        raise HuggingFaceServiceError("Unexpected response while searching models.")

    models: List[Dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        repo_id = str(item.get("id", "")).strip()
        if not repo_id:
            continue

        gguf_files = _extract_gguf_files(item)
        if not gguf_files:
            continue

        models.append(
            {
                "repo_id": repo_id,
                "downloads": int(item.get("downloads", 0) or 0),
                "likes": int(item.get("likes", 0) or 0),
                "last_modified": str(item.get("lastModified", "") or ""),
                "gguf_files": gguf_files,
            }
        )

    return models


def list_gguf_files(*, repo_id: str) -> List[Dict[str, Any]]:
    repo_value = repo_id.strip()
    if not repo_value:
        raise HuggingFaceServiceError("repo_id must be non-empty.")

    encoded_repo = quote(repo_value, safe="/")
    url = f"{HF_API_BASE}/models/{encoded_repo}?blobs=true"
    payload = _request_json(url)
    if not isinstance(payload, dict):
        raise HuggingFaceServiceError("Unexpected response while loading model files.")

    return _extract_gguf_files(payload)


def _download_with_hf_cli(*, repo_id: str, file_name: str, destination_dir: Path, token: str) -> Path:
    binary = shutil.which("huggingface-cli")
    if not binary:
        raise HuggingFaceServiceError("huggingface-cli was not found in PATH.")

    command = [binary, "download", repo_id, file_name, "--local-dir", str(destination_dir)]
    if token.strip():
        command.extend(["--token", token.strip()])

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip() or "Unknown error"
        raise HuggingFaceServiceError(f"huggingface-cli download failed: {details}")

    expected = destination_dir / file_name
    if expected.exists():
        return expected.resolve()

    for line in reversed((result.stdout or "").splitlines()):
        path = line.strip()
        if path and Path(path).exists():
            return Path(path).resolve()

    matches = list(destination_dir.rglob(Path(file_name).name))
    if matches:
        return matches[0].resolve()

    raise HuggingFaceServiceError("Download finished but the GGUF file path could not be determined.")


def _download_with_http(*, repo_id: str, file_name: str, destination_path: Path, token: str) -> Path:
    encoded_repo = quote(repo_id, safe="/")
    encoded_file = quote(file_name)
    url = f"https://huggingface.co/{encoded_repo}/resolve/main/{encoded_file}?download=true"

    headers = {"User-Agent": "data-calibrator/0.1"}
    if token.strip():
        headers["Authorization"] = f"Bearer {token.strip()}"

    request = Request(url, headers=headers)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with urlopen(request, timeout=45) as response, destination_path.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
    except urllib_error.HTTPError as exc:
        if destination_path.exists():
            destination_path.unlink()
        raise HuggingFaceServiceError(f"Download failed with HTTP {exc.code}.") from exc
    except urllib_error.URLError as exc:
        if destination_path.exists():
            destination_path.unlink()
        raise HuggingFaceServiceError(f"Download failed: {exc.reason}.") from exc
    except TimeoutError as exc:
        if destination_path.exists():
            destination_path.unlink()
        raise HuggingFaceServiceError("Download timed out.") from exc

    return destination_path.resolve()


def download_gguf_model(
    *,
    repo_id: str,
    file_name: str,
    destination_dir: str = "",
    hf_token: str = "",
) -> Dict[str, str]:
    repo_value = repo_id.strip()
    file_value = file_name.strip()
    if not repo_value:
        raise HuggingFaceServiceError("repo_id must be non-empty.")
    if not file_value:
        raise HuggingFaceServiceError("file_name must be non-empty.")
    if not file_value.lower().endswith(".gguf"):
        raise HuggingFaceServiceError("Only .gguf files are supported.")

    available_files = list_gguf_files(repo_id=repo_value)
    available_file_names = {str(entry.get("file_name", "")) for entry in available_files}
    if file_value not in available_file_names:
        raise HuggingFaceServiceError(f"File not found in repository GGUF files: {file_value}")

    if destination_dir.strip():
        destination_path = Path(destination_dir).expanduser()
    else:
        destination_path = DEFAULT_DOWNLOAD_DIR

    destination_path.mkdir(parents=True, exist_ok=True)

    try:
        downloaded_path = _download_with_hf_cli(
            repo_id=repo_value,
            file_name=file_value,
            destination_dir=destination_path,
            token=hf_token,
        )
    except HuggingFaceServiceError as exc:
        if "huggingface-cli was not found" not in str(exc):
            raise
        fallback_target = destination_path / file_value
        downloaded_path = _download_with_http(
            repo_id=repo_value,
            file_name=file_value,
            destination_path=fallback_target,
            token=hf_token,
        )

    return {
        "repo_id": repo_value,
        "file_name": file_value,
        "destination_dir": str(destination_path.resolve()),
        "downloaded_path": str(downloaded_path),
    }
