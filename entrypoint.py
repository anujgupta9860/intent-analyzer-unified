"""Download the unified analyzer checkpoint from GCS on container startup.

Layout: <ANALYZER_CHECKPOINT_DIR>/unified/
Holds: pytorch_model.bin, config.json, tokenizer.json,
tokenizer_config.json, calibration.json (one checkpoint for all agents).

Skips download when the local dir already exists and is complete.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

CHECKPOINT_GCS = os.environ.get(
    "ANALYZER_CHECKPOINT_GCS",
    "gs://laya-checkpoints-anuj/intent-analyzer-unified/",
)
CHECKPOINT_LOCAL = os.environ.get("ANALYZER_CHECKPOINT_DIR", "/srv/app/models")

_NEEDED = (
    "pytorch_model.bin", "config.json", "tokenizer.json",
    "tokenizer_config.json", "calibration.json",
)


def _has_all(d: Path) -> bool:
    return all((d / f).exists() for f in _NEEDED)


def download_unified() -> None:
    dest = Path(CHECKPOINT_LOCAL) / "unified"
    if _has_all(dest):
        print(f"[unified] checkpoint present at {dest}, skipping download",
              flush=True)
        return
    dest.mkdir(parents=True, exist_ok=True)
    src = CHECKPOINT_GCS.rstrip("/") + "/unified/"
    print(f"[unified] downloading checkpoint from {src} ...", flush=True)
    try:
        from google.cloud import storage
    except ImportError:
        print("google-cloud-storage not installed; cannot download",
              file=sys.stderr)
        sys.exit(1)
    assert src.startswith("gs://"), src
    bucket_name, prefix = src[5:].split("/", 1)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    n = 0
    for blob in bucket.list_blobs(prefix=prefix):
        name = blob.name[len(prefix):]
        if not name or "/" in name:
            continue  # flat layout only
        blob.download_to_filename(str(dest / name))
        n += 1
    if not _has_all(dest):
        missing = [f for f in _NEEDED if not (dest / f).exists()]
        print(f"[unified] WARNING: missing files after download: {missing}",
              file=sys.stderr)
    else:
        print(f"[unified] downloaded {n} files", flush=True)


def main() -> None:
    download_unified()
    # hand off to the service
    os.execvp("uvicorn", ["uvicorn", "src.app:app",
                          "--host", "0.0.0.0",
                          "--port", os.environ.get("PORT", "8080")])


if __name__ == "__main__":
    main()
