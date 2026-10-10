# Train & Deploy the Unified Intent Analyzer Locally

One model, one checkpoint, all five agents. The agent name in each request
selects which per-agent question set the model answers.

## 1. Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

CPU is fine for training (~15–25 min for the full run); a GPU is faster.

## 2. Get the encoder weights

`train/finetune_unified.py` defaults to `model-hub/ModernBERT-base` — copy it
from the sibling repo, or fetch it yourself:

```bash
# option A: reuse the local copy from intent-analyzer-laya
cp -r ~/workspace/intent-analyzer-laya/model-hub ./model-hub

# option B: direct hub download (machine with direct HF access)
python train/finetune_unified.py --model answerdotai/ModernBERT-base ...

# option C: curl when the HF client is proxy-blocked
mkdir -p model-hub/ModernBERT-base && cd model-hub/ModernBERT-base
BASE=https://huggingface.co/answerdotai/ModernBERT-base/resolve/main
for f in config.json tokenizer.json tokenizer_config.json model.safetensors; do
  curl -L -o "$f" "$BASE/$f"
done
```

## 3. Train

```bash
source .venv/bin/activate

# 1. datasets (one JSONL per agent; same builder as the per-agent track)
for a in billing orders support account sales; do
  python train/build_dataset.py --agent $a --out train/data
done

# 2. joint training — one run, all agents
python train/finetune_unified.py --data train/data --out models/unified
```

Flags: `--model` (default `model-hub/ModernBERT-base`), `--epochs` (5.0),
`--lr` (2e-5), `--batch` (8), `--seed` (42).

Output `models/unified/`:
- `pytorch_model.bin` — shared encoder + all per-agent heads (~600 MB)
- `config.json`, `tokenizer.json`, `tokenizer_config.json`
- `calibration.json` — per-agent label contracts + fitted temperatures.
  The server validates these against `src/analyzer_questions.py` at load and
  refuses to serve on any mismatch.

## 4. BERT fallback

If ModernBERT weights are blocked in any way, train on BERT instead — one flag,
everything else identical:

```bash
python train/finetune_unified.py --data train/data --out models/unified \
    --model bert-base-uncased
```

Expected: slightly lower accuracy (~0.90–0.97 vs 0.95–1.00), smaller/faster
checkpoint. Never mix encoder tracks in one deployment.

## 5. Serve

```bash
source .venv/bin/activate
ANALYZER_CHECKPOINT_DIR=models uvicorn src.app:app --port 8081
```

The service loads `models/unified/` once and routes by agent name:

```bash
curl -s -X POST localhost:8081/analyze -H 'content-type: application/json' \
  -d '{"query": "delete my account immediately", "worker_agent": "account"}' \
  | python3 -m json.tool
# → delete_account @ ~1.0, is_irreversible=1.0

curl -s localhost:8081/health | python3 -m json.tool
# → {"backend": "unified", "available": {"billing": true, ...}}
```

Env vars: `ANALYZER_CHECKPOINT_DIR` (default `models`),
`ANALYZER_CONFIDENCE_THRESHOLD` (default 0.6 → `needs_human`).

## 6. Test

```bash
.venv/bin/python -m pytest tests/ -q
```

7 contract tests: spec inventory, dataset/spec consistency, choice-order
stability, unified checkpoint validation (label mismatch / wrong model type /
empty agents rejected), head dimensions.

## 7. Deploy

Docker:

```dockerfile
FROM python:3.12-slim
WORKDIR /srv/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY models/unified/ ./models/unified/
ENV ANALYZER_CHECKPOINT_DIR=/srv/app/models
EXPOSE 8081
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8081"]
```

Cloud Run (GCS download on startup):

```bash
ANALYZER_CHECKPOINT_GCS=gs://<bucket>/intent-analyzer-unified \
ANALYZER_CHECKPOINT_DIR=/srv/app/models \
  python entrypoint.py
```

Upload the five serving files to
`gs://<bucket>/intent-analyzer-unified/unified/` after training.

## 8. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `unified checkpoint not found` | train (§3) or check `ANALYZER_CHECKPOINT_DIR` |
| `choice labels ... != spec` | retrain with current `src/analyzer_questions.py`; never hand-edit labels |
| `not a unified checkpoint` | you're pointing at a per-agent checkpoint dir; use `models/unified` |
| OOM in training | `--batch 4` |
| slow first request | one-time model load (~seconds); subsequent requests are ms |
