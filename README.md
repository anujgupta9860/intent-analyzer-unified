# Intent Analyzer — Unified ("Giant")

One intent-analyzer model for **all** worker agents. Each agent keeps its own
domain-specific question set (choice taxonomy + Noul flags), but there is only
**one** model and **one** checkpoint: the request's agent name selects which
per-agent heads answer.

```
POST /analyze {"query": "where is my order?", "worker_agent": "orders"}
  → shared encoder → orders heads → order_status @ 0.97 + order flags

POST /analyze {"query": "delete my account", "worker_agent": "account"}
  → shared encoder → account heads → delete_account @ 1.0 + is_irreversible=1.0
```

## Architecture

```
query + agent name
      │
      ▼
┌─────────────┐
│ shared       │   one transformer encoder (ModernBERT default,
│ encoder      │   BERT fallback) — trained jointly on all agents
└──────┬──────┘
       │  agent name routes here
       ▼
┌──────────────────────────────────────────────┐
│ per-agent heads                               │
│  billing: billing_action + 4 noul             │
│  orders:  order_action   + 3 noul             │
│  support: support_topic  + 2 noul             │
│  account: account_action + 2 noul             │
│  sales:   sales_intent   + 2 noul             │
└──────────────────────────────────────────────┘
```

Why one model instead of five: a single checkpoint to version, deploy, and
monitor; the shared encoder learns cross-domain representations (a refund
phrasing learned for billing helps orders); one GPU/CPU slot in memory
instead of five.

## Quickstart (local)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. build datasets (one JSONL per agent)
for a in billing orders support account sales; do
  python train/build_dataset.py --agent $a --out train/data
done

# 2. train the unified model (~15–25 min on CPU)
python train/finetune_unified.py --data train/data --out models/unified

# 3. serve
ANALYZER_CHECKPOINT_DIR=models uvicorn src.app:app --port 8081

# 4. test
curl -s -X POST localhost:8081/analyze -H 'content-type: application/json' \
  -d '{"query": "I want a refund for the $25 overcharge", "worker_agent": "billing"}' \
  | python3 -m json.tool
```

## Decision framework (layman-configurable)

The analyzer turns text into structured facts; the **decision tree** turns
facts into actions for the A2A agent layer. Rules live in
`policy/decision-tree.yaml` — plain YAML a non-technical person can read —
and are edited visually in the **Decision Tree Studio**:

```bash
ANALYZER_CHECKPOINT_DIR=models uvicorn src.app:app --port 8081
# open http://localhost:8081/studio
```

- Rules are checked **top to bottom, first match wins**. Conditions use
  only valid values (agents, choices, Noul flags come from the analyzer
  spec), comparisons `> >= < <= == !=` on confidence and flag values.
- The Studio's **test panel** dry-runs any query (or a hand-made analyzer
  output) and shows which rule fired and what action results.
- Saving validates the whole tree first — an invalid rule is rejected
  with a plain-English error naming the rule and field.
- `POST /decide {"query", "worker_agent"}` is the agent integration
  point: analyzer output + policy decision (`action`, `worker`, `risk`)
  in one call, ready for A2A dispatch. High-risk decisions
  (e.g. `delete_account` → `escalate_human`) never auto-execute.

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | liveness + per-agent checkpoint coverage |
| POST | `/analyze` | `{"query", "worker_agent", "router_context?"}` → that agent's typed decisions |
| GET | `/feedback/stats` | low-confidence log counters |
| GET | `/feedback/pending` | records awaiting human review |
| POST | `/feedback/review/{id}` | approve \| correct \| reject |

Response contract per agent (unchanged from the per-agent track):

```json
{
  "agent": "billing",
  "choice_question": "billing_action",
  "billing_action": "refund_request",
  "choice_confidence": 0.93,
  "choice_probabilities": {"refund_request": 0.93, "...": 0.01},
  "needs_payment_method": 0.1,
  "is_dispute": 0.2,
  "amount_mentioned": 0.9,
  "is_recurring": 0.05,
  "needs_human": false,
  "model": "unified:models/unified",
  "latency_ms": 210.4
}
```

Confidence below `ANALYZER_CONFIDENCE_THRESHOLD` (default 0.6) →
`needs_human=true` and the decision is logged for human review (RLCD loop).

## Training details

`train/finetune_unified.py`:
- Loads all five `train/data/<agent>.jsonl` datasets, tags each record with
  its agent, stratified 80/20 split per agent.
- Joint multi-task loss: `CE(choice) + masked-BCE(noul)` per agent slice.
- Per-agent temperature calibration on the held-out split (one temperature
  for the choice head + one per Noul head, per agent).
- Exports `models/unified/`: `pytorch_model.bin`, `config.json`,
  `tokenizer*.json`, `calibration.json` (per-agent label contracts +
  temperatures).

The serving client validates the checkpoint against
`src/analyzer_questions.py` (the single source of truth) at load time and
**refuses** to serve on any label mismatch — same guard as the per-agent
track.

**BERT fallback:** if ModernBERT weights are unavailable, train with
`--model bert-base-uncased`. One flag; everything else identical. Do not
mix encoder tracks in one deployment.

## Custom agents (Studio Training Data tab)

A **new agent** needs no code changes:

1. Open the Studio → **Training Data** tab → **+ New agent**.
2. Define the Choice question (its action taxonomy: labels + what each means),
   Noul flags (instructions + trigger keywords + implied-by labels), and
   2+ seed queries per label.
3. **Save agent** → writes `agents/<agent>.json` (validated; bad files are
   ignored with a warning, keeping the last good state).
4. **Build dataset** → generates `train/data/<agent>.jsonl` (synthetic
   paraphrases of your seeds; Noul labels from your keyword/implied-by rules),
   with a balance report and sample preview.
5. Retrain (`train analyzer` via `scripts/laya-stack.sh`, or
   `python train/finetune_unified.py --data train/data --out models/unified`)
   and restart the service to serve it.

The JSON format is documented in `agents/README.md`. Built-in agents stay in
code (`src/analyzer_questions.py`); custom files are merged in at import, so
training, serving, and the dataset builder all see one spec.

## Deploy

Docker (bake or mount `models/unified/`), or Cloud Run via `entrypoint.py`,
which downloads the checkpoint from
`gs://laya-checkpoints-anuj/intent-analyzer-unified/unified/` on startup:

```bash
ANALYZER_CHECKPOINT_GCS=gs://<bucket>/intent-analyzer-unified \
ANALYZER_CHECKPOINT_DIR=/srv/app/models \
  python entrypoint.py
```

## Layout

```text
intent-analyzer-unified/
├── src/
│   ├── analyzer_questions.py  # per-agent question specs (source of truth)
│   ├── unified_model.py       # UnifiedAnalyzer: shared encoder + per-agent heads
│   ├── unified_analyzers.py   # serving client (agent-name routing)
│   ├── app.py                 # FastAPI service
│   ├── config.py              # env settings
│   └── feedback.py            # RLCD human-review loop
├── train/
│   ├── build_dataset.py       # synthetic datasets per agent
│   └── finetune_unified.py    # joint training
├── tests/test_contract.py      # train/serve contract tests
├── entrypoint.py              # GCS download + serve (Cloud Run)
└── docs/LOCAL_TRAIN_AND_DEPLOY.md
```

## Related

- [intent-analyzer-laya](https://github.com/anujgupta9860/intent-analyzer-laya) —
  the per-agent (five-checkpoint) track this unifies.
- [intent-router-laya](https://github.com/anujgupta9860/intent-router-laya) —
  the Tier-1 router whose `worker_agent` decision feeds this service.
