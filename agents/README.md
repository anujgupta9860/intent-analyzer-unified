# Custom agents

Define a **new agent's** intent taxonomy here — one JSON file per agent
(`<agent>.json`) — or through the Studio's **Training Data** tab, which
writes these files for you.

## Format

```json
{
  "agent": "refunds",
  "choice_question": "refund_action",
  "choice_instructions": "What refund action does this customer need?",
  "labels": {
    "request_refund": "asks for money back for a purchase",
    "check_status": "asks about the status of an existing refund"
  },
  "noul": [
    {"name": "amount_mentioned",
     "instructions": "Does the query mention a specific amount of money?",
     "keywords": ["$", "dollar", "usd"],
     "implied_by": ["request_refund"]}
  ],
  "seeds": {
    "request_refund": ["I want a refund", "Give me my $50 back"],
    "check_status": ["Where is my refund?"]
  }
}
```

- **labels**: 2–20 choice labels, each with a criteria description (what it means).
- **noul**: 0–8 flags. Each fires (`1.0`) when a seed query contains one of its
  `keywords` (case-insensitive) or its choice label is in `implied_by`.
- **seeds**: 1–30 example queries per label. Labels come from these; templates
  paraphrase them without changing labels.

## Workflow

1. Create the spec (Studio tab or by hand; invalid files are ignored with a
   console warning, keeping the last good state).
2. Build the dataset: Studio **Build dataset**, or
   `python train/build_dataset.py --agent refunds --out train/data`.
3. Train: `python train/finetune_unified.py --data train/data --out models/unified`
   (picks up all agents, built-in + custom).
4. Restart the service to serve the new agent (`POST /analyze` with
   `"agent": "refunds"`).

Built-in agents (`billing`, `orders`, `support`, `account`, `sales`) stay in
code (`src/analyzer_questions.py`); these files are merged in at import.
