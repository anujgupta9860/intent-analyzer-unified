"""Train the unified ("giant") intent analyzer: one shared encoder,
per-agent heads, single checkpoint for all agents.

Joint training on every agent's dataset at once. Each record is tagged
with its agent; the model routes each sample to that agent's heads.
The API contract is unchanged: POST /analyze {"query", "worker_agent"}
— the worker_agent (agent name) now selects heads inside the one model
instead of selecting which per-agent checkpoint to load.

Usage:
    # build per-agent datasets first (same as per-agent track)
    for a in billing orders support account sales; do
        python train/build_dataset.py --agent $a --out train/data
    done
    python train/finetune_unified.py --data train/data --out models/unified

    # serve (auto-detected when models/unified/calibration.json exists)
    ANALYZER_CHECKPOINT_DIR=models uvicorn src.app:app

Supports --model for the BERT fallback (see docs/LOCAL_TRAIN_AND_DEPLOY.md §7):
    python train/finetune_unified.py --data train/data --out models/unified \
        --model bert-base-uncased
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.analyzer_questions import (  # noqa: E402
    AGENT_IDS,
    choice_options,
    choice_question,
    noul_questions,
)
from src.unified_model import UnifiedAnalyzer  # noqa: E402

# Default: local copy (same reason as train/finetune.py — the sandbox
# egress proxy breaks huggingface_hub's httpx client).
MODEL_ID = str(Path(__file__).resolve().parent.parent
               / "model-hub" / "ModernBERT-base")

_CHOICE_QID = {
    "billing": "billing_action", "orders": "order_action",
    "support": "support_topic", "account": "account_action",
    "sales": "sales_intent",
}


def load_records(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True,
                        help="dir with <agent>.jsonl files (from build_dataset.py)")
    parser.add_argument("--out", required=True,
                        help="output dir, e.g. models/unified")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.model_selection import train_test_split
    from transformers import AutoTokenizer, Trainer, TrainingArguments

    data_dir = Path(args.data)
    agent_to_idx = {a: i for i, a in enumerate(AGENT_IDS)}
    max_noul = max(len(noul_questions(a)) for a in AGENT_IDS)

    # --- load + validate every agent's dataset ---
    per_agent: dict[str, list[dict]] = {}
    for agent in AGENT_IDS:
        p = data_dir / f"{agent}.jsonl"
        if not p.exists():
            raise SystemExit(
                f"missing {p} — run: python train/build_dataset.py "
                f"--agent {agent} --out {data_dir}")
        recs = load_records(p)
        cqid = _CHOICE_QID[agent]
        labels = choice_options(agent)
        nqs = noul_questions(agent)
        label2id = {l: i for i, l in enumerate(labels)}
        for r in recs:
            assert r[cqid] in label2id, f"{agent}: unknown label {r[cqid]!r}"
            for n in nqs:
                assert n in r, f"{agent}: missing noul {n}"
        per_agent[agent] = recs
        print(f"{agent}: {len(recs)} records "
              f"(choice={len(labels)}, noul={len(nqs)})")

    # --- stratified split per agent, then merge ---
    train_all, val_all = [], []
    for agent, recs in per_agent.items():
        cqid = _CHOICE_QID[agent]
        strat = [r[cqid] for r in recs]
        tr, va = train_test_split(recs, test_size=0.2,
                                  random_state=args.seed, stratify=strat)
        train_all += [(agent, r) for r in tr]
        val_all += [(agent, r) for r in va]
    print(f"unified: {len(train_all)} train / {len(val_all)} val "
          f"({len(AGENT_IDS)} agents)")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def encode(pairs):
        agents = [a for a, _ in pairs]
        recs = [r for _, r in pairs]
        enc = tokenizer([r["query"] for r in recs], truncation=True,
                        padding=True, max_length=128)
        enc["agent_ids"] = [agent_to_idx[a] for a in agents]
        enc["choice_labels"] = [
            choice_options(a).index(r[_CHOICE_QID[a]])
            for a, r in pairs
        ]
        nl, nm = [], []
        for a, r in pairs:
            nqs = noul_questions(a)
            row = [float(r[n]) for n in nqs]
            mask = [1.0] * len(nqs)
            pad = max_noul - len(nqs)
            nl.append(row + [0.0] * pad)
            nm.append(mask + [0.0] * pad)
        enc["noul_labels"] = nl
        enc["noul_mask"] = nm
        return enc

    train_enc, val_enc = encode(train_all), encode(val_all)

    class DictDataset(torch.utils.data.Dataset):
        def __init__(self, enc):
            self.enc = enc

        def __len__(self):
            return len(self.enc["input_ids"])

        def __getitem__(self, i):
            return {k: torch.tensor(v[i]) for k, v in self.enc.items()}

    model = UnifiedAnalyzer(args.model)

    out = Path(args.out)
    training_args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=args.seed,
        report_to="none",
    )

    # Trainer needs a single "logits" key; the core forward already
    # returns the routed per-agent loss. NOTE: the forward signature
    # must list argument names explicitly — Trainer filters batch keys
    # against it, and **kwargs makes it drop everything.
    class TrainerWrapper(nn.Module):
        def __init__(self, core):
            super().__init__()
            self.core = core

        def forward(self, input_ids, attention_mask, agent_ids,
                    choice_labels=None, noul_labels=None, noul_mask=None):
            o = self.core(input_ids=input_ids, attention_mask=attention_mask,
                          agent_ids=agent_ids, choice_labels=choice_labels,
                          noul_labels=noul_labels, noul_mask=noul_mask)
            return {"loss": o["loss"], "logits": o["choice_logits"]}

    wrapper = TrainerWrapper(model)
    trainer = Trainer(model=wrapper, args=training_args,
                      train_dataset=DictDataset(train_enc),
                      eval_dataset=DictDataset(val_enc))
    trainer.train()

    # --- per-agent temperature calibration on the val split ---
    core = wrapper.core
    core.eval()

    def fit_temp(logits, targets, is_bce=False):
        log_temp = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=50)

        def closure():
            opt.zero_grad()
            temp = torch.exp(log_temp)
            if is_bce:
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits / temp, targets)
            else:
                loss = torch.nn.functional.cross_entropy(logits / temp, targets)
            loss.backward()
            return loss

        opt.step(closure)
        temp = float(torch.exp(log_temp).detach())
        if not (0.05 <= temp <= 10.0):
            log.warning("degenerate temperature %.3f, using 1.0", temp)
            return 1.0
        return temp

    calibration: dict[str, dict] = {}
    # The Trainer handles device placement, but this manual loop must do it
    # explicitly — on MPS/CUDA the model is not on CPU.
    _dev = next(core.parameters()).device
    with torch.no_grad():
        for agent in AGENT_IDS:
            idx = agent_to_idx[agent]
            sel = [i for i, (a, _) in enumerate(val_all) if a == agent]
            if not sel:
                continue
            nq = len(noul_questions(agent))
            cl_list, nl_list, ct_list, nt_list = [], [], [], []
            for i in range(0, len(sel), 32):
                chunk = sel[i:i + 32]
                batch = {
                    k: torch.tensor([v[j] for j in chunk], device=_dev)
                    for k, v in val_enc.items()
                    if k not in ("choice_labels", "noul_labels", "noul_mask")
                }
                o = core(input_ids=batch["input_ids"],
                         attention_mask=batch["attention_mask"],
                         agent_ids=batch["agent_ids"])
                # inference path routes via agent_ids[0]; chunk is
                # single-agent so this is exact; logits back to CPU for metrics
                cl_list.append(o["choice_logits"].cpu())
                nl_list.append(o["noul_logits"].cpu())
                ct_list.append(torch.tensor(
                    [val_enc["choice_labels"][j] for j in chunk]))
                nt_list.append(torch.tensor(
                    [val_enc["noul_labels"][j][:nq] for j in chunk],
                    dtype=torch.float32))
            cl = torch.cat(cl_list)
            nl = torch.cat(nl_list)
            ct = torch.cat(ct_list)
            nt = torch.cat(nt_list)
            t_choice = fit_temp(cl, ct)
            t_noul = [fit_temp(nl[:, j], nt[:, j], is_bce=True)
                      for j in range(nq)]
            preds = (cl / t_choice).argmax(dim=-1)
            acc = (preds == ct).float().mean().item()
            conf = torch.softmax(cl / t_choice, dim=-1).max(
                dim=-1).values.mean().item()
            nprobs = torch.sigmoid(nl / torch.tensor(t_noul))
            nacc = (((nprobs >= 0.5).float() == nt).float().mean().item())
            calibration[agent] = {
                "choice_question": _CHOICE_QID[agent],
                "choice_labels": choice_options(agent),
                "noul_questions": noul_questions(agent),
                "temperature_choice": t_choice,
                "temperatures_noul": t_noul,
                "val_choice_accuracy": acc,
                "val_noul_accuracy": nacc,
                "mean_confidence": conf,
                "val_records": len(sel),
            }
            print(f"{agent}: val choice acc={acc:.3f} "
                  f"(t={t_choice:.3f}) | noul acc={nacc:.3f}")

    # --- export ---
    out.mkdir(parents=True, exist_ok=True)
    torch.save(core.state_dict(), out / "pytorch_model.bin")
    core.encoder.config.save_pretrained(out)
    tokenizer.save_pretrained(out)
    (out / "calibration.json").write_text(json.dumps({
        "model_type": "unified",
        "agents": calibration,
        "base_model": args.model,
    }, indent=2))
    print(f"exported unified checkpoint -> {out}")


if __name__ == "__main__":
    main()
