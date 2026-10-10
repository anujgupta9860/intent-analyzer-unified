"""Unified ("giant") intent analyzer: one model for all agents.

Architecture: a single shared encoder plus per-agent head pairs
(one choice head + one noul head per agent). The agent name selects
which heads answer — so one checkpoint serves every agent's
domain-specific question set.

This mirrors src/analyzer_model.py (per-agent MultiTaskAnalyzer) but
shares the encoder across agents. Per-agent heads keep each agent's
label contract (choice options order, noul question order) independent,
exactly as in the per-agent checkpoints.

Training: train/finetune_unified.py (joint training on all agents' data).
Serving:  src/unified_analyzers.py (UnifiedAnalyzerClient).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from .analyzer_questions import AGENT_IDS, choice_options, noul_questions


class UnifiedAnalyzer(nn.Module):
    """One shared encoder, per-agent choice/noul heads."""

    def __init__(self, base_id: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_id)
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.choice_heads = nn.ModuleDict({
            agent: nn.Linear(h, len(choice_options(agent)))
            for agent in AGENT_IDS
        })
        self.noul_heads = nn.ModuleDict({
            agent: nn.Linear(h, len(noul_questions(agent)))
            for agent in AGENT_IDS
        })
        # agent name -> row index, for tensor-based routing
        self.agent_to_idx = {a: i for i, a in enumerate(AGENT_IDS)}

    def forward(self, input_ids, attention_mask, agent_ids,
                choice_labels=None, noul_labels=None, noul_mask=None):
        """agent_ids: LongTensor [B] of agent row indices.

        choice_labels: LongTensor [B]. noul_labels / noul_mask:
        FloatTensor [B, max_noul] (mask=1 where the agent has that
        noul question; labels are padded with 0 elsewhere).
        """
        pooled = self.dropout(
            self.encoder(input_ids=input_ids,
                         attention_mask=attention_mask).last_hidden_state[:, 0])
        out: dict = {}
        if choice_labels is None:
            # inference: every row in the batch is one agent (serving
            # always sends a single-agent batch); route via first row.
            agent = AGENT_IDS[int(agent_ids[0])]
            out["choice_logits"] = self.choice_heads[agent](pooled)
            out["noul_logits"] = self.noul_heads[agent](pooled)
            return out

        # training: per-sample routing, loss accumulated per agent slice.
        total_loss, total_n = 0.0, 0
        max_noul = noul_labels.shape[1]
        for agent in AGENT_IDS:
            idx = self.agent_to_idx[agent]
            sel = (agent_ids == idx)
            n = int(sel.sum())
            if n == 0:
                continue
            p = pooled[sel]
            cl = self.choice_heads[agent](p)
            nl = self.noul_heads[agent](p)
            ce = nn.functional.cross_entropy(cl, choice_labels[sel])
            nq = len(noul_questions(agent))
            bce = nn.functional.binary_cross_entropy_with_logits(
                nl, noul_labels[sel][:, :nq] * noul_mask[sel][:, :nq],
                reduction="none")
            # masked mean over valid noul entries only
            bce = (bce * noul_mask[sel][:, :nq]).sum() / max(
                noul_mask[sel][:, :nq].sum(), torch.tensor(1.0))
            total_loss = total_loss + (ce + bce) * n
            total_n += n
        out["loss"] = total_loss / max(total_n, 1)
        # concatenated logits for the Trainer's single-logits expectation
        # (not used for the loss; the wrapper below recomputes per agent)
        out["choice_logits"] = torch.zeros(
            pooled.shape[0], 1, device=pooled.device)
        out["noul_logits"] = torch.zeros(
            pooled.shape[0], max_noul, device=pooled.device)
        return out

    @classmethod
    def from_local_config(cls, config, agent_specs: dict) -> "UnifiedAnalyzer":
        """Build from a local HF config (no hub download); caller loads
        the state dict afterwards. agent_specs maps agent -> number of
        noul questions (choice counts come from the spec)."""
        from transformers import AutoModel
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.encoder = AutoModel.from_config(config)
        h = config.hidden_size
        obj.dropout = nn.Dropout(0.1)
        obj.choice_heads = nn.ModuleDict({
            agent: nn.Linear(h, len(choice_options(agent)))
            for agent in AGENT_IDS
        })
        obj.noul_heads = nn.ModuleDict({
            agent: nn.Linear(h, agent_specs[agent])
            for agent in AGENT_IDS
        })
        obj.agent_to_idx = {a: i for i, a in enumerate(AGENT_IDS)}
        return obj
