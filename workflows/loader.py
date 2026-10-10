"""Hot-reloaded workflow definitions.

Workflows live in workflows/workflows.yaml, not in code. The loader
checks the file's mtime on every access and swaps in a new set after
validating it — a broken edit keeps the last good set serving and
records the error (see status()).

The router fetches definitions from GET /workflows and caches them;
the Studio edits them through PUT /workflows.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

TERMINALS = {"complete", "abort", "retry"}


class WorkflowError(RuntimeError):
    pass


class WorkflowLoader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._workflows: list[dict] = []
        self._by_name: dict[str, dict] = {}
        self._mtime: float | None = None
        self.last_reload: str | None = None
        self.last_error: str | None = None
        self.reload()

    # ------------------------------------------------------------------ API
    def reload(self) -> None:
        import yaml
        if not self.path.exists():
            raise WorkflowError(f"workflows file not found: {self.path}")
        data = yaml.safe_load(self.path.read_text()) or {}
        workflows = data.get("workflows", [])
        if not isinstance(workflows, list):
            raise WorkflowError("'workflows' must be a list")
        by_name: dict[str, dict] = {}
        for i, w in enumerate(workflows):
            where = f"workflow #{i + 1} ({w.get('name', 'unnamed')!r})"
            if not isinstance(w, dict):
                raise WorkflowError(f"{where}: each workflow must be a mapping")
            for key in ("name", "description", "agent", "steps"):
                if not w.get(key):
                    raise WorkflowError(f"{where}: missing {key!r}")
            if w["name"] in by_name:
                raise WorkflowError(f"{where}: duplicate workflow name")
            steps = w["steps"]
            if not isinstance(steps, list) or not steps:
                raise WorkflowError(f"{where}: steps must be a non-empty list")
            step_ids = set()
            for j, s in enumerate(steps):
                swhere = f"{where} step #{j + 1}"
                if not isinstance(s, dict):
                    raise WorkflowError(f"{swhere}: each step must be a mapping")
                for key in ("id", "action", "on_success", "on_failure"):
                    if not s.get(key):
                        raise WorkflowError(f"{swhere}: missing {key!r}")
                if s["id"] in step_ids:
                    raise WorkflowError(f"{swhere}: duplicate step id")
                step_ids.add(s["id"])
                for key in ("on_success", "on_failure"):
                    target = s[key]
                    if target not in TERMINALS and target not in step_ids:
                        # forward reference: check after all ids collected
                        pass
                if not isinstance(s.get("needs_slots", []), list):
                    raise WorkflowError(f"{swhere}: needs_slots must be a list")
            # validate transition targets now that all ids are known
            for s in steps:
                for key in ("on_success", "on_failure"):
                    target = s[key]
                    if target not in TERMINALS and target not in step_ids:
                        raise WorkflowError(
                            f"{where} step {s['id']!r}: {key} targets "
                            f"unknown step {target!r}")
            if not isinstance(w.get("triggers", []), list):
                raise WorkflowError(f"{where}: triggers must be a list")
            by_name[w["name"]] = w
        self._workflows = workflows
        self._by_name = by_name
        self._mtime = self.path.stat().st_mtime
        self.last_reload = datetime.now(timezone.utc).isoformat()
        self.last_error = None

    def maybe_reload(self) -> bool:
        """Reload if the file changed. Broken edits keep the old set."""
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            return False
        if self._mtime is not None and mtime <= self._mtime:
            return False
        try:
            self.reload()
            return True
        except WorkflowError as exc:
            self._mtime = mtime
            self.last_error = str(exc)
            return False

    def workflows(self) -> list[dict]:
        self.maybe_reload()
        return list(self._workflows)

    def get(self, name: str) -> dict | None:
        self.maybe_reload()
        return self._by_name.get(name)

    def match(self, text: str, agent: str | None = None) -> dict | None:
        """Identify a workflow from trigger phrases (System 1's workflow
        decision uses this until the Laya workflow head is trained).

        Workflows with any_agent: true match regardless of agent —
        e.g. greeting.
        """
        t = (text or "").lower()
        for w in self.workflows():
            if agent and w.get("agent") != agent and not w.get("any_agent"):
                continue
            for trigger in w.get("triggers", []):
                trig = trigger.lower()
                if len(trig) <= 3:
                    # short triggers ("hi") need word boundaries —
                    # otherwise "hi" matches inside "ship"
                    import re as _re
                    if _re.search(r"\b" + _re.escape(trig) + r"\b", t):
                        return w
                elif trig in t:
                    return w
        return None

    def status(self) -> dict:
        return {
            "path": str(self.path),
            "workflows": [w["name"] for w in self._workflows],
            "last_reload": self.last_reload,
            "last_error": self.last_error,
            "live": self.last_error is None,
        }
