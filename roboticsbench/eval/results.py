"""Aggregation of episode results into benchmark scores.

Per-episode score = 1.0 on success, otherwise the latched partial credit (milestone ladder, always < 1).
Per-task score = mean over seeds. Headline score = mean over tasks (every task weighs the same), in percent.
Episodes with ``infra_ok = False`` (model API outage, crashed harness) are excluded and reported separately.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def load_results(run_dir: str | Path) -> list[dict[str, Any]]:
    out = []
    for p in sorted(Path(run_dir).rglob("result.json")):
        try:
            out.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            continue
    return out


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _stderr(xs: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n)


def summarize(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    results = list(results)
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        key = r.get("model") or (r.get("agent") or {}).get("agent") or "unknown"
        by_model[key].append(r)
    models = {}
    for model, rs in by_model.items():
        ok = [r for r in rs if r.get("infra_ok", True)]
        by_task: dict[str, list[dict]] = defaultdict(list)
        for r in ok:
            v = r.get("variant") or "skills"
            by_task[r["task"] + ("" if v in ("skills", "standard") else f"@{v}")].append(r)
        tasks = {}
        for t, trs in sorted(by_task.items()):
            scores = [float(r["score"]) for r in trs]
            tasks[t] = {"score": _mean(scores), "success_rate": _mean([float(r["success"]) for r in trs]),
                        "n": len(trs), "category": trs[0].get("category"), "robot": trs[0].get("robot"),
                        "difficulty": trs[0].get("difficulty"),
                        "mean_actions": _mean([float(r.get("n_actions", 0)) for r in trs]),
                        "statuses": dict(_count(r["status"] for r in trs))}
        task_scores = [v["score"] for v in tasks.values()]
        cats: dict[str, list[float]] = defaultdict(list)
        robots: dict[str, list[float]] = defaultdict(list)
        for v in tasks.values():
            cats[v["category"]].append(v["score"])
            robots[v["robot"]].append(v["score"])
        usage: dict[str, float] = defaultdict(float)
        for r in ok:
            for k, v in (r.get("usage") or {}).items():
                usage[k] += float(v)
        models[model] = {
            "score": 100 * _mean(task_scores) if task_scores else float("nan"),
            "stderr": None if _stderr(task_scores) is None else 100 * _stderr(task_scores),
            "success_rate": 100 * _mean([float(r["success"]) for r in ok]) if ok else float("nan"),
            "episodes": len(rs), "infra_failures": len(rs) - len(ok),
            "by_robot": {k: 100 * _mean(v) for k, v in robots.items()},
            "by_category": {k: 100 * _mean(v) for k, v in cats.items()},
            "tasks": tasks, "usage": dict(usage),
        }
    return {"models": models, "n_results": len(results)}


def _count(it: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for x in it:
        out[x] += 1
    return out


def leaderboard_text(summary: dict[str, Any]) -> str:
    rows = sorted(summary["models"].items(), key=lambda kv: -(kv[1]["score"] if kv[1]["score"] == kv[1]["score"] else -1))
    lines = [f"{'model':42s} {'score':>7s} {'success':>8s} {'humanoid':>9s} {'arm':>6s} {'episodes':>9s}"]
    for m, s in rows:
        br = s["by_robot"]
        lines.append(f"{m[:42]:42s} {s['score']:7.1f} {s['success_rate']:7.1f}% {br.get('g1', float('nan')):9.1f} "
                     f"{br.get('panda', float('nan')):6.1f} {s['episodes']:9d}")
    return "\n".join(lines)
