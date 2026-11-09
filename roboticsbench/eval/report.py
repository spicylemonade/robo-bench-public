"""Static HTML report for a run directory (results + links to videos/traces)."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from roboticsbench.eval.results import load_results, summarize

CSS = """
:root{--bg:#fbfbfa;--fg:#1d1d1f;--muted:#6b6b70;--line:#e3e3e0;--ok:#1f7a3a;--bad:#b3261e;--mid:#8a6d00;--card:#fff}
@media (prefers-color-scheme: dark){:root{--bg:#161618;--fg:#ececee;--muted:#9a9aa0;--line:#2c2c30;--ok:#5fcf80;
--bad:#ff7b72;--mid:#e3c14b;--card:#1f1f22}}
body{background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
margin:0;padding:24px 16px}main{max-width:1200px;margin:0 auto}h1{font-size:22px;margin:0 0 4px}
h2{font-size:16px;margin:28px 0 8px}.muted{color:var(--muted)}table{border-collapse:collapse;width:100%;
background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;font-variant-numeric:tabular-nums}
th{font-weight:600;font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.03em}
.num{text-align:right}.success{color:var(--ok);font-weight:600}.failure,.error{color:var(--bad)}
.timeout,.incomplete{color:var(--mid)}.ep{display:inline-block;margin:0 6px 2px 0;white-space:nowrap}
.wrap{overflow-x:auto}a{color:inherit}
"""


def _fmt(x: Any, d: int = 1) -> str:
    try:
        if x != x:  # nan
            return "-"
        return f"{float(x):.{d}f}"
    except (TypeError, ValueError):
        return "-"


def write_report(run_dir: str | Path, results: list[dict[str, Any]] | None = None) -> Path:
    run_dir = Path(run_dir)
    if results is None:
        results = load_results(run_dir)
        if not results and (run_dir / "results.jsonl").exists():
            results = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines() if line.strip()]
    s = summarize(results)
    esc = html.escape
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>RoboticsBench run</title><style>{CSS}</style></head><body><main>",
             f"<h1>RoboticsBench run <span class='muted'>{esc(run_dir.name)}</span></h1>",
             f"<div class='muted'>{len(results)} episodes</div>", "<h2>Leaderboard</h2><div class='wrap'><table>",
             "<tr><th>model / agent</th><th class='num'>score</th><th class='num'>success</th><th class='num'>humanoid</th>"
             "<th class='num'>arm</th><th class='num'>episodes</th><th class='num'>infra fail</th><th class='num'>tokens in/out</th></tr>"]
    for m, v in sorted(s["models"].items(), key=lambda kv: -(kv[1]["score"] if kv[1]["score"] == kv[1]["score"] else -1)):
        u = v.get("usage", {})
        tok = f"{int(u.get('in_tokens', 0) + u.get('cache_read_tokens', 0)):,} / {int(u.get('out_tokens', 0)):,}" if u else "-"
        parts.append(f"<tr><td>{esc(m)}</td><td class='num'>{_fmt(v['score'])}</td><td class='num'>{_fmt(v['success_rate'])}%</td>"
                     f"<td class='num'>{_fmt(v['by_robot'].get('g1'))}</td><td class='num'>{_fmt(v['by_robot'].get('panda'))}</td>"
                     f"<td class='num'>{v['episodes']}</td><td class='num'>{v['infra_failures']}</td><td class='num'>{tok}</td></tr>")
    parts.append("</table></div>")
    by_task: dict[str, list[dict]] = {}
    for r in results:
        by_task.setdefault(r["task"], []).append(r)
    parts.append("<h2>Episodes</h2><div class='wrap'><table><tr><th>task</th><th>difficulty</th><th>episodes (status / score)</th></tr>")
    for t in sorted(by_task):
        eps = []
        for r in sorted(by_task[t], key=lambda r: (str(r.get("model")), r["seed"])):
            ep = r.get("episode_id", "")
            links = []
            if r.get("video"):
                links.append(f"<a href='{esc(ep)}/{esc(r['video'])}'>video</a>")
            for fname, label in (("trace.json", "trace"), ("referee_trace.json", "trace"),
                                 ("agent_stdout.jsonl", "agent log")):
                if (run_dir / ep / fname).exists():
                    links.append(f"<a href='{esc(ep)}/{fname}'>{label}</a>")
            cost = (r.get("usage") or {}).get("cost_usd")
            if cost is not None:
                links.append(f"<span class='muted'>${float(cost):.2f}</span>")
            label = f"{esc(str(r.get('model') or (r.get('agent') or {}).get('agent')))} s{r['seed']}"
            eps.append(f"<span class='ep'>{label}: <span class='{esc(r['status'])}'>{esc(r['status'])}</span> "
                       f"{_fmt(r['score'], 2)} {' '.join(links)}</span>")
        parts.append(f"<tr><td>{esc(t)}</td><td>{esc(str(by_task[t][0].get('difficulty', '')))}</td><td>{''.join(eps)}</td></tr>")
    parts.append("</table></div></main></body></html>")
    out = run_dir / "report.html"
    out.write_text("\n".join(parts))
    return out
