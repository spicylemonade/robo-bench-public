"""Launchers for coding-agent harnesses (Claude Code, Codex, or any CLI via a command template).

Each harness builds a non-interactive command that runs in the episode workspace and parses its JSON event stream
into usage / turn counts for the results table.

    claude-code   claude -p ... --output-format stream-json          (auth: ANTHROPIC_API_KEY or a logged-in CLI)
    codex         codex exec --json ...                              (auth: OPENAI_API_KEY / CODEX_API_KEY or codex login)
    gemini-cli    gemini -p ... --output-format stream-json          (auth: GEMINI_API_KEY)
    antigravity   rbench-antigravity ... -- agy --output-format stream-json ...   (auth: GEMINI_API_KEY)
    grok-build    grok --prompt-file ... --output-format streaming-messages-json   (auth: XAI_API_KEY)
    muse-code     muse exec --json --prompt-file ...                 (auth: META_API_KEY)
    kimi-code     kimi -p ... --output-format stream-json            (auth: KIMI_API_KEY)
    opencode      opencode run --format json ... (brief on stdin)    (auth: the provider's key, e.g. FIREWORKS_API_KEY)
    cmd:<tmpl>    any CLI, e.g. "cmd:opencode run --model {model} {prompt}"  (placeholders: {prompt} {model}
                  {effort} {workspace}; the stdout is stored but not parsed)
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Launch:
    cmd: list[str]
    env: dict[str, str] = field(default_factory=dict)
    # the prompt, fed on stdin: kept off the command line, where an agent's own `pkill -f walk.py` matched the
    # prompt text and killed the agent (2026-10-09, Opus humanoid episodes exited with code 143)
    stdin: str | None = None
    # files written into the workspace before the agent starts (relative path -> text): harness settings, or the
    # prompt for CLIs that read it from a file
    files: dict[str, str] = field(default_factory=dict)


class Harness:
    name = "harness"
    binary = ""

    def available(self) -> bool:
        return bool(shutil.which(self.binary)) if self.binary else True

    def launch(self, prompt: str, workspace: Path, *, model: str | None, effort: str | None,
               budget_usd: float | None, isolated: bool) -> Launch:  # pragma: no cover - interface
        raise NotImplementedError

    def parse(self, stdout_path: Path) -> dict[str, Any]:
        return {}


def _jsonl(path: Path):
    if not path.exists():
        return
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("{"):
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


class ClaudeCode(Harness):
    name = "claude-code"
    binary = "claude"

    def launch(self, prompt, workspace, *, model, effort, budget_usd, isolated):
        cmd = ["claude", "-p", "--output-format", "stream-json", "--verbose",
               "--permission-mode", "bypassPermissions",
               "--disallowedTools", "WebFetch", "WebSearch",
               "--strict-mcp-config", "--no-session-persistence"]
        if isolated:
            # sandboxed runs: no user settings / hooks / CLAUDE.md discovery; auth via ANTHROPIC_API_KEY
            cmd.append("--bare")
        else:
            # local runs keep the operator's login but must not see their CLAUDE.md, ~/.claude/rules, skills,
            # plugins or hooks (operator instructions would contaminate the benchmark)
            cmd.append("--safe-mode")
        if model:
            cmd += ["--model", model]
        if effort:
            cmd += ["--effort", effort]
        if budget_usd:
            cmd += ["--max-budget-usd", f"{budget_usd:g}"]
        return Launch(cmd, {"CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}, stdin=prompt)

    def parse(self, stdout_path):
        out: dict[str, Any] = {"turns": 0, "usage": {}}
        usage: dict[str, float] = {}
        models: set[str] = set()
        # fallback when the agent is killed mid-turn (no final "result" event): per-message usage + the in-flight
        # thinking estimate, so a cut-off episode still accounts for the tokens it spent (marked estimated)
        per_msg: dict[str, dict[str, int]] = {}
        inflight = 0
        for ev in _jsonl(stdout_path):
            if ev.get("type") == "system" and ev.get("subtype") == "thinking_tokens":
                inflight = max(inflight, int(ev.get("estimated_tokens") or 0))
            if ev.get("type") == "assistant":
                out["turns"] += 1
                msg = ev.get("message") or {}
                if msg.get("model"):
                    models.add(msg["model"])
                mu = msg.get("usage") or {}
                acc = per_msg.setdefault(str(msg.get("id")), {})
                for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
                    acc[k] = max(acc.get(k, 0), int(mu.get(k) or 0))
                if msg.get("stop_reason"):
                    inflight = 0
            elif ev.get("type") == "system" and ev.get("subtype") == "init":
                out["harness_version"] = ev.get("claude_code_version") or ev.get("version")
                if ev.get("model"):
                    models.add(ev["model"])
            elif ev.get("type") == "result":
                u = ev.get("usage") or {}
                usage = {"in_tokens": u.get("input_tokens", 0), "out_tokens": u.get("output_tokens", 0),
                         "cache_read_tokens": u.get("cache_read_input_tokens", 0),
                         "cache_write_tokens": u.get("cache_creation_input_tokens", 0)}
                if ev.get("total_cost_usd") is not None:
                    usage["cost_usd"] = float(ev["total_cost_usd"])
                out["num_turns"] = ev.get("num_turns")
                out["final_message"] = str(ev.get("result") or "")[:4000]
                out["is_error"] = bool(ev.get("is_error"))
                out["stop"] = ev.get("subtype")
                for mname, mu in (ev.get("modelUsage") or {}).items():
                    models.add(mname)
        if not usage and (per_msg or inflight):
            tot = {k: sum(m.get(k, 0) for m in per_msg.values()) for k in
                   ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")}
            usage = {"in_tokens": tot["input_tokens"], "out_tokens": tot["output_tokens"] + inflight,
                     "cache_read_tokens": tot["cache_read_input_tokens"],
                     "cache_write_tokens": tot["cache_creation_input_tokens"], "estimated": True}
            if not per_msg:  # killed inside its very first thought: the prompt's tokens were never reported
                usage["input_unknown"] = True
        out["usage"] = usage
        out["served_models"] = sorted(models)
        return out


class Codex(Harness):
    name = "codex"

    @property
    def binary(self) -> str:  # RBENCH_CODEX_BIN: a specific Codex CLI build (new models need recent clients)
        return os.environ.get("RBENCH_CODEX_BIN", "codex")

    def launch(self, prompt, workspace, *, model, effort, budget_usd, isolated):
        # the user's own config.toml (default model, service tier, rules, MCP servers) must not leak into runs
        cmd = [self.binary, "exec", "--json", "--skip-git-repo-check", "--ephemeral", "--ignore-user-config",
               "--ignore-rules", "-C", str(workspace)]
        if isolated:
            # already inside an external sandbox (container, unprivileged user): no nested sandbox
            cmd += ["--dangerously-bypass-approvals-and-sandbox"]
        else:
            cmd += ["--sandbox", "workspace-write", "-c", "sandbox_workspace_write.network_access=true"]
        if model:
            cmd += ["-m", model]
        if effort:
            cmd += ["-c", f"model_reasoning_effort={effort}"]
        cmd.append("-")  # read the prompt from stdin
        return Launch(cmd, stdin=prompt)

    def parse(self, stdout_path):
        out: dict[str, Any] = {"turns": 0}
        usage: dict[str, float] = {"in_tokens": 0, "out_tokens": 0, "cache_read_tokens": 0, "reasoning_tokens": 0}
        last_msg = ""
        for ev in _jsonl(stdout_path):
            t = ev.get("type")
            if t == "turn.completed":
                u = ev.get("usage") or {}
                usage["in_tokens"] += int(u.get("input_tokens", 0) or 0)
                usage["cache_read_tokens"] += int(u.get("cached_input_tokens", 0) or 0)
                usage["out_tokens"] += int(u.get("output_tokens", 0) or 0)
                usage["reasoning_tokens"] += int(u.get("reasoning_output_tokens", 0) or 0)
            elif t == "item.completed":
                item = ev.get("item") or {}
                if item.get("type") in ("agent_message", "assistant_message"):
                    last_msg = str(item.get("text") or "")
                out["turns"] += 1 if item.get("type") in ("command_execution", "function_call", "mcp_tool_call") else 0
            elif t in ("turn.failed", "error"):
                out["error"] = str(ev.get("error") or ev.get("message") or ev)[:2000]
        out["usage"] = usage
        out["final_message"] = last_msg[:4000]
        return out


PROMPT_FILE = ".rbench_prompt.md"  # workspace file for CLIs that take the prompt from a file


def _int(v: Any) -> int:
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


class GeminiCLI(Harness):
    """Google's Gemini CLI. Settings go into the workspace's .gemini/settings.json: no web tools (parity with the
    other harnesses), no telemetry or update checks (the sandbox only reaches the model API), and the thinking level
    set explicitly - the CLI's own definition of some Gemini 3 models has thinking off."""
    name = "gemini-cli"
    binary = "gemini"

    def launch(self, prompt, workspace, *, model, effort, budget_usd, isolated):
        settings: dict[str, Any] = {
            "privacy": {"usageStatisticsEnabled": False},
            "general": {"enableAutoUpdate": False, "enableAutoUpdateNotification": False},
            "telemetry": {"enabled": False},
            "tools": {"exclude": ["web_fetch", "google_web_search"]},
        }
        if model and effort:  # Gemini 3 thinking levels top out at HIGH
            level = {"minimal": "MINIMAL", "low": "LOW", "medium": "MEDIUM"}.get(effort.lower(), "HIGH")
            settings["modelConfigs"] = {"customOverrides": [{"match": {"model": model}, "modelConfig": {
                "generateContentConfig": {"thinkingConfig": {"thinkingLevel": level}}}}]}
        # -p is appended to whatever arrives on stdin: the brief comes on stdin, off the command line
        cmd = ["gemini", "--output-format", "stream-json", "--approval-mode", "yolo", "--skip-trust",
               "-p", "Carry out the task described above."]
        if model:
            cmd += ["-m", model]
        return Launch(cmd, {"GEMINI_TELEMETRY_ENABLED": "false"}, stdin=prompt,
                      files={".gemini/settings.json": json.dumps(settings, indent=1)})

    def parse(self, stdout_path):
        out: dict[str, Any] = {"turns": 0, "usage": {}}
        models: set[str] = set()
        text: list[str] = []
        for ev in _jsonl(stdout_path):
            t = ev.get("type")
            if t == "init" and ev.get("model"):
                models.add(ev["model"])
            elif t == "tool_use":
                out["turns"] += 1
            elif t == "message" and ev.get("role") == "assistant":
                text.append(str(ev.get("content") or ""))
            elif t == "result":
                st = ev.get("stats") or {}
                inp, total = _int(st.get("input_tokens")), _int(st.get("total_tokens"))
                # output_tokens counts candidates only; thoughts (billed as output) are the rest of the total
                out["usage"] = {"in_tokens": inp, "out_tokens": max(_int(st.get("output_tokens")), total - inp),
                                "cache_read_tokens": _int(st.get("cached")), "cache_write_tokens": 0}
                models |= set((st.get("models") or {}).keys())
                if ev.get("status") == "error":
                    out["error"] = str(ev.get("error"))[:2000]
            elif t == "error":
                out["error"] = str(ev.get("message") or ev)[:2000]
        out["final_message"] = "".join(text)[-4000:]
        out["served_models"] = sorted(models)
        return out


class GrokBuild(Harness):
    """xAI's Grok Build CLI, headless: prompt from a workspace file. `--output-format streaming-messages-json` emits
    Claude-Code-style events (system / assistant / user / result) whose assistant messages carry per-call usage in
    Anthropic form; the result event's usage stays zero in this format, so usage is always totalled per message."""
    name = "grok-build"
    binary = "grok"

    def launch(self, prompt, workspace, *, model, effort, budget_usd, isolated):
        cmd = ["grok", "--prompt-file", str(Path(workspace) / PROMPT_FILE),
               "--output-format", "streaming-messages-json", "--always-approve",
               "--permission-mode", "bypassPermissions", "--disable-web-search", "--cwd", str(workspace)]
        if model:
            cmd += ["-m", model]
        if effort:
            cmd += ["--reasoning-effort", effort]
        return Launch(cmd, files={PROMPT_FILE: prompt})

    def parse(self, stdout_path):
        per_msg: dict[str, dict[str, int]] = {}
        models: set[str] = set()
        text = ""
        for ev in _jsonl(stdout_path):
            if ev.get("type") == "system" and ev.get("subtype") == "init" and ev.get("model"):
                models.add(ev["model"])
            elif ev.get("type") == "assistant":
                msg = ev.get("message") or {}
                if msg.get("model"):
                    models.add(msg["model"])
                mu = msg.get("usage") or {}
                acc = per_msg.setdefault(str(msg.get("id")), {})
                for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
                    acc[k] = max(acc.get(k, 0), _int(mu.get(k)))
                for block in msg.get("content") or []:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = str(block.get("text") or "")
        tot = {k: sum(m.get(k, 0) for m in per_msg.values()) for k in
               ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")}
        usage = {"in_tokens": tot["input_tokens"], "out_tokens": tot["output_tokens"],
                 "cache_read_tokens": tot["cache_read_input_tokens"],
                 "cache_write_tokens": tot["cache_creation_input_tokens"]} if per_msg else {}
        return {"turns": len(per_msg), "usage": usage, "served_models": sorted(models), "final_message": text[-4000:]}


class MuseCode(Harness):
    """Meta's Muse Code CLI, headless (`muse exec --json`): prompt from a workspace file. Its event log carries no
    token counts, so in isolated runs it runs behind the rbench-muse wrapper (part of the hosted runner, not included
    here), a loopback proxy that writes one receipt per Model API request to stderr (served model, token counts);
    usage and turns come from those. Without the wrapper (local runs), usage is collected from any event-log payload
    carrying token counts."""
    name = "muse-code"
    binary = "muse"
    _IN = ("input_tokens", "prompt_tokens")
    _OUT = ("output_tokens", "completion_tokens")
    _CACHE = ("cached_tokens", "cache_read_tokens", "cached_input_tokens", "cache_read_input_tokens")
    _REASON = ("reasoning_tokens", "reasoning_output_tokens")

    def launch(self, prompt, workspace, *, model, effort, budget_usd, isolated):
        cmd = ["muse", "exec", "--json", "--prompt-file", str(Path(workspace) / PROMPT_FILE), "--disable-web-tools",
               "--workspace", str(workspace)]
        if isolated:
            # already inside an external sandbox (container, unprivileged user): no nested sandbox. Its bubblewrap
            # sandbox cannot configure a loopback in a container, and the fallback (unsandboxed) needs a human approval
            cmd += ["--yolo"]
        else:
            cmd += ["--approval-mode", "never", "--approval-judge", "off"]
        if model:
            cmd += ["--model", model]
        if effort:
            cmd += ["--reasoning-effort", effort]
        if isolated:
            cmd = ["rbench-muse", "--", *cmd]  # meters each Model API request; adds --base-url itself
        return Launch(cmd, files={PROMPT_FILE: prompt})

    _CALLS = ("/responses", "/chat/completions")

    def _usage_dicts(self, obj: Any):
        if isinstance(obj, dict):
            if any(k in obj for k in self._IN + self._OUT):
                yield obj
            else:
                for v in obj.values():
                    yield from self._usage_dicts(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from self._usage_dicts(v)

    def parse(self, stdout_path):
        usage = {"in_tokens": 0, "out_tokens": 0, "cache_read_tokens": 0, "reasoning_tokens": 0}
        types: dict[str, int] = {}
        text: list[str] = []
        steps = 0
        for ev in _jsonl(stdout_path):
            pt = str(ev.get("payload_type") or "")
            types[pt] = types.get(pt, 0) + 1
            payload = ev.get("payload") or {}
            if pt == "run.output.delta":
                text.append(str(payload.get("text") or ""))
            if "usage" not in pt and "model" not in pt:
                continue
            for u in self._usage_dicts(payload):
                steps += 1
                usage["in_tokens"] += next((_int(u[k]) for k in self._IN if k in u), 0)
                usage["out_tokens"] += next((_int(u[k]) for k in self._OUT if k in u), 0)
                usage["cache_read_tokens"] += next((_int(u[k]) for k in self._CACHE if k in u), 0)
                usage["reasoning_tokens"] += next((_int(u[k]) for k in self._REASON if k in u), 0)
        out: dict[str, Any] = {"turns": steps, "usage": usage if steps else {}, "final_message": "".join(text)[-4000:],
                               "payload_types": types}
        receipts = _upstream_receipts(stdout_path)
        if not receipts:  # no rbench-muse wrapper (local runs)
            return out
        # Responses usage: input_tokens counts the cached ones and output_tokens the reasoning ones, as cost_usd expects
        calls = [r for r in receipts if r.get("method") == "POST"
                 and str(r.get("path") or "").rstrip("/").endswith(self._CALLS)]
        ok = [r for r in calls if r.get("status") == 200]
        u = [r.get("usage") or {} for r in ok]
        errors: dict[str, int] = {}
        for r in receipts:
            if r.get("status") != 200:
                key = str(r.get("status") or r.get("error") or "no response")
                errors[key] = errors.get(key, 0) + 1
        out.update(turns=len(ok), served_models=sorted({str(r.get("model")) for r in ok if r.get("model")}),
                   detail={"upstream_requests": len(receipts), "upstream_errors": errors,
                           "calls_without_usage": sum(1 for x in u if not x)})
        out["usage"] = {"in_tokens": sum(_int(x.get("input_tokens")) for x in u),
                        "out_tokens": sum(_int(x.get("output_tokens")) for x in u),
                        "cache_read_tokens": sum(_int(x.get("cached_tokens")) for x in u),
                        "reasoning_tokens": sum(_int(x.get("reasoning_tokens")) for x in u),
                        "cache_write_tokens": 0} if ok else {}
        if calls and not ok:  # every model call failed (key, access, quota): say why
            out["is_error"] = True
            out["final_message"] = (out["final_message"] or f"Meta API calls failed: {errors}")[-4000:]
        return out


UPSTREAM_RECEIPT = "RBENCH_UPSTREAM "  # stderr line prefix of the hosted runner wrappers' per-request receipts


def _upstream_receipts(stdout_path) -> list[dict]:
    """The rbench-antigravity / rbench-muse receipts in the agent's stderr log (next to its stdout log)."""
    receipts = []
    stderr = Path(stdout_path).with_name("agent_stderr.log")
    if stderr.exists():
        for line in stderr.read_text(errors="replace").splitlines():
            if line.startswith(UPSTREAM_RECEIPT):
                try:
                    receipts.append(json.loads(line[len(UPSTREAM_RECEIPT):]))
                except ValueError:
                    pass
    return receipts


class Antigravity(Harness):
    """Google's Antigravity CLI (`agy`), headless: the brief arrives as one stream-json user event on stdin, output is
    stream-json, every tool call is allowed, web tools are denied (deny rules win over always-proceed), no telemetry
    or update checks. agy runs only the models in its own catalog; any other model runs on a catalog carrier whose
    model calls the rbench-antigravity wrapper (part of the hosted runner, not included here) sends to the
    requested model. The wrapper writes one receipt per Gemini API request to stderr (served model, token counts);
    usage and turns come from those, as agy reports usage only in a final event that a CLI killed at the time limit
    never writes."""
    name = "antigravity"
    binary = "agy"
    CARRIER = "gemini-3.8-flash"
    CATALOG = ("gemini-3.8-flash",)  # run directly; anything else goes through the carrier
    _GENERATE = ("generateContent", "streamGenerateContent")

    def launch(self, prompt, workspace, *, model, effort, budget_usd, isolated):
        # Gemini's top thinking level is high; agy names catalog models <model>-<effort>
        level = "medium" if (effort or "high").lower() in ("minimal", "low", "medium") else "high"
        model = model or self.CARRIER
        carrier = model if model in self.CATALOG else self.CARRIER
        settings = {"modelProvider": "gemini", "toolPermission": "always-proceed", "enableTelemetry": False,
                    "permissions": {"deny": ["read_url(*)", "execute_url(*)"]}}
        agy = ["agy", "--input-format", "stream-json", "--output-format", "stream-json",
               "--model", f"{carrier}-{level}", "--effort", level, "--mode", "accept-edits",
               "--dangerously-skip-permissions", "--print-timeout", "1h"]
        cmd = ["rbench-antigravity", "--carrier", carrier] + (["--model", model] if model != carrier else [])
        env = {"AGY_RBENCH_SETTINGS": json.dumps(settings), "AGY_CLI_DISABLE_AUTO_UPDATE": "1", "NO_COLOR": "1"}
        return Launch(cmd + ["--", *agy], env,
                      stdin=json.dumps({"event": "user", "message": {"content": prompt}}) + "\n")

    @staticmethod
    def _tokens(receipts: list[dict]) -> dict[str, int]:
        # Gemini usageMetadata: promptTokenCount includes the cached tokens; thoughts are billed as output
        u = [r.get("usage") or {} for r in receipts]
        return {"in_tokens": sum(_int(x.get("promptTokenCount")) for x in u),
                "out_tokens": sum(_int(x.get("candidatesTokenCount")) + _int(x.get("thoughtsTokenCount")) for x in u),
                "cache_read_tokens": sum(_int(x.get("cachedContentTokenCount")) for x in u), "cache_write_tokens": 0}

    def parse(self, stdout_path):
        receipts = _upstream_receipts(stdout_path)
        calls = [r for r in receipts if str(r.get("path", "")).rsplit(":", 1)[-1] in self._GENERATE]
        main = [r for r in calls if r.get("scope") == "main"]
        ok = [r for r in main if r.get("status") == 200]
        aux: dict[str, list[dict]] = {}
        for r in calls:
            if r.get("scope") != "main":
                aux.setdefault(str(r.get("model")), []).append(r)
        errors: dict[str, int] = {}
        for r in receipts:
            if r.get("status") != 200:
                key = str(r.get("status") or r.get("error") or "no response")
                errors[key] = errors.get(key, 0) + 1
        # agy stream-json: {"event": "step_update", ...} per step, and a final {"event": "result", "result":
        # {"status", "response", "error", "num_turns", "usage"}} when it ends by itself
        events: dict[str, int] = {}
        text = ""
        cli: dict[str, Any] = {}
        for ev in _jsonl(stdout_path):
            et = str(ev.get("event") or ev.get("type") or "")
            events[et] = events.get(et, 0) + 1
            if et == "result" and isinstance(ev.get("result"), dict):
                cli = ev["result"]
                text = str(cli.get("response") or "") or text
            elif isinstance(ev.get("message"), dict) and ev["message"].get("role") in (None, "assistant"):
                text = str(ev["message"].get("content") or "") or text
        out: dict[str, Any] = {
            "turns": len(ok), "usage": self._tokens(ok) if ok else {},
            "served_models": sorted({str(r.get("model_version") or r.get("model")) for r in ok}),
            "final_message": text[-4000:],
            "detail": {"upstream_requests": len(receipts), "upstream_errors": errors, "event_types": events,
                       "auxiliary_usage": {m: self._tokens(rs) for m, rs in aux.items()},
                       "cli_result": {k: cli.get(k) for k in ("status", "num_turns", "usage", "error") if k in cli}}}
        if main and not ok:  # every model call failed (key, access, quota): say why
            out["is_error"] = True
            out["final_message"] = (out["final_message"] or str(cli.get("error") or "")
                                    or f"Gemini API calls failed: {errors}")[-4000:]
        return out


class KimiCode(Harness):
    """Moonshot's Kimi Code CLI, headless (`kimi -p ... --output-format stream-json`, which runs in its never-ask mode).
    Print mode takes no prompt on stdin, so the brief is the -p argument; the CLI renames its process ("kimi-code") as
    it starts, so on Linux the brief leaves /proc/<pid>/cmdline and an agent's `pkill -f` cannot match it. A
    per-episode KIMI_CODE_HOME inside the workspace holds config.toml (Moonshot's open platform with the key read from
    KIMI_API_KEY, the model at its effort, web tools denied, no telemetry, session titles or updates) and the session
    logs. stream-json carries no token counts: usage and turns come from the session wire logs, one step.end per model
    call (subagents' logs included)."""
    name = "kimi-code"
    binary = "kimi"
    HOME = ".kimi-code"
    BASE_URL = "https://api.moonshot.ai/v1"
    EFFORTS = ("low", "high", "max")  # Kimi K3.1 Pro: thinking is always on, max is the default and the top

    def __init__(self) -> None:
        self._home: Path | None = None

    def launch(self, prompt, workspace, *, model, effort, budget_usd, isolated):
        self._home = Path(workspace) / self.HOME
        effort = (effort or "max").lower()
        effort = effort if effort in self.EFFORTS else "max"
        deny = "\n".join(f'[[permission.rules]]\ndecision = "deny"\npattern = "{tool}"\nreason = "no web tools"\n'
                         for tool in ("WebSearch", "FetchURL"))
        config = (f'default_model = "rbench"\ntelemetry = false\nauto_session_title = false\n\n'
                  f'[providers.moonshot]\ntype = "kimi"\nbase_url = "{self.BASE_URL}"\napi_key_env = "KIMI_API_KEY"\n\n'
                  f'[models.rbench]\nprovider = "moonshot"\nmodel = {json.dumps(model or "kimi-k3.1-pro")}\n'
                  'max_context_size = 1048576\n'
                  'capabilities = ["thinking", "always_thinking", "image_in", "video_in", "tool_use"]\n'
                  f'support_efforts = {json.dumps(list(self.EFFORTS))}\ndefault_effort = "{effort}"\n\n'
                  f'[thinking]\nenabled = true\neffort = "{effort}"\n\n[upgrade]\nauto_install = false\n\n{deny}')
        env = {"KIMI_CODE_HOME": str(self._home), "KIMI_DISABLE_TELEMETRY": "1", "KIMI_CODE_NO_AUTO_UPDATE": "1",
               "NO_COLOR": "1"}
        return Launch(["kimi", "-p", prompt, "--output-format", "stream-json", "-m", "rbench"], env,
                      files={f"{self.HOME}/config.toml": config})

    def parse(self, stdout_path):
        text, version, tool_calls = "", None, 0
        for ev in _jsonl(stdout_path):
            if ev.get("type") == "system.version":
                version = ev.get("version")
            elif ev.get("role") == "assistant":
                text = str(ev.get("content") or "") or text
                tool_calls += len(ev.get("tool_calls") or [])
        usage = {"in_tokens": 0, "out_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0}
        steps, models = 0, set()
        wires = sorted(self._home.glob("sessions/*/*/agents/*/wire.jsonl")) if self._home else []
        for wire in wires:
            for ev in _jsonl(wire):
                if ev.get("type") == "llm.request" and ev.get("model"):
                    models.add(str(ev["model"]))
                step = ev.get("event") if ev.get("type") == "context.append_loop_event" else None
                if isinstance(step, dict) and step.get("type") == "step.end" and isinstance(step.get("usage"), dict):
                    # Kimi usage: inputOther excludes the cache reads and writes; output includes the reasoning
                    u = step["usage"]
                    steps += 1
                    usage["in_tokens"] += _int(u.get("inputOther"))
                    usage["out_tokens"] += _int(u.get("output"))
                    usage["cache_read_tokens"] += _int(u.get("inputCacheRead"))
                    usage["cache_write_tokens"] += _int(u.get("inputCacheCreation"))
        out: dict[str, Any] = {"turns": steps, "usage": usage if steps else {}, "served_models": sorted(models),
                               "final_message": text[-4000:], "harness_version": version,
                               "detail": {"tool_calls": tool_calls, "wire_logs": len(wires)}}
        stderr = Path(stdout_path).with_name("agent_stderr.log")
        errors = [ln for ln in stderr.read_text(errors="replace").splitlines()
                  if ln.startswith("error:")] if stderr.exists() else []
        if errors:
            out["error"] = errors[-1][:2000]
            if not steps:  # no model call went through (key, access, quota): say why
                out["is_error"] = True
                out["final_message"] = (text or errors[-1])[-4000:]
        return out


class OpenCode(Harness):
    """OpenCode, for models whose vendor ships no CLI fit for these sandboxes (DeepSeek's Harness is a developer preview
    that uploads session logs by default), headless: `opencode run --format json` with the brief on stdin. The
    workspace's opencode.json declares the provider (an OpenAI-compatible endpoint, key from the environment) and the
    model (image input, reasoning variants), so nothing comes from the models.dev catalog; there are no plugins,
    updates, sharing, LSP downloads, formatters or session titles, and web fetch is denied. Every step_finish event
    carries one model call's tokens: input without the cache reads, reasoning apart from the output."""
    name = "opencode"
    binary = "opencode"
    # provider id in the model key -> OpenAI-compatible base URL and the key variable
    PROVIDERS = {"fireworks": ("https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY")}
    LIMITS = {"accounts/fireworks/models/deepseek-v4p1-flash-1017": (1_048_576, 131_072)}  # context, output tokens
    EFFORTS = ("low", "high", "max")

    def launch(self, prompt, workspace, *, model, effort, budget_usd, isolated):
        provider, _, model_id = (model or "").partition("/")
        if provider not in self.PROVIDERS or not model_id:
            raise ValueError(f"opencode needs <provider>/<model>, provider one of {sorted(self.PROVIDERS)}: {model!r}")
        base_url, key_var = self.PROVIDERS[provider]
        context, output = self.LIMITS.get(model_id, (1_000_000, 131_072))
        config = {
            "$schema": "https://opencode.ai/config.json", "autoupdate": False, "share": "disabled",
            "formatter": False, "lsp": False, "permission": {"webfetch": "deny"},
            "provider": {provider: {
                "npm": "@ai-sdk/openai-compatible", "name": provider,
                "options": {"baseURL": base_url, "apiKey": "{env:" + key_var + "}"},
                "models": {model_id: {
                    "name": model_id, "reasoning": True, "attachment": True, "tool_call": True,
                    "modalities": {"input": ["text", "image"], "output": ["text"]},
                    "limit": {"context": context, "output": output},
                    "variants": {e: {"reasoningEffort": e} for e in self.EFFORTS}}}}},
        }
        cmd = ["opencode", "run", "--pure", "-m", f"{provider}/{model_id}", "--format", "json", "--auto",
               "--title", "RoboticsBench episode"]
        if effort and effort.lower() in self.EFFORTS:
            cmd += ["--variant", effort.lower()]
        env = {"OPENCODE_DISABLE_AUTOUPDATE": "1", "OPENCODE_DISABLE_MODELS_FETCH": "1",
               "OPENCODE_DISABLE_LSP_DOWNLOAD": "1", "OPENCODE_DISABLE_DEFAULT_PLUGINS": "1", "NO_COLOR": "1"}
        return Launch(cmd, env, stdin=prompt, files={"opencode.json": json.dumps(config, indent=1)})

    def parse(self, stdout_path):
        usage = {"in_tokens": 0, "out_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0,
                 "reasoning_tokens": 0}
        steps, text, errors, tool_calls = 0, "", [], 0
        for ev in _jsonl(stdout_path):
            t, part = ev.get("type"), ev.get("part") or {}
            if t == "step_finish":
                tok = part.get("tokens") or {}
                cache = tok.get("cache") or {}
                steps += 1
                usage["in_tokens"] += _int(tok.get("input"))
                usage["out_tokens"] += _int(tok.get("output")) + _int(tok.get("reasoning"))  # reasoning bills as output
                usage["reasoning_tokens"] += _int(tok.get("reasoning"))
                usage["cache_read_tokens"] += _int(cache.get("read"))
                usage["cache_write_tokens"] += _int(cache.get("write"))
            elif t == "text" and part.get("text"):
                text = str(part["text"])
            elif t == "tool_use":
                tool_calls += 1
            elif t == "error":
                err = ev.get("error") or {}
                data = err.get("data") or {}
                errors.append(" ".join(str(x) for x in (err.get("name"), data.get("statusCode"), data.get("message"))
                                       if x not in (None, "")))
        out: dict[str, Any] = {"turns": steps, "usage": usage if steps else {}, "final_message": text[-4000:],
                               "detail": {"tool_calls": tool_calls}}
        if errors:
            out["error"] = errors[-1][:2000]
            if not steps:  # no model call went through (key, access, quota): say why
                out["is_error"] = True
                out["final_message"] = (text or errors[-1])[-4000:]
        return out


class CommandTemplate(Harness):
    name = "cmd"

    def __init__(self, template: str):
        self.template = template
        self.binary = shlex.split(template)[0] if template.strip() else ""

    def launch(self, prompt, workspace, *, model, effort, budget_usd, isolated):
        subs = {"{prompt}": prompt, "{model}": model or "", "{effort}": effort or "", "{workspace}": str(workspace)}
        parts = []
        for tok in shlex.split(self.template):
            for key, val in subs.items():  # plain replacement: other braces in the command stay literal
                tok = tok.replace(key, val)
            parts.append(tok)
        return Launch([p for p in parts if p != ""])


HARNESSES: dict[str, type[Harness]] = {"claude-code": ClaudeCode, "codex": Codex, "gemini-cli": GeminiCLI,
                                        "antigravity": Antigravity, "grok-build": GrokBuild, "muse-code": MuseCode,
                                        "kimi-code": KimiCode, "opencode": OpenCode}


def make_harness(spec: str) -> Harness:
    if spec.startswith("cmd:"):
        return CommandTemplate(spec[4:])
    if spec not in HARNESSES:
        raise KeyError(f"unknown harness {spec!r}; known: {sorted(HARNESSES)} or cmd:<template>")
    return HARNESSES[spec]()


_HOST_PREFIXES = ("CLAUDE", "MCP_", "RBENCH_")  # host-agent session plumbing and RoboticsBench operator secrets


def child_env(extra: dict[str, str]) -> dict[str, str]:
    """Environment for the agent process: the caller's env minus RoboticsBench secrets and the variables of any
    coding-agent session that launched the benchmark (so the benchmarked agent is a clean, independent session)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(_HOST_PREFIXES)}
    env.update(extra)
    return env
