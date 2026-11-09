"""Usage parsing for the Gemini CLI, Antigravity, Grok Build, Muse Code, Kimi Code and OpenCode event streams."""

import json

from roboticsbench.harness.agents import UPSTREAM_RECEIPT, Antigravity, GeminiCLI, GrokBuild, MuseCode


def _write(path, events):
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return path


def test_gemini_result_stats_count_thoughts_as_output(tmp_path):
    f = _write(tmp_path / "g.jsonl", [
        {"type": "init", "model": "gemini-3.8-flash"},
        {"type": "tool_use", "tool_name": "run_shell_command"},
        {"type": "message", "role": "assistant", "content": "done"},
        {"type": "result", "status": "success", "stats": {"total_tokens": 1500, "input_tokens": 1000,
                                                           "output_tokens": 200, "cached": 400, "input": 600,
                                                           "models": {"gemini-3.8-flash": {}}}},
    ])
    out = GeminiCLI().parse(f)
    assert out["turns"] == 1 and out["final_message"] == "done"
    assert out["usage"] == {"in_tokens": 1000, "out_tokens": 500, "cache_read_tokens": 400, "cache_write_tokens": 0}


def test_grok_assistant_message_usage(tmp_path):
    """The format Grok Build 1.0.50 really emits (2026-10-12 check run): Claude-Code-style events."""
    f = _write(tmp_path / "x.jsonl", [
        {"type": "system", "subtype": "init", "model": "grok-4.7"},
        {"type": "assistant", "message": {"id": "msg_0", "model": "grok-4.7", "content": [
            {"type": "thinking", "thinking": "..."}, {"type": "text", "text": "reading the docs"}],
            "usage": {"input_tokens": 900, "output_tokens": 120, "cache_read_input_tokens": 3000,
                      "cache_creation_input_tokens": 0}}},
        {"type": "user", "message": {"content": [{"type": "tool_result"}]}},
        {"type": "assistant", "message": {"id": "msg_1", "usage": {"input_tokens": 50, "output_tokens": 30}}},
        {"type": "result", "usage": {"input_tokens": 0, "output_tokens": 0}, "total_cost_usd": 0.0},
    ])
    out = GrokBuild().parse(f)
    assert out["turns"] == 2 and out["served_models"] == ["grok-4.7"] and out["final_message"] == "reading the docs"
    assert out["usage"] == {"in_tokens": 950, "out_tokens": 150, "cache_read_tokens": 3000, "cache_write_tokens": 0}


def test_muse_usage_from_model_payloads(tmp_path):
    f = _write(tmp_path / "m.jsonl", [
        {"record_type": "event", "payload_type": "run.output.delta", "payload": {"text": "ok"}},
        {"record_type": "event", "payload_type": "model.response.completed",
         "payload": {"response": {"id": "r1", "usage": {"input_tokens": 700, "output_tokens": 90,
                                                        "cached_tokens": 200, "reasoning_tokens": 40}}}},
        {"record_type": "event", "payload_type": "task.lifecycle.completed", "payload": {"input_tokens": 5}},
    ])
    out = MuseCode().parse(f)
    assert out["usage"] == {"in_tokens": 700, "out_tokens": 90, "cache_read_tokens": 200, "reasoning_tokens": 40}
    assert out["final_message"] == "ok" and out["payload_types"]["model.response.completed"] == 1


def test_muse_usage_comes_from_the_wrapper_receipts(tmp_path):
    """Muse Code's event log has no token counts: usage, turns and the served model come from the rbench-muse
    receipts in agent_stderr.log (Responses semantics: cached inside input, reasoning inside output)."""
    out_f = _write(tmp_path / "agent_stdout.jsonl", [
        {"record_type": "event", "payload_type": "run.output.delta", "payload": {"text": "pressing B2"}},
    ])
    receipts = [
        {"method": "GET", "path": "/v1/models", "status": 200},
        {"method": "POST", "path": "/v1/responses", "status": 200, "model": "muse_spark_1_3",
         "usage": {"input_tokens": 1000, "cached_tokens": 400, "output_tokens": 300, "reasoning_tokens": 200}},
        {"method": "POST", "path": "/v1/responses", "status": 200, "model": "muse_spark_1_3",
         "usage": {"input_tokens": 2000, "cached_tokens": 1600, "output_tokens": 100, "reasoning_tokens": 20}},
        {"method": "POST", "path": "/v1/responses", "status": 200},  # stream cut at the time limit: no usage
        {"method": "POST", "path": "/v1/responses", "status": 503},
    ]
    (tmp_path / "agent_stderr.log").write_text("muse starting\n" + "".join(
        UPSTREAM_RECEIPT + json.dumps(r) + "\n" for r in receipts))
    out = MuseCode().parse(out_f)
    assert out["turns"] == 3 and out["served_models"] == ["muse_spark_1_3"]
    assert out["usage"] == {"in_tokens": 3000, "out_tokens": 400, "cache_read_tokens": 2000, "reasoning_tokens": 220,
                            "cache_write_tokens": 0}
    assert out["detail"]["upstream_errors"] == {"503": 1} and out["detail"]["calls_without_usage"] == 1
    assert out["final_message"] == "pressing B2" and not out.get("is_error")


def test_muse_says_why_when_every_model_call_fails(tmp_path):
    out_f = _write(tmp_path / "agent_stdout.jsonl", [])
    (tmp_path / "agent_stderr.log").write_text(UPSTREAM_RECEIPT + json.dumps(
        {"method": "POST", "path": "/v1/responses", "status": 404}) + "\n")
    out = MuseCode().parse(out_f)
    assert out["is_error"] and "404" in out["final_message"] and out["usage"] == {}


def test_antigravity_usage_comes_from_the_wrapper_receipts(tmp_path):
    """agy was killed at the time limit (no result event): usage, turns and the served model come from the
    rbench-antigravity receipts in agent_stderr.log; auxiliary models are kept apart."""
    out_f = _write(tmp_path / "agent_stdout.jsonl", [
        {"event": "init", "model": "gemini-3.8-flash-high"},
        {"event": "message", "message": {"role": "assistant", "content": "reading ROBOT.md"}},
    ])
    gen = "/v1beta/models/gemini-4-argon:streamGenerateContent"
    receipts = [
        {"method": "POST", "path": gen, "model": "gemini-4-argon", "scope": "main", "status": 200, "model_version": "gemini-4-argon",
         "usage": {"promptTokenCount": 1000, "cachedContentTokenCount": 400, "candidatesTokenCount": 50,
                   "thoughtsTokenCount": 150}},
        {"method": "POST", "path": gen, "model": "gemini-4-argon", "scope": "main", "status": 200, "model_version": "gemini-4-argon",
         "usage": {"promptTokenCount": 1200, "candidatesTokenCount": 20}},
        {"method": "POST", "path": gen, "model": "gemini-4-argon", "scope": "main", "status": 429},
        {"method": "POST", "path": "/v1beta/models/gemini-3.1-flash-lite:generateContent", "model": "gemini-3.1-flash-lite",
         "scope": "auxiliary", "status": 200, "usage": {"promptTokenCount": 300, "candidatesTokenCount": 10}},
        {"method": "GET", "path": "/v1beta/models/gemini-3.8-flash-high", "scope": "main", "status": 200},
    ]
    (tmp_path / "agent_stderr.log").write_text("agy starting\n" + "".join(
        UPSTREAM_RECEIPT + json.dumps(r) + "\n" for r in receipts))
    out = Antigravity().parse(out_f)
    assert out["turns"] == 2 and out["served_models"] == ["gemini-4-argon"] and out["final_message"] == "reading ROBOT.md"
    assert out["usage"] == {"in_tokens": 2200, "out_tokens": 220, "cache_read_tokens": 400, "cache_write_tokens": 0}
    assert out["detail"]["upstream_errors"] == {"429": 1}
    assert out["detail"]["auxiliary_usage"]["gemini-3.1-flash-lite"]["in_tokens"] == 300
    assert "is_error" not in out


def test_antigravity_flags_an_episode_whose_model_calls_all_failed(tmp_path):
    """The agy 1.3.3 stream of the fake-key smoke run (2026-10-12): the error reaches the infra-error message."""
    out_f = _write(tmp_path / "agent_stdout.jsonl", [
        {"event": "step_update", "step_update": {"step_index": 0, "state": "DONE", "step_type": "user_input"}},
        {"event": "step_update", "step_update": {"step_index": 1, "state": "DONE", "step_type": "error_message"}},
        {"event": "result", "result": {"status": "ERROR", "response": "", "num_turns": 1,
                                       "error": "Error 400, Message: API key not valid.",
                                       "usage": {"input_tokens": 0, "output_tokens": 0}}},
    ])
    (tmp_path / "agent_stderr.log").write_text(UPSTREAM_RECEIPT + json.dumps(
        {"path": "/v1beta/models/gemini-4-argon:streamGenerateContent", "scope": "main", "status": 400}) + "\n")
    out = Antigravity().parse(out_f)
    assert out["is_error"] and out["turns"] == 0 and out["usage"] == {}
    assert "API key not valid" in out["final_message"] and out["detail"]["cli_result"]["status"] == "ERROR"
    assert out["detail"]["upstream_errors"] == {"400": 1} and out["detail"]["event_types"]["step_update"] == 2


def test_antigravity_launch_routes_models_outside_its_catalog_through_the_carrier():
    from pathlib import Path

    launch = Antigravity().launch("brief", Path("/w"), model="gemini-4-argon", effort="max", budget_usd=None, isolated=True)
    assert launch.cmd[:5] == ["rbench-antigravity", "--carrier", "gemini-3.8-flash", "--model", "gemini-4-argon"]
    agy = launch.cmd[launch.cmd.index("--") + 1:]
    assert agy[0] == "agy" and agy[agy.index("--model") + 1] == "gemini-3.8-flash-high"
    assert agy[agy.index("--effort") + 1] == "high" and "--dangerously-skip-permissions" in agy
    assert json.loads(launch.stdin) == {"event": "user", "message": {"content": "brief"}}
    settings = json.loads(launch.env["AGY_RBENCH_SETTINGS"])
    assert settings["permissions"]["deny"] == ["read_url(*)", "execute_url(*)"] and not settings["enableTelemetry"]
    native = Antigravity().launch("brief", Path("/w"), model="gemini-3.8-flash", effort="high", budget_usd=None,
                                  isolated=True)
    assert "--model" not in native.cmd[:native.cmd.index("--")]


def test_kimi_code_usage_comes_from_the_session_wire_logs(tmp_path):
    """Kimi Code 2.3.1: stream-json has no token counts; each model call ends with a step.end carrying its usage in
    the session's wire log (the shapes a run against a mock Moonshot API wrote, 2026-10-24)."""
    from pathlib import Path

    from roboticsbench.harness.agents import KimiCode

    h = KimiCode()
    h.launch("p", tmp_path, model="kimi-k3.1-pro", effort="max", budget_usd=None, isolated=True)
    out = _write(tmp_path / "agent_stdout.jsonl", [
        {"role": "meta", "type": "system.version", "version": "2.3.1"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "Bash"}}]},
        {"role": "tool", "content": "ok", "tool_call_id": "c1"},
        {"role": "assistant", "content": "Done."},
    ])

    def step(n, other, output, read, write=0):
        return {"type": "context.append_loop_event", "agentId": "main", "event": {
            "type": "step.end", "step": n, "usage": {"inputOther": other, "output": output, "inputCacheRead": read,
                                                       "inputCacheCreation": write}}}

    for agent, events in (("main", [{"type": "llm.request", "model": "kimi-k3.1-pro"}, step(1, 176, 30, 1024, 50),
                                    {"type": "usage.record", "usage": {"inputOther": 176, "output": 30}},
                                    step(2, 40, 300, 1200)]),
                          ("sub-1", [step(1, 10, 5, 0)])):
        wire = Path(tmp_path) / ".kimi-code/sessions/wd_ws_1/session_1/agents" / agent / "wire.jsonl"
        wire.parent.mkdir(parents=True)
        _write(wire, events)
    r = h.parse(out)
    assert r["turns"] == 3 and r["final_message"] == "Done." and r["served_models"] == ["kimi-k3.1-pro"]
    assert r["harness_version"] == "2.3.1" and r["detail"] == {"tool_calls": 1, "wire_logs": 2}
    assert r["usage"] == {"in_tokens": 226, "out_tokens": 335, "cache_read_tokens": 2224, "cache_write_tokens": 50}
    assert "is_error" not in r


def test_kimi_code_says_why_when_no_model_call_went_through(tmp_path):
    from roboticsbench.harness.agents import KimiCode

    h = KimiCode()
    h.launch("p", tmp_path, model="kimi-k3.1-pro", effort="max", budget_usd=None, isolated=True)
    out = _write(tmp_path / "agent_stdout.jsonl", [{"role": "meta", "type": "system.version", "version": "2.3.1"}])
    (tmp_path / "agent_stderr.log").write_text("error: failed to run prompt: provider.auth_error: 401 Invalid "
                                               "Authentication\nSee log: /home/player/.kimi-code/logs/kimi-code.log\n")
    r = h.parse(out)
    assert r["is_error"] and r["usage"] == {} and "401 Invalid Authentication" in r["final_message"]


def test_opencode_step_finish_tokens(tmp_path):
    """OpenCode 1.19.4 `--format json` (a run against a mock OpenAI-compatible API, 2026-10-24): input excludes the
    cache reads; reasoning is reported apart from output and billed as output."""
    from roboticsbench.harness.agents import OpenCode

    def finish(inp, output, reasoning, read):
        return {"type": "step_finish", "part": {"type": "step-finish", "reason": "stop", "tokens": {
            "total": inp + output + reasoning + read, "input": inp, "output": output, "reasoning": reasoning,
            "cache": {"write": 0, "read": read}}, "cost": 0}}

    f = _write(tmp_path / "agent_stdout.jsonl", [
        {"type": "step_start", "part": {"type": "step-start"}},
        {"type": "tool_use", "part": {"type": "tool", "tool": "bash"}},
        finish(176, 10, 20, 1024),
        {"type": "text", "part": {"type": "text", "text": "Done."}},
        finish(50, 5, 0, 1200),
    ])
    r = OpenCode().parse(f)
    assert r["turns"] == 2 and r["final_message"] == "Done." and r["detail"] == {"tool_calls": 1}
    assert r["usage"] == {"in_tokens": 226, "out_tokens": 35, "cache_read_tokens": 2224, "cache_write_tokens": 0,
                          "reasoning_tokens": 20}


def test_opencode_says_why_when_no_model_call_went_through(tmp_path):
    from roboticsbench.harness.agents import OpenCode

    f = _write(tmp_path / "agent_stdout.jsonl", [{"type": "error", "error": {"name": "APIError", "data": {
        "message": "Invalid Authentication", "statusCode": 401, "isRetryable": False}}}])
    r = OpenCode().parse(f)
    assert r["is_error"] and r["usage"] == {} and r["final_message"] == "APIError 401 Invalid Authentication"
