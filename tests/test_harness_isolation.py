"""Benchmarked coding agents must never see the operator's own instructions (CLAUDE.md, ~/.claude/rules,
Codex config/rules): those would contaminate the run."""

from pathlib import Path

from roboticsbench.harness.agents import ClaudeCode, Codex

KW = dict(model="m", effort="high", budget_usd=None)


def test_claude_code_local_runs_use_safe_mode():
    cmd = ClaudeCode().launch("go", Path("/tmp/ws"), isolated=False, **KW).cmd
    assert "--safe-mode" in cmd and "--bare" not in cmd


def test_claude_code_sandboxed_runs_use_bare():
    cmd = ClaudeCode().launch("go", Path("/tmp/ws"), isolated=True, **KW).cmd
    assert "--bare" in cmd


def test_codex_ignores_user_config_and_rules():
    for isolated in (False, True):
        cmd = Codex().launch("go", Path("/tmp/ws"), isolated=isolated, **KW).cmd
        assert "--ignore-user-config" in cmd and "--ignore-rules" in cmd


def test_agent_prompts_stay_off_the_command_line():
    """An agent's `pkill -f <script>` matched its own prompt in argv and killed it (2026-10-09)."""
    from pathlib import Path

    from roboticsbench.harness.agents import Antigravity, ClaudeCode, Codex, GeminiCLI, GrokBuild, MuseCode, OpenCode

    prompt = "control loop: run python walk.py and keep the robot standing"
    for h, model in ((ClaudeCode(), "m"), (Codex(), "m"), (GeminiCLI(), "m"), (Antigravity(), "m"),
                     (GrokBuild(), "m"), (MuseCode(), "m"),
                     (OpenCode(), "fireworks/accounts/fireworks/models/deepseek-v4p1-flash-1017")):
        launch = h.launch(prompt, Path("/tmp"), model=model, effort="max", budget_usd=1.0, isolated=True)
        assert any(prompt in s for s in (launch.stdin or "", *launch.files.values())), h.name
        assert not any("walk.py" in a for a in launch.cmd), h.name


def test_kimi_code_takes_the_brief_as_its_print_mode_argument():
    """Kimi Code's print mode reads no stdin (`-p -` sends "-") and expands no @file mention, so the brief is the -p
    argument. The CLI sets process.title = "kimi-code" as it starts, which on Linux overwrites the whole argv block:
    /proc/<pid>/cmdline, and so an agent's `pkill -f`, no longer shows the brief. The lifecycle smoke checks it."""
    from pathlib import Path

    from roboticsbench.harness.agents import KimiCode

    launch = KimiCode().launch("brief", Path("/tmp/ws"), model="kimi-k3.1-pro", effort="max", budget_usd=1.0,
                               isolated=True)
    assert launch.cmd[:3] == ["kimi", "-p", "brief"] and launch.stdin is None


def test_new_harnesses_keep_keys_out_of_files_and_web_tools_off():
    from pathlib import Path

    from roboticsbench.harness.agents import KimiCode, OpenCode

    kimi = KimiCode().launch("p", Path("/tmp/ws"), model="kimi-k3.1-pro", effort="max", budget_usd=None,
                             isolated=True)
    config = kimi.files[".kimi-code/config.toml"]
    assert 'api_key_env = "KIMI_API_KEY"' in config and "api_key =" not in config
    assert 'model = "kimi-k3.1-pro"' in config and 'effort = "max"' in config and "telemetry = false" in config
    assert config.count('decision = "deny"') == 2 and '"WebSearch"' in config and '"FetchURL"' in config
    assert kimi.env["KIMI_CODE_HOME"] == "/tmp/ws/.kimi-code" and kimi.env["KIMI_CODE_NO_AUTO_UPDATE"] == "1"
    oc = OpenCode().launch("p", Path("/tmp/ws"), model="fireworks/accounts/fireworks/models/deepseek-v4p1-flash-1017",
                           effort="max", budget_usd=None, isolated=True)
    import json
    cfg = json.loads(oc.files["opencode.json"])
    provider = cfg["provider"]["fireworks"]
    assert provider["options"] == {"baseURL": "https://api.fireworks.ai/inference/v1",
                                   "apiKey": "{env:FIREWORKS_API_KEY}"}
    model = provider["models"]["accounts/fireworks/models/deepseek-v4p1-flash-1017"]
    assert model["modalities"]["input"] == ["text", "image"] and model["variants"]["max"] == {"reasoningEffort": "max"}
    assert cfg["permission"] == {"webfetch": "deny"} and cfg["autoupdate"] is False and cfg["share"] == "disabled"
    assert oc.cmd[-2:] == ["--variant", "max"] and "--pure" in oc.cmd and "--title" in oc.cmd


def test_muse_code_skips_its_own_sandbox_when_isolated():
    """Muse Code's bubblewrap sandbox cannot set up a loopback in a container: every shell call failed (2026-10-12)."""
    from pathlib import Path

    from roboticsbench.harness.agents import MuseCode

    isolated = MuseCode().launch("p", Path("/tmp"), model="m", effort="max", budget_usd=1.0, isolated=True).cmd
    assert "--yolo" in isolated and "--approval-mode" not in isolated
    # behind the bundle's metering proxy, which adds --base-url itself
    assert isolated[:3] == ["rbench-muse", "--", "muse"] and "--base-url" not in isolated
    local = MuseCode().launch("p", Path("/tmp"), model="m", effort="max", budget_usd=1.0, isolated=False).cmd
    assert "--yolo" not in local and "--approval-mode" in local
