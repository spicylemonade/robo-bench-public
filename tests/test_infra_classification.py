"""A model that never acts before the clock runs out failed the task; only a CLI that crashes on its own is an infra
error (2026-10-09: Opus killed mid-thought was reported as infrastructure trouble)."""

import sys

from roboticsbench.harness.run import run_agent_episode

KW = dict(control="motor", vision_only=True, realtime=True, time_limit_s=2.0, grace_s=1.0, wall_limit_s=60.0,
          python=sys.executable, log=lambda *a: None)


def test_agent_stopped_while_thinking_is_a_model_failure(tmp_path):
    res = run_agent_episode("arm/balance_scale", 0, "cmd:sleep 30", out_dir=tmp_path, **KW)
    assert res["agent_stop_reason"].startswith("episode ended")
    assert res["infra_ok"] and res["error"] is None and res["status"] != "error"


def test_agent_crashing_on_its_own_is_an_infra_error(tmp_path):
    res = run_agent_episode("arm/balance_scale", 0, "cmd:false", out_dir=tmp_path, **KW)
    assert res["agent_stop_reason"].startswith("agent exited")
    assert not res["infra_ok"] and "made no robot calls" in res["error"] and res["status"] == "error"
