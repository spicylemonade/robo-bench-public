"""Real-time referee: the free-running clock thread records video while another thread asks for camera images.

Regression: GL contexts belong to one thread (CGL keeps them locked, EGL refuses them elsewhere), so rendering from
the clock thread after the HTTP thread had rendered froze the whole referee on the agent's first motor command.
"""

import threading

import pytest

from roboticsbench.harness.referee import Referee


def test_realtime_clock_renders_while_agent_looks(tmp_path):
    ref = Referee("humanoid/ninja_course", 0, tmp_path / "ep", video=True, vision_only=True, control="motor",
                  realtime=True)
    got: dict = {}

    def agent() -> None:  # what the HTTP handlers do, from another thread
        with ref.lock:
            ref.start_clock()
            got["t0"] = ref.state()["sensors"]["time"]
        with ref.lock:
            finish = ref.look_begin(["head"], 160, 120, depth=True)
        got["look"] = finish()
        with ref.lock:
            got["t1"] = ref.step(None, 10)["sensors"]["time"]

    th = threading.Thread(target=agent, daemon=True)
    th.start()
    th.join(timeout=60)
    if th.is_alive():  # leave the stuck (daemon) threads alone: finalize() would wait on them forever
        pytest.fail("real-time referee deadlocked")
    try:
        assert got["t1"] - got["t0"] >= 0.2 - 1e-6
        assert set(got["look"]["images"]) == {"head"}
        assert set(got["look"]["depth_png_mm"]) == {"head"}
        res = ref.finalize()
        assert res["status"] == "incomplete" and res["realtime"]
        assert (tmp_path / "ep" / "referee_result.json").exists() and (tmp_path / "ep" / "video.mp4").exists()
    finally:
        ref.finalize()
        ref.env.close()


def test_time_limit_counts_from_clock_start(tmp_path):
    """A per-run budget replaces the task's limit; once started (the orchestrator starts the clock as it launches the
    agent) the world runs to the limit with nobody touching the robot - and the robot stays standing meanwhile."""
    ref = Referee("humanoid/ninja_course", 0, tmp_path / "ep", control="motor", realtime=True, time_limit=1.0)
    assert ref.info()["time_limit_s"] == 1.0
    with ref.lock:
        ref.start_clock()
    deadline = threading.Event()
    deadline.wait(4.0)
    try:
        assert not ref.env.running and ref.env.status == "timeout"
        assert abs(ref.env.t - 1.0) < 0.05
        assert ref.result is not None and ref.result["status"] == "timeout"
    finally:
        ref.finalize()
        ref.env.close()
