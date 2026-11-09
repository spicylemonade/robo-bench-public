"""robot.py, the live robot module in the agent's workspace: senses streamed every tick, commands, control loops
(with hot reload) and cameras - against a real referee HTTP server in real-time mode."""

import threading
import time
from http.server import ThreadingHTTPServer

import pytest

from roboticsbench.harness import robot_live
from roboticsbench.harness.referee import Referee, make_handler


@pytest.fixture()
def live(tmp_path):
    ref = Referee("humanoid/ninja_course", 0, tmp_path / "ep", control="motor", realtime=True, time_limit=60)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(ref, "op"))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    with ref.lock:
        ref.start_clock()
    r = robot_live.Robot(f"http://127.0.0.1:{srv.server_address[1]}")
    r.views_dir = tmp_path / "views"
    yield r
    srv.shutdown()
    ref.finalize()
    ref.env.close()


def test_senses_stream_and_commands(live):
    r = live
    t0 = r.t
    s = r.wait(5)
    assert r.t >= t0 + 5 * 0.02 - 1e-9 and r.running and r.time_left > 0
    assert s.joint_positions.shape == (29,) and s.imu_pelvis.quaternion.shape == (4,)
    assert len(s.foot_contact_forces) == 2 and r.joint_index["left_knee_joint"] == 3
    time.sleep(0.2)  # the background stream keeps the state current without any call
    assert r.t >= s.time + 0.1
    r.send({"joints": {"left_shoulder_pitch_joint": {"q": -0.4}}})  # raise the left arm a little
    r.wait(30)
    assert r.state.joint_positions[15] < -0.2
    with pytest.raises(robot_live.RobotError):
        r.send({"bogus": 1})
    v = r.see(depth=True, width=160, height=120)
    assert v.image.shape == (120, 160, 3) and v.depth.shape == (120, 160) and v.calib["camera"] == "head"


def test_loop_runs_every_tick_and_hot_reloads(live, tmp_path):
    r = live
    calls = []
    r.loop(lambda s: calls.append(s.time), seconds=0.3)
    assert 10 <= len(calls) <= 18 and len(set(calls)) == len(calls)  # one call per 20 ms tick
    ctl = tmp_path / "ctl.py"
    ctl.write_text("def control(s):\n    return {'joints': {'right_shoulder_pitch_joint': {'q': -0.4}}}\n")
    th = threading.Thread(target=r.loop, args=(str(ctl),), kwargs={"seconds": 1.2}, daemon=True)
    th.start()
    time.sleep(0.6)
    assert r.state.joint_positions[22] < -0.2
    ctl.write_text("def control(s):\n    return {'joints': {'right_shoulder_pitch_joint': {'q': 0.2}}}\n")
    th.join(timeout=5)
    assert r.state.joint_positions[22] > 0.0  # the edited controller took over without stopping the loop
