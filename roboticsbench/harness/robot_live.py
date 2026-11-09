"""The live robot: ``from robot import robot`` - connected at first use, its senses streaming in every tick.

Copied into the agent's workspace as ``robot.py`` (run scripts with ``./python``, which has numpy and pillow)::

    from robot import robot
    s = robot.state                         # latest sensors, refreshed every 20 ms in the background
    s.joint_positions, s.imu_pelvis.quaternion, s.foot_contact_forces, ...   # numpy arrays (keys: ROBOT.md)
    robot.send({"q": q, "kp": kp, "kd": kd})  # motor command: applied at the next tick, persists until changed
    robot.wait(5)                           # block for 5 ticks (100 ms) and return the newest state
    robot.loop(control)                     # call control(state) -> command (or None) every tick
    robot.loop("controller.py")             # ... with control() from that file, re-loaded whenever it changes
    v = robot.see(depth=True)               # a camera image now: v.image, v.depth (metres), v.calib, v.path
    robot.done()                            # declare the task complete

It talks to the episode's referee over HTTP (URL in ``.robot_api`` next to this file); ``robot_client.py`` is the
plain request/response client. Standard library plus numpy and pillow when they are installed.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

try:
    import numpy as np
except ImportError:  # pragma: no cover - plain python3: vectors stay lists
    np = None

HERE = Path(__file__).resolve().parent


class RobotError(RuntimeError):
    pass


class State(dict):
    """One sensor reading: the keys of the sensor table in ROBOT.md, also as attributes (``s.joint_positions``);
    number lists are numpy arrays."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None


def _wrap(o: Any) -> Any:
    if isinstance(o, dict):
        return State({k: _wrap(v) for k, v in o.items()})
    if isinstance(o, list) and np is not None and o and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                                            for x in o):
        return np.asarray(o, dtype=float)
    return o


class View:
    """One camera image: ``image`` (H x W x 3 uint8), ``depth`` (H x W metres, with depth=True), ``calib`` (pose and
    intrinsics, see ROBOT.md), ``path`` (the saved JPEG), ``t`` (simulated time of the picture)."""

    def __init__(self, camera: str, path: str, image: Any, calib: dict, depth: Any, t: float):
        self.camera, self.path, self.image, self.calib, self.depth, self.t = camera, path, image, calib, depth, t

    def __repr__(self) -> str:
        shape = getattr(self.image, "shape", None)
        return f"View(camera={self.camera!r}, t={self.t}, path={self.path!r}, image={shape}, depth={self.depth is not None})"


def _api_url() -> str:
    url = os.environ.get("ROBOT_API_URL", "").strip()
    if not url and (HERE / ".robot_api").exists():
        url = (HERE / ".robot_api").read_text().strip()
    if not url:
        raise RobotError("robot API is not configured (missing .robot_api)")
    return url.rstrip("/")


class Robot:
    """The robot of this episode. Real time: a background thread keeps ``state`` current from the referee's
    per-tick stream. Turn-based episodes have no stream: ``wait(n)`` then advances the simulation by n ticks."""

    def __init__(self, url: str | None = None, timeout: float = 30.0, stream: bool = True):
        self.url = (url or _api_url()).rstrip("/")
        self.timeout = timeout
        info = self._request("GET", "/info")
        self.realtime = bool(info.get("realtime"))
        self.spec: dict = info.get("motor_spec") or {}
        self.cameras: list[str] = list(info.get("cameras") or [])
        self.joints: list[str] = [j["name"] for j in self.spec.get("joints", [])]
        self.joint_index: dict[str, int] = {n: i for i, n in enumerate(self.joints)}
        self.hand_joints: list[str] = [j["name"] for j in self.spec.get("hand_joints", [])]
        self.tick_s = float(self.spec.get("tick_s", 0.02))
        self.views_dir = HERE / "views"
        self._cond = threading.Condition()
        self._msg: dict = {}
        self._state: State = State()
        self._tick = -1
        self._errors: dict[str, int] = {}
        if stream and self.realtime:
            threading.Thread(target=self._stream, name="robot-senses", daemon=True).start()
            with self._cond:
                self._cond.wait_for(lambda: self._tick >= 0, timeout=10.0)
        if self._tick < 0:
            self._update(self._request("GET", "/state"))

    # ------------------------------------------------------------------ senses
    @property
    def state(self) -> State:
        """The latest sensor reading (refreshed every tick in real-time episodes)."""
        return self._state

    @property
    def t(self) -> float:
        """Simulated time of the latest reading (s)."""
        return float(self._state.get("time", 0.0))

    @property
    def time_left(self) -> float:
        return float(self._msg.get("time_left", 0.0))

    @property
    def status(self) -> str:
        """Episode status: running, success, failure, timeout or incomplete."""
        return str(self._msg.get("episode_status", "running"))

    @property
    def running(self) -> bool:
        return self.status == "running"

    @property
    def task(self) -> dict:
        """The task's public status (see GOAL.md)."""
        return self._msg.get("task") or {}

    def _update(self, msg: dict) -> None:
        sensors = msg.get("sensors")
        if not isinstance(sensors, dict):
            return
        with self._cond:
            tick = int(sensors.get("tick", self._tick))
            if tick < self._tick:  # an older reply arriving after a newer stream line
                return
            self._msg, self._state, self._tick = msg, _wrap(sensors), tick
            self._cond.notify_all()

    def _stream(self) -> None:
        failures = 0
        while failures < 20:
            try:
                with urllib.request.urlopen(self.url + "/stream", timeout=10.0) as r:
                    for line in r:
                        failures = 0
                        self._update(json.loads(line))
                        if self._msg.get("episode_status", "running") != "running":
                            return
            except (OSError, ValueError):
                failures += 1
                time.sleep(0.25)
        with self._cond:
            self._cond.notify_all()

    def wait(self, ticks: int = 1, timeout: float = 10.0) -> State:
        """Wait ``ticks`` control ticks (20 ms each) and return the newest state. In turn-based episodes this is what
        advances the simulation."""
        if not self.realtime:
            self._update(self._post_step(None, int(ticks)))
            return self._state
        target = self._tick + max(1, int(ticks))
        with self._cond:
            fresh = self._cond.wait_for(lambda: self._tick >= target or not self.running, timeout=timeout)
        if not fresh:  # the stream is gone: ask directly
            self._update(self._request("GET", "/state"))
        return self._state

    # ------------------------------------------------------------------ motors
    def send(self, command: dict) -> dict:
        """Apply a motor command (format: ROBOT.md). It takes effect at the next tick and persists until changed.
        Raises RobotError if the robot rejects it (then nothing was applied)."""
        rep = self._post_step(command, 0)
        if not rep.get("ok", False):
            raise RobotError(rep.get("message", "command rejected"))
        return rep

    def loop(self, controller: Callable[[State], dict | None] | str | os.PathLike, seconds: float | None = None,
             until: Callable[[State], bool] | None = None) -> None:
        """Run a controller every tick: ``controller(state)`` returns a motor command, or None to keep the last one.
        ``controller`` may be the path of a .py file that defines ``control(state)``: the file is re-loaded whenever
        it changes, so you can edit the behaviour while the loop keeps the robot under control (``robot`` is
        available inside it). Stops after ``seconds``, when ``until(state)`` is true or when the episode ends.
        An exception in the controller is printed and that tick sends nothing."""
        get = self._controller(controller)
        t_end = None if seconds is None else self.t + float(seconds)
        while self.running:
            s = self._state
            fn = get()
            cmd = None
            if fn is not None:
                try:
                    cmd = fn(s)
                except Exception:  # noqa: BLE001 - keep the loop alive
                    self._report(traceback.format_exc())
            if cmd:
                try:
                    self.send(cmd)
                except (RobotError, OSError) as e:
                    self._report(f"command not applied: {e}")
            if (until is not None and until(s)) or (t_end is not None and self.t >= t_end):
                return
            self.wait(1)

    def _controller(self, controller) -> Callable[[], Callable | None]:
        if callable(controller):
            return lambda: controller
        path = Path(controller)
        cache: dict[str, Any] = {"mtime": None, "fn": None}

        def get():
            try:
                mtime = path.stat().st_mtime_ns
            except OSError:
                return cache["fn"]
            if mtime != cache["mtime"]:
                cache["mtime"] = mtime
                try:
                    ns: dict[str, Any] = {"__name__": "controller", "__file__": str(path), "robot": self}
                    exec(compile(path.read_text(), str(path), "exec"), ns)
                    if not callable(ns.get("control")):
                        raise RobotError(f"{path} defines no control(state) function")
                    cache["fn"] = ns["control"]
                    print(f"[robot.loop] t={self.t:.2f}s loaded control() from {path}", file=sys.stderr, flush=True)
                except Exception:  # noqa: BLE001 - keep the previous controller
                    self._report(traceback.format_exc())
            return cache["fn"]

        return get

    def _report(self, msg: str) -> None:
        n = self._errors.get(msg, 0)
        self._errors[msg] = n + 1
        if n < 3:
            print(f"[robot.loop] t={self.t:.2f}s {msg.rstrip()}", file=sys.stderr, flush=True)

    # ------------------------------------------------------------------ cameras
    def see(self, camera: str | None = None, depth: bool = False, width: int | None = None,
            height: int | None = None) -> View:
        """A picture from ``camera`` (default: the first one in ``robot.cameras``) taken now, saved under views/
        with its calibration (.json) and, with ``depth=True``, a depth map (_depth.png, mm)."""
        cam = camera or self.cameras[0]
        rep = self._request("POST", "/look", {"cameras": [cam], "width": width, "height": height,
                                              "depth": bool(depth)})
        if not rep.get("ok"):
            raise RobotError(rep.get("message", "look failed"))
        self.views_dir.mkdir(parents=True, exist_ok=True)
        t = float(rep.get("t_sim", 0.0))
        stem = f"{len(list(self.views_dir.glob('*.jpg'))):04d}_t{t:07.2f}_{cam}"
        jpg = base64.b64decode(rep["images"][cam])
        path = self.views_dir / f"{stem}.jpg"
        path.write_bytes(jpg)
        calib = (rep.get("calibration") or {}).get(cam) or {}
        (self.views_dir / f"{stem}.json").write_text(json.dumps(calib, indent=1))
        image = dep = None
        try:
            from PIL import Image

            image = np.asarray(Image.open(io.BytesIO(jpg)).convert("RGB")) if np is not None else None
            if depth and (rep.get("depth_png_mm") or {}).get(cam):
                png = base64.b64decode(rep["depth_png_mm"][cam])
                (self.views_dir / f"{stem}_depth.png").write_bytes(png)
                if np is not None:
                    dep = np.asarray(Image.open(io.BytesIO(png)), dtype=np.float32) / 1000.0
        except ImportError:  # pragma: no cover - no pillow: files only
            pass
        try:
            shown = str(path.relative_to(Path.cwd()))
        except ValueError:
            shown = str(path)
        return View(cam, shown, image, calib, dep, t)

    # ------------------------------------------------------------------ episode
    def done(self) -> dict:
        """Declare the task complete: the episode is scored after a short settle."""
        return self._request("POST", "/done", {})

    def __repr__(self) -> str:
        return (f"<robot {self.spec.get('robot', '?')} t={self.t:.2f}s status={self.status} "
                f"time_left={self.time_left:.0f}s cameras={self.cameras}>")

    # ------------------------------------------------------------------ transport
    def _post_step(self, command: dict | None, ticks: int) -> dict:
        rep = self._request("POST", "/step", {"command": command, "ticks": int(ticks)})
        self._update(rep)
        return rep

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            try:
                return json.loads(e.read())
            except Exception:  # noqa: BLE001
                raise RobotError(f"HTTP {e.code} from the robot") from e
        except urllib.error.URLError as e:
            raise RobotError(f"cannot reach the robot at {self.url}: {e.reason} (the episode may have ended)") from e


class _Robot:
    """``robot``: the episode's Robot, connected on first use."""

    _robot: Robot | None = None
    _lock = threading.Lock()

    def _get(self) -> Robot:
        if _Robot._robot is None:
            with _Robot._lock:
                if _Robot._robot is None:
                    _Robot._robot = Robot()
        return _Robot._robot

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get(), name)

    def __repr__(self) -> str:
        return repr(self._get())


robot = _Robot()
