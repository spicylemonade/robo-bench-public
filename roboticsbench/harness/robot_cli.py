#!/usr/bin/env python3
"""RoboticsBench robot CLI and Python client (standard library only).

This file is copied into the agent's workspace twice: as the executable ``./robot`` and as the importable
``robot_client.py``. It talks to the episode's referee over HTTP (URL in ``.robot_api`` next to this file, or
``$ROBOT_API_URL``); the referee owns the simulator.

    ./robot info | state | skills | objects | done
    ./robot look [camera ...] [--width W --height H] [--depth]
    ./robot <skill> key=value ... [--look]

    from robot_client import Robot
    robot = Robot()
    robot.call("walk_to", x=1.0, y=0.5)      # or robot.walk_to(x=1.0, y=0.5)
    robot.look(["head"])                       # -> ["views/0003_head.jpg"]
"""

from __future__ import annotations

import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _api_url() -> str:
    url = os.environ.get("ROBOT_API_URL", "").strip()
    if not url:
        f = HERE / ".robot_api"
        if f.exists():
            url = f.read_text().strip()
    if not url:
        raise SystemExit("robot API is not configured (missing .robot_api)")
    return url.rstrip("/")


class RobotError(RuntimeError):
    pass


class Robot:
    """Client for the episode referee. Every method returns the referee's JSON reply as a dict."""

    def __init__(self, url: str | None = None, views_dir: str | os.PathLike | None = None, timeout: float = 900.0):
        self.url = (url or _api_url()).rstrip("/")
        self.views = Path(views_dir) if views_dir else HERE / "views"
        self.timeout = timeout

    # ------------------------------------------------------------------ transport
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
                raise RobotError(f"HTTP {e.code} from the referee") from e
        except urllib.error.URLError as e:
            raise RobotError(f"cannot reach the robot referee at {self.url}: {e.reason} "
                             "(the episode may have ended)") from e

    # ------------------------------------------------------------------ API
    def info(self) -> dict:
        return self._request("GET", "/info")

    def state(self) -> dict:
        return self._request("GET", "/state")

    def skills(self) -> list[dict]:
        return self.info().get("tools", [])

    def call(self, skill: str, **args) -> dict:
        return self._request("POST", "/act", {"skill": skill, "args": args})

    def objects(self) -> dict:
        """Task objects with exact positions (when the task allows it): {"ok", "objects": [...]}."""
        return self._request("POST", "/objects", {})

    # ---- motor control (ROBOT.md says which control mode this episode uses)
    def sensors(self) -> dict:
        """Current sensor readings (motor control modes)."""
        return self.state().get("sensors", {})

    def step(self, ticks: int = 1, command: dict | None = None) -> dict:
        """Apply ``command`` (persistent until changed; None = keep the last one) and advance ``ticks`` control
        ticks. Returns {"ok", "sensors", "events", "episode_status", "time_left", "task", ...}."""
        return self._request("POST", "/step", {"command": command, "ticks": int(ticks)})

    def spec(self) -> dict:
        """Joint table, limits and gains (motor control modes)."""
        return self.info().get("motor_spec", {})

    def done(self) -> dict:
        return self._request("POST", "/done", {})

    def look(self, cameras: list[str] | None = None, width: int | None = None, height: int | None = None,
             depth: bool = False) -> list[str]:
        """Save the current camera images under views/ and return their paths (relative to the workspace).
        Next to every image ``<name>.jpg`` a ``<name>.json`` holds that camera's calibration at this instant
        (world pose + pinhole intrinsics); with ``depth=True`` a ``<name>_depth.png`` (16-bit, millimetres along
        the optical axis) is saved too."""
        rep = self._request("POST", "/look", {"cameras": cameras, "width": width, "height": height,
                                              "depth": bool(depth)})
        if not rep.get("ok"):
            raise RobotError(rep.get("message", "look failed"))
        self.views.mkdir(parents=True, exist_ok=True)
        n = len(list(self.views.glob("*.jpg")))
        paths = []
        for cam, b64 in rep["images"].items():
            stem = f"{n:04d}_t{rep.get('t_sim', 0):07.2f}_{cam}"
            n += 1
            p = self.views / f"{stem}.jpg"
            p.write_bytes(base64.b64decode(b64))
            cal = (rep.get("calibration") or {}).get(cam)
            if cal is not None:
                (self.views / f"{stem}.json").write_text(json.dumps(cal, indent=1))
            dep = (rep.get("depth_png_mm") or {}).get(cam)
            if dep is not None:
                (self.views / f"{stem}_depth.png").write_bytes(base64.b64decode(dep))
            try:
                paths.append(str(p.relative_to(Path.cwd())))
            except ValueError:
                paths.append(str(p))
        return paths

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda **kw: self.call(name, **kw)


# ---------------------------------------------------------------------- CLI
def _value(s: str):
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s


_NUM_LIST = None


def _print(obj) -> None:
    """Pretty JSON with short lists of numbers kept on one line."""
    global _NUM_LIST
    import re

    if _NUM_LIST is None:
        _NUM_LIST = re.compile(r"\[\s*((?:-?[\d.eE+-]+|true|false|null)(?:,\s*(?:-?[\d.eE+-]+|true|false|null))*)\s*\]")
    text = json.dumps(obj, indent=1, ensure_ascii=False)
    print(_NUM_LIST.sub(lambda m: "[" + ", ".join(x.strip() for x in m.group(1).split(",")) + "]", text))


USAGE = """usage:
  ./robot info                          task, control mode, budgets, cameras
  ./robot state                         simulated time, robot state / sensors, task status
  ./robot objects                       task objects with positions (when the task allows it)
  ./robot look [camera ...] [--width W] [--height H] [--depth]   save camera images (+ calibration .json,
                                                       + 16-bit depth .png in mm with --depth) to views/
  ./robot done                          declare the task complete
 skills mode:
  ./robot skills                        every skill with its parameters
  ./robot <skill> key=value ... [--look]               run a skill (e.g. ./robot wait seconds=1)
 motor / velocity modes:
  ./robot sensors                       sensor readings
  ./robot spec                          joint table, limits, gains (JSON)
  ./robot step [ticks=N] [command='{...json...}'] [--look]   apply a motor command and advance N x 20 ms"""


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0
    robot = Robot()
    cmd, rest = argv[0], argv[1:]
    try:
        if cmd == "info":
            info = robot.info()
            _print({k: v for k, v in info.items() if k not in ("tools", "manual_md", "goal_md")})
        elif cmd == "state":
            _print(robot.state())
        elif cmd == "skills":
            for t in robot.skills():
                props = t["parameters"].get("properties", {})
                req = set(t["parameters"].get("required", []))
                args = []
                for k, spec in props.items():
                    typ = "|".join(map(str, spec["enum"])) if "enum" in spec else spec.get("type", "")
                    args.append(f"{k}:{typ}" + ("" if k in req else f"={spec.get('default', '?')}"))
                print(f"{t['name']}({', '.join(args)})\n    {t['description']}")
        elif cmd == "objects":
            rep = robot.objects()
            _print(rep)
        elif cmd == "sensors":
            _print(robot.sensors())
        elif cmd == "spec":
            _print(robot.spec())
        elif cmd == "step":
            kw = {"ticks": 1, "command": None}
            for tok in rest:
                if tok == "--look":
                    continue
                if "=" not in tok:
                    print(f"bad argument {tok!r}: use ticks=N command='{{...}}'", file=sys.stderr)
                    return 2
                k, v = tok.split("=", 1)
                if k == "ticks":
                    kw["ticks"] = int(v)
                elif k == "command":
                    try:
                        kw["command"] = json.loads(v)
                    except json.JSONDecodeError as e:
                        print(f"command is not valid JSON: {e}", file=sys.stderr)
                        return 2
                else:
                    print(f"unknown argument {k!r} (use ticks=N command='{{...}}')", file=sys.stderr)
                    return 2
            rep = robot.step(**kw)
            if "--look" in rest and rep.get("episode_status") == "running":
                rep["views"] = robot.look()
            _print(rep)
            if not rep.get("ok", False):
                return 1
        elif cmd == "look":
            cams, kw, i = [], {}, 0
            while i < len(rest):
                if rest[i] in ("--width", "--height") and i + 1 < len(rest):
                    kw[rest[i][2:]] = int(rest[i + 1])
                    i += 2
                    continue
                if rest[i] == "--depth":
                    kw["depth"] = True
                    i += 1
                    continue
                cams.append(rest[i])
                i += 1
            paths = robot.look(cams or None, **kw)
            _print({"saved": paths})
        elif cmd == "done":
            _print(robot.done())
        else:
            look_after = "--look" in rest
            args = {}
            for tok in rest:
                if tok == "--look":
                    continue
                if "=" not in tok:
                    print(f"bad argument {tok!r}: use key=value (see ./robot skills)", file=sys.stderr)
                    return 2
                k, v = tok.split("=", 1)
                args[k.strip()] = _value(v.strip())
            rep = robot.call(cmd, **args)
            if look_after and rep.get("state", {}).get("episode_status") == "running":
                rep["views"] = robot.look()
            _print(rep)
            if not rep.get("result", {}).get("ok", False):
                return 1
    except RobotError as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
