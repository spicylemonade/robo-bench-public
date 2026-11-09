"""Robot/task briefing text shared by every agent interface.

* :func:`system_prompt` - system prompt for the built-in tool-calling LLM agent (direct model APIs).
* :func:`manual_markdown` / :func:`goal_markdown` - ``ROBOT.md`` / ``GOAL.md`` for coding agents (Claude Code,
  Codex, ...) that drive the robot through the ``./robot`` CLI of the agent harness.

Both are rendered from the same sections so every harness gets the same facts.
"""

from __future__ import annotations

import json
import math
from typing import Any

CAMERA_NOTES = {
    "head": "head - the robot's own head camera, looking forward and slightly down (what the robot sees).",
    "chase": "chase - a third-person camera that follows the robot from behind.",
    "overview": "overview - a fixed camera showing the whole task area.",
    "front": ("front - from the far side of the table looking back at the robot: +x points toward the camera "
              "and +y is on the image's RIGHT."),
    "top": "top - straight down on the table: image up = +x (away from the robot), image left = +y.",
    "side": "side - from the robot's right-hand side (-y) looking across the table: image right = +x.",
    "wrist": "wrist - mounted on the gripper, looking along the fingers.",
}

HUMANOID_INTRO = """You are controlling a Unitree G1 humanoid robot (1.3 m tall, 35 kg, two 7-DoF arms with 3-finger \
hands) in a physics simulation (MuJoCo). The physics is realistic: the robot balances and walks with a learned \
whole-body controller, objects have mass and friction, and the robot can trip or fall over if it walks into walls \
or obstacles. A fall does not end the episode: the robot can get back up and carry on."""

HUMANOID_INTRO_MOTOR = """You are controlling a Unitree G1 humanoid robot (1.3 m tall, 35 kg, 29 body motors, two 3-finger Dex3 hands) in a physics simulation (MuJoCo). The physics is realistic: the robot has mass and inertia, its motors have torque limits, and it falls over when it loses balance - nothing balances it for you. Objects have mass and friction. A fall does not end the episode: getting back up is up to you."""

ARM_INTRO = """You are controlling a Franka Emika Panda 7-DoF robot arm with a parallel-jaw gripper, mounted on a \
table, in a physics simulation (MuJoCo). The physics is realistic: objects have mass and friction, can slip out of \
the gripper, topple, roll, or fall off the table. The arm stops itself (protective stop) if it collides hard."""

HUMANOID_COORDS = """- World frame in metres, floor at z = 0. Headings in degrees: 0 = +x, 90 = +y (counter-clockwise \
is positive).
- The robot starts at ({x0:.2f}, {y0:.2f}) facing {h0:.0f} deg.
- Robot frame (walk(), reach(frame='robot')): x forward, y to the robot's left, z = height above the floor.
- Body facts: pelvis height 0.74 m when standing (top of the head ~1.28 m, shoulders ~1.05 m); the body is ~0.36 m \
wide with the arms in the 'zero'/'tucked'/'push' postures ('ready' ~0.54 m). Crouched walking (set_height 0.60-0.66) \
lowers the head top to ~1.15 m but only works going forward at >= 0.5 m/s. Comfortable walking speed 0.6 m/s; \
walk_to/walk stop within ~5 cm of the target. Palms reach ~0.2-0.5 m in front of the pelvis at heights ~0.7-1.2 m."""

HUMANOID_TIPS = """- walk_to() walks in a straight line and does not avoid obstacles: plan waypoints around them.
- Contact with obstacles is physical: brushing a wall can push or topple the robot."""

ARM_COORDS = """- World frame in metres. The robot base is at the origin, the table surface is z = 0, +x points away \
from the robot, +y is to the robot's left. The table has a 10 cm grid; a red mark (+x) and a green mark (+y) sit \
near the base.
- Reachable workspace: roughly x 0.25-0.80, y -0.45..0.45, z 0-0.6.
- Gripper positions refer to the tool centre point between the fingertips; the fingertips extend about 1 cm below \
it. To grasp an object, open the gripper, move the TCP above it, descend to about the object's centre height, close.
- The gripper points straight down by default. Gripper yaw 0 deg = the fingers close along the world y axis; rotate \
the yaw to match the object's orientation. Maximum finger opening is 8 cm. The finger pads cover ~1.7 cm of height \
around the TCP; each finger is ~2.1 cm wide and ~2.4 cm thick along the closing direction, so it needs that much free \
space beside an object. The hand housing above the fingers is ~21 cm long along the closing direction and ~6 cm wide, \
its underside ~3.7 cm above the TCP. Large wrist rotations in one motion can be refused as out of reach - turn in \
steps of <= ~45-90 deg; a gripped object can slip or twist a little in the pads during fast moves."""

TOOLCALL_CONTROL = """- You act by calling tools. Each tool runs a closed-loop controller for some simulated time and \
returns a text result followed by the latest camera images.
- The simulation is paused while you think; only your actions advance simulated time.
- Several tool calls in one response are executed in order; if one fails, the remaining calls of that response are \
skipped.
- Budget: {max_actions} tool calls and {time_limit:.0f} s of simulated time."""

SYSTEM_TEMPLATE = """{intro}

## How control works
{control}

## Coordinates
{coords}
{tips}
## Cameras
{cameras}
{perception}
## Your task
{instruction}

When you believe the task is complete, call done(). If you stop calling tools, the episode ends and is scored as it is."""


VISION_NOTE = ("EVALUATION MODE - CAMERA-ONLY PERCEPTION: like a real robot, you get no ground-truth object state. "
               "The object list (list_objects() / robot.objects() / ./robot objects) is not available in this run, "
               "even where the task text mentions it or describes what it would return, and the task status does not "
               "report where objects are. Perceive the scene from the calibrated cameras instead: every saved image comes "
               "with its camera's pose and intrinsics (see Camera calibration in ROBOT.md) and depth maps are available "
               "on request. Aids in the scene: the table has a 10 cm grid, venue floors a 1 m grid with lines at whole "
               "metres.")

CAMERA_CALIB = """### Camera calibration
The cameras are calibrated. Every image saved by `./robot look` / `robot.look()` has a `.json` file with the same name
next to it: the camera's world `position`, its `rotation` (3x3, columns = the camera's x / y / z axes in world
coordinates) and the pinhole intrinsics `fx`, `fy`, `cx`, `cy` for that image size (`projection: orthographic`
cameras give `metres_per_pixel` instead). Camera frame: x = image right, y = image up, the camera looks along -z. A
world point p appears at c = R^T (p - position), pixel u = cx + fx * c_x / (-c_z), v = cy - fy * c_y / (-c_z)
(u to the right, v down, (0, 0) = top-left corner of the image). Cameras on the robot (head, wrist) report their pose
at the moment of the image. `./robot look --depth` (or `robot.look(depth=True)`) also saves `<name>_depth.png`:
16-bit depth in millimetres along the optical axis (the -z distance), like an RGB-D camera. Larger images:
`--width W --height H` (up to 1280 x 960)."""


def _instruction(env, tool_names: set[str]) -> str:
    """Task instruction, plus the vision-only note when the run withholds an object list the task normally has."""
    if env.task.allow_object_list and "list_objects" not in tool_names:
        return env.task.instruction_for(True) + "\n\n" + VISION_NOTE
    return env.task.instruction


def _is_humanoid(env) -> bool:
    return env.robot.name == "g1"


def failure_rule(env) -> str:
    """How a failure ends the episode. Humanoid tasks run on after a fall or a broken rule (2026-10-09)."""
    if env.task.ends_on_failure:
        return "A failure (fall, broken rule) ends the episode immediately."
    return ("A fall does not end the episode: get back up and carry on. A broken task rule rules out success, but the "
            "episode still runs to the time limit and partial credit counts.")


def episode_ends(env, time_runs_out: str, done: str) -> str:
    if env.task.ends_on_failure:
        return f"The episode ends on success, failure (e.g. a fall or a broken rule), {time_runs_out}, or {done}."
    return (f"The episode ends on success, {time_runs_out}, or {done} - never early on a fall (get back up and carry "
            "on) or a broken task rule (that rules out success; partial credit still counts).")


def _sections(env, cameras: tuple[str, ...], allow_objects: bool) -> dict[str, str]:
    humanoid = _is_humanoid(env)
    if humanoid:
        b = env.robot.base_pos
        coords = HUMANOID_COORDS.format(x0=b[0] + 0.0, y0=b[1] + 0.0,
                                        h0=round(math.degrees(env.robot.yaw)) + 0.0)
        tips = "\n## Tips\n" + HUMANOID_TIPS + "\n"
    else:
        coords, tips = ARM_COORDS, ""
    if allow_objects:
        perception = ("\n## Perception\nlist_objects() returns the task-relevant objects with exact positions "
                      "(simulated perfect perception). Hidden properties such as mass are not included.\n")
    else:
        perception = "\n## Perception\nThere is no object list in this task: rely on the camera images.\n"
    return {"intro": HUMANOID_INTRO if humanoid else ARM_INTRO, "coords": coords, "tips": tips,
            "cameras": "\n".join(f"- {CAMERA_NOTES.get(c, c)}" for c in cameras), "perception": perception}


def system_prompt(env, sk, max_actions: int, cameras: tuple[str, ...]) -> str:
    """System prompt for the built-in tool-calling agent."""
    names = {t["name"] for t in sk.tools()}
    s = _sections(env, cameras, "list_objects" in names)
    control = TOOLCALL_CONTROL.format(max_actions=max_actions, time_limit=env.task.time_limit)
    return SYSTEM_TEMPLATE.format(control=control, instruction=_instruction(env, names), **s)


def observation_text(env, actions_used: int, max_actions: int, result: dict[str, Any] | None = None) -> str:
    lines = []
    if result is not None:
        lines.append(f"[result] {'ok' if result['ok'] else 'FAILED'}: {result['message']}")
    lines.append(f"[time] {env.t:.1f} s elapsed, {env.time_left:.1f} s left; tool calls used {actions_used}/{max_actions}")
    from roboticsbench.core.jsonutil import jsonable

    lines.append("[robot] " + json.dumps(jsonable(env.robot.proprio()), separators=(",", ":")))
    st = jsonable(env.task.status(env))
    if st:
        lines.append("[task] " + json.dumps(st, separators=(",", ":")))
    if env.status != "running":
        lines.append(f"[episode] ENDED: {env.status}" + (f" ({env.fail_reason})" if env.fail_reason else ""))
    return "\n".join(lines)


# --------------------------------------------------------------------------- coding-agent harness documents
CLI_CONTROL = """- You control the robot ONLY through the `./robot` command in this directory (or the `robot_client.py` \
Python module next to it, which exposes the same calls). There is no other access to the simulator.
- Every robot skill runs a closed-loop controller for some simulated time and prints the result plus the new robot \
state. The simulation is paused between commands: take as long as you need to think or write code - only robot \
skills advance simulated time.
- Camera images are saved as JPEG files under `views/`; open them with your image-viewing tool to see the scene.
- Budget: {max_actions} robot actions (every skill except `look`/`objects`/`state`) and {time_limit:.0f} s of \
simulated time. {ends}"""

CLI_REFERENCE = """```
./robot info                    # task, budgets and cameras (same as GOAL.md)
./robot state                   # simulated time, robot state, task status, budget use
./robot look [camera ...]       # save camera images to views/ and print their paths
{objects_line}./robot skills                  # list every skill with its parameters
./robot <skill> key=value ...   # run a skill, e.g.  {example}
./robot done                    # declare the task complete (ends the episode; it is then scored)
```
Arguments are `key=value` pairs (numbers, strings, true/false). Add `--look` to any skill to also save the camera \
views afterwards. Output is JSON on stdout.

From Python (e.g. for scripted, closed-loop behaviour):
```python
from robot_client import Robot
robot = Robot()                  # connects to the same episode
res = robot.call({example_py})
print(res["result"]["ok"], res["result"]["message"], res["state"]["time"])
paths = robot.look()             # list of saved image paths
```"""


def _skill_lines(tools: list[dict[str, Any]]) -> str:
    out = []
    for t in sorted(tools, key=lambda t: t["name"]):
        props = t["parameters"].get("properties", {})
        req = set(t["parameters"].get("required", []))
        args = []
        for k, spec in props.items():
            typ = spec.get("type", "")
            if "enum" in spec:
                typ = "|".join(str(e) for e in spec["enum"])
            a = f"{k}:{typ}"
            if k not in req:
                a += f"={spec['default']}" if "default" in spec else "?"
            args.append(a)
        out.append(f"- `{t['name']}({', '.join(args)})` - {t['description']}")
    return "\n".join(out)


def manual_markdown(env, tools: list[dict[str, Any]], cameras: tuple[str, ...], max_actions: int) -> str:
    """ROBOT.md - the contract for coding agents."""
    names = {t["name"] for t in tools}
    s = _sections(env, cameras, "list_objects" in names)
    humanoid = _is_humanoid(env)
    example = "walk_to x=2.0 y=-0.5 heading_deg=90" if humanoid else "move_to x=0.5 y=0.1 z=0.15 yaw_deg=30"
    example_py = ('"walk_to", x=2.0, y=-0.5, heading_deg=90' if humanoid else '"move_to", x=0.5, y=0.1, z=0.15')
    objects_line = "./robot objects                 # list_objects(): task objects with exact positions\n" \
        if "list_objects" in names else ""
    body = [
        "# Robot manual", "", s["intro"], "",
        "## How control works",
        CLI_CONTROL.format(max_actions=max_actions, time_limit=env.task.time_limit,
                          ends=episode_ends(env, "when the simulated time runs out", "when you run `./robot done`")), "",
        "## The ./robot command", CLI_REFERENCE.format(objects_line=objects_line, example=example,
                                                       example_py=example_py), "",
        "## Skills", _skill_lines([t for t in tools if t["name"] not in ("look",)]), "",
        "## Coordinates", s["coords"], s["tips"],
        "## Cameras", s["cameras"], "", CAMERA_CALIB, s["perception"],
        "## Rules",
        "- Only interact with the robot through `./robot` / `robot_client.py`. Do not try to access, inspect or modify "
        "the simulator, the referee process, or anything outside this directory; doing so disqualifies the run.",
        "- Scoring: 100% for success; otherwise partial credit for the milestones reached (shown in the task status "
        "where available). " + failure_rule(env),
        "",
    ]
    return "\n".join(body)


def goal_markdown(env, max_actions: int, tools: list[dict[str, Any]] | None = None) -> str:
    """GOAL.md - the problem statement for coding agents."""
    names = {t["name"] for t in tools} if tools is not None else ({"list_objects"} if env.task.allow_object_list
                                                                  else set())
    return "\n".join([
        f"# Task: {env.task.name}", "",
        _instruction(env, names), "",
        "## How to work",
        "- Read `ROBOT.md` first: it explains the robot, the coordinate frames, the cameras and the `./robot` command.",
        "- Look before you act (`./robot look`, `./robot state`" + (", `./robot objects`" if "list_objects" in names
                                                                    else "") + ") and check the result of every "
        "action.",
        f"- Budget: {max_actions} robot actions and {env.task.time_limit:.0f} s of simulated time.",
        "- When the task is complete, run `./robot done`.",
        "",
    ])


# --------------------------------------------------------------------------- motor-control documents
MOTOR_CONTROL = """- You drive the robot's motors directly. There are NO high-level skills: no walk-to, no inverse \
kinematics, no grasp routine{no_balance}. Everything the robot does, you command joint by joint.
- Use `robot_client.py` from Python for anything closed-loop (recommended) or the `./robot` CLI for single \
commands. A motor command persists until you change it. {clock}
- Python: run your scripts with `./python` - Python 3.12 with numpy, scipy, mujoco and pillow (mujoco can load \
`robot_model.xml`). The system `python3` may lack these packages; do not spend your time installing them.
- Everything about the robot - interface, sensors, joint axes and sign conventions, limits, gains, dimensions, \
masses, cameras - is in this manual and, machine-readable, in `robot_spec.json` (`robot_model.xml` is its full \
MuJoCo model). All values are exact: the sensors have no noise, bias or delay and the model matches the robot.
- Camera images: `robot.look()` saves JPEGs under `views/`; open them with your image-viewing tool.
- {attempt}
- Budget: {budget}. {ends}"""

CLOCK_TURN = (
    "Simulated time only advances when you step: `robot.step(ticks=n, command=...)` applies the command (if any) and "
    "advances n ticks of {tick_ms:.0f} ms ({rate} Hz control rate, 1 <= n <= 250), then returns the sensor readings. "
    "The simulation is paused between calls, so take as long as you need to think or compute - a script can run a "
    "{rate} Hz control loop in simulated time.")

CLOCK_REAL = (
    "REAL TIME, ALREADY RUNNING: the clock started the moment you were launched, and the simulation runs "
    "continuously at real-time speed (1 simulated second per second) whether or not you send commands. Your whole "
    "budget ({budget}) covers reading these files, understanding the robot, writing code and doing the task. While "
    "you think or write code the robot keeps executing the last command (joint targets and gains persist; in velocity "
    "mode the last body velocity persists - command a stop before you pause). `robot.step(ticks=n, command=...)` "
    "applies the command (if any) at once and returns after n ticks of {tick_ms:.0f} ms ({rate} Hz control rate, "
    "0 <= n <= 250) with the sensor readings of that moment; `ticks=0` returns at once. A Python loop around "
    "`robot.step(1, ...)` is a {rate} Hz controller as long as your code keeps up. Run whole behaviours as scripts "
    "that close the loop themselves - a model that pauses to think mid-task loses that time.")

MOTOR_CLIENT = """```python
from robot_client import Robot              # run with ./python
robot = Robot()
s = robot.sensors()                          # current sensor readings (dict, keys in "Sensors" below)
for _ in range(250):                         # 5 s at {rate} Hz
    cmd = my_controller(s)                   # your code: a dict in the command format below (or None)
    reply = robot.step(ticks=1, command=cmd)
    s = reply["sensors"]
    if reply["episode_status"] != "running":
        break
robot.look()                                 # camera images -> views/*.jpg (+ calibration .json)
{objects_py}robot.done()                                  # declare the task complete
```"""

PANDA_COMMANDS = """## Commands
- `{{"joint_positions": [q1, ..., q7]}}` - radians, targets for the 7 joint servos (order and limits below). The \
servos track them with gravity compensation. Target changes are rate-limited to each joint's maximum speed \
(joints 1-4: 2.175 rad/s, 5-7: 2.61 rad/s), so a distant target is approached at full speed - send intermediate \
targets for slow, straight or careful motions. Targets are clipped to the joint limits (minus 0.005 rad).
- `{{"gripper": {{"width": w, "force": f}}}}` - finger opening target w in metres (0 - 0.08) with the gripper \
motor limited to f newtons (5 - 70, default 40). To grasp, command a width below the object's width (e.g. 0) and \
the fingers squeeze it with f; to release, command a width larger than the object.
- Both fields may be combined in one command.
- Collision reflex: if an arm link (not the fingers) hits something with more than 45 N, or the external force on \
the hand exceeds 80 N (e.g. pushing a held part hard into something), the robot stops the arm (its targets are \
reset to where it is). This is reported in the step reply's `events` and in `sensors.protective_stop` until you \
send new joint targets.
- The gripper pose is NOT reported: compute it from the joint angles with the kinematics below."""

PANDA_KINEMATICS = """Forward kinematics, standard Franka Emika Panda modified Denavit-Hartenberg parameters (Craig \
convention: T_i = RotX(alpha_(i-1)) TransX(a_(i-1)) RotZ(q_i) TransZ(d_i)), metres / radians, from the base frame:

| joint | a_(i-1) | d_i | alpha_(i-1) |
|---|---|---|---|
| 1 | 0 | 0.333 | 0 |
| 2 | 0 | 0 | -pi/2 |
| 3 | 0 | 0.316 | pi/2 |
| 4 | 0.0825 | 0 | pi/2 |
| 5 | -0.0825 | 0.384 | -pi/2 |
| 6 | 0 | 0 | pi/2 |
| 7 | 0.088 | 0 | pi/2 |

Then the flange: TransZ(0.107); the hand is mounted on the flange rotated by -pi/4 about z; the tool centre \
point (TCP, between the fingertips) is TransZ(0.1034) from the hand. In the TCP frame z points out along the \
fingers and the fingers open/close along its y axis. The fingertips reach ~0.009 m beyond the TCP along z; each \
finger is ~2.1 cm wide (along TCP x) and ~2.6 cm thick beyond its pad (along TCP y); the pads cover ~1.7 cm around \
the TCP along z. The hand housing spans +-10.4 cm along TCP y and +-3.2 cm along TCP x, its underside 3.7 cm above \
the TCP (towards the flange). At the start posture (q = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]) the TCP is at \
(0.307, 0, 0.487) pointing straight down with the fingers closing along the world y axis."""

G1_HANDS = ("With the arms in the zero posture the forearms point straight forward (palm points ~0.3 m ahead of the "
            "pelvis, 0.15 m to either side and ~0.1 m above it, palms facing inward); with open hands the thumbs "
            "point inward to ~3.4 cm from the robot's centre line. The fingers are weak and only hold small objects "
            "reliably in a supporting grip - lift larger loads on the forearms or palms.")

G1_COMMANDS = """## Commands (low-level motor commands, like Unitree's SDK)
Every body motor applies `tau = kp * (q - q_measured) + kd * (dq - dq_measured) + tau_ff`, clipped to its torque \
limit; the motor drivers evaluate it at {physics_hz} Hz with your latest values. Like real motors they have a \
torque-speed curve: full torque up to 70% of the joint's no-load speed (`no_load_speed` in robot_spec.json), falling \
to zero at it while driving the joint faster (braking is not limited). Command any subset of fields; unspecified \
values keep their last setting:
- `{{"q": [29], "dq": [29], "kp": [29], "kd": [29], "tau": [29]}}` - arrays in the joint order below (rad, \
rad/s, N m/rad, N m s/rad, N m). Gains are clipped to [0, kp_max] / [0, kd_max], targets to the joint limits, \
tau to the torque limits.
- `{{"joints": {{"left_knee_joint": {{"q": 0.8, "kp": 300}}, ...}}}}` - the same per joint, by name.
- `{{"hand_q": [14]}}` - position targets for the Dex3 hand joints (left 7 then right 7, order below). The hand \
motors are weak (~1.4 N m, thumb rotation 2.45 N m). {hands}
- The start gains (`kp_start` / `kd_start`) hold the robot still like a statue. The robot's standard gains \
(`kp_default` / `kd_default`, the real G1's) are too soft to hold a static stance on their own. Moving the arms, \
shifting weight, stepping or walking all need active balance control from you: there is no balance or walking \
controller. The robot counts as fallen if its pelvis drops below 0.35 m or tilts more than 55 deg; a fall does \
not end the episode - you can get back up."""

G1_VELOCITY_COMMANDS = """## Commands (legs: walking policy; arms and hands: joint targets)
The legs are driven by the robot's built-in walking/balance policy (a learned whole-body controller), which \
follows a commanded body velocity - like driving the robot with a joystick. It keeps the robot balanced, but it \
does not know where anything is: you close the loop yourself (there is no walk-to).
- `{{"base_velocity": [vx, vy, wz]}}` - forward (-0.6 .. 1.0 m/s), left (-0.5 .. 0.5 m/s) and yaw rate \
(-1 .. 1 rad/s, + = counter-clockwise) of the pelvis. The achieved speed is ~90% of the command and ramps up over \
~0.5 s; [0, 0, 0] stands still.
- `{{"pelvis_height": h}}` - 0.55 - 0.78 m (0.74 = normal standing, head top ~1.28 m). Crouched walking works at \
0.60 - 0.66 only going forward at >= 0.5 m/s (head top ~1.15 m); the robot cannot turn in place below 0.68.
- `{{"arm_q": [14]}}` - arm joint targets (left 7 then right 7, the arm rows of the joint tables below), tracked \
by PD with gravity compensation. The targets move towards a new command at most `arm_speed` rad/s per joint.
- `{{"arm_speed": s}}` - that slew limit, 0.2 - 10 rad/s (default 2.5). Fast swings (e.g. throwing: ~5 m/s at the \
hand with 6+ rad/s) are possible; the 25 N m arm motors and the balance of the robot are the real limits.
- `{{"hand_q": [14]}}` - Dex3 hand joint targets (order below) (weak: ~1.4 N m, thumb rotation 2.45 N m). {hands}"""

API_REFERENCE = """Live module - `from robot import robot` (run with `./python`): `robot.state` (latest sensors, \
refreshed every tick by a background stream), `robot.t`, `robot.time_left`, `robot.status`, `robot.task`, \
`robot.send(command)`, `robot.wait(ticks=1)`, `robot.loop(control_or_file, seconds=None, until=None)`, \
`robot.see(camera=None, depth=False, width=None, height=None)` -> View (`image`, `depth`, `calib`, `path`, `t`), \
`robot.done()`; `robot.joints` / `robot.joint_index` (names <-> indices), `robot.cameras`, `robot.spec` (the \
datasheet). `Robot(url)` makes another instance.

Request/response client - `from robot_client import Robot`; every method returns the referee's JSON reply as a \
dict:
- `robot = Robot()` connects to this episode (base URL in `.robot_api`); `Robot(timeout=s)` sets the HTTP timeout \
(default 900 s).
- `robot.step(ticks=1, command=None)` applies `command` (unless None; it persists) and returns after `ticks` \
control ticks (0 - 250; 0 = at once): `{{"ok", "sensors", "events", "episode_status", "time_left", "task"}}`. \
`ok` is false with a `message` for a rejected command (then nothing was applied).
- `robot.sensors()` - the current sensor readings; `robot.state()` - `{{"sensors", "time_left", "task", \
"episode_status"}}`.
- `robot.look(cameras=None, width=None, height=None, depth=False)` saves \
`views/<n>_t<sim time>_<camera>.jpg` for the listed cameras (default: all), each with a `.json` calibration file \
(and `_depth.png` with depth=True), and returns the image paths. Default 640 x 480, up to 1280 x 960.
{objects_api}- `robot.spec()` - the robot datasheet (the same as `robot_spec.json`).
- `robot.done()` - declare the task complete; the episode is then scored after a short settle.

CLI - `./robot sensors | state | spec | done`, `./robot step ticks=N command='{{"kp": [...]}}'`, \
`./robot look [camera ...] [--depth] [--width W --height H]`{objects_cli}.

HTTP (any language) - base URL in `.robot_api`, JSON in and out: `GET /state`, `GET /stream` (the `/state` \
body once per tick, one JSON line each), `POST /step {{"command": {{...}} or \
null, "ticks": n}}`, `POST /look {{"cameras": [...], "width": w, "height": h, "depth": true}}` (base64 JPEGs in \
`images`, calibrations in `calibration`, base64 16-bit PNGs in `depth_png_mm`), `POST /done`, `GET /info`.

Timing and concurrency - a step round trip adds ~1-2 ms to the wait for its ticks; a command takes effect at the \
next tick boundary. Several processes or threads may use the robot at once (e.g. a {rate} Hz control loop in one \
and camera captures in another); camera requests do not pause the world or the control loop. Capturing all \
cameras at 640 x 480 takes ~0.1 s (~0.2 s with depth).

Limits - 0 - 250 ticks per step call, 200,000 step calls and 50,000 state / look requests per episode."""


def _table(rows: list[dict[str, Any]], cols: list[tuple[str, str]]) -> str:
    head = "| " + " | ".join(label for _, label in cols) + " |"
    sep = "|" + "---|" * len(cols)
    body = []
    for r in rows:
        cells = []
        for key, _ in cols:
            v = r.get(key, "")
            cells.append(f"{v:.3g}" if isinstance(v, float) else str(v))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *body])


def time_budget_text(env, realtime: bool) -> str:
    limit = float(env.task.time_limit)
    if realtime:
        return f"{limit / 60:g} minutes ({limit:.0f} s) of real time in total, counted from the moment you were launched"
    return f"{limit:.0f} s of simulated time"


def motor_manual(env, control: str, spec: dict[str, Any], cameras: tuple[str, ...], allow_objects: bool,
                 realtime: bool = False) -> str:
    """ROBOT.md for the motor-control modes: how control works, then the robot's datasheet (see
    :mod:`roboticsbench.core.robot_spec`) and the API reference."""
    from roboticsbench.core.robot_spec import spec_markdown

    s = _sections(env, cameras, allow_objects)
    sheet = spec_markdown(spec)
    humanoid = _is_humanoid(env)
    rate = spec["control_rate_hz"]
    tick_ms = 1000 * spec["tick_s"]
    budget = time_budget_text(env, realtime)
    no_balance = ", no balance or walking controller" if humanoid and control == "motor" else ""
    if not humanoid:
        commands, kinematics = PANDA_COMMANDS.format(), PANDA_KINEMATICS + "\n\n"
    else:
        tmpl = G1_COMMANDS if control == "motor" else G1_VELOCITY_COMMANDS
        commands, kinematics = tmpl.format(physics_hz=spec["physics_rate_hz"], hands=G1_HANDS), ""
    if humanoid:
        b = env.robot.base_pos
        coords = ("- World frame in metres, floor at z = 0. Headings in degrees: 0 = +x, 90 = +y (counter-clockwise "
                  "is positive).\n"
                  f"- The robot starts with its pelvis at ({b[0] + 0.0:.2f}, {b[1] + 0.0:.2f}) facing "
                  f"{round(math.degrees(env.robot.yaw)) + 0.0:.0f} deg in the world frame.")
    else:
        coords = ("- World frame in metres. The robot base is at the origin, the table surface is z = 0, +x points "
                  "away from the robot, +y is to the robot's left. The table has a 10 cm grid; a red mark (+x) and a "
                  "green mark (+y) sit near the base.\n"
                  "- Comfortable workspace for the gripper: roughly x 0.25-0.80, y -0.45..0.45, z 0-0.6.")
    objects_py = "objs = robot.objects()                       # task objects with exact positions\n" \
        if allow_objects else ""
    clock = (CLOCK_REAL if realtime else CLOCK_TURN).format(tick_ms=tick_ms, rate=rate, budget=budget)
    intro = HUMANOID_INTRO_MOTOR if humanoid and control == "motor" else s["intro"]
    parts = [
        f"# Robot manual - {'motor' if control == 'motor' else 'velocity + joint'} control", "", intro, "",
        "## How control works",
        MOTOR_CONTROL.format(
            no_balance=no_balance, clock=clock, budget=budget,
            attempt=("The episode is a single attempt: a fall or a broken rule ends it for good - there is no reset."
                     if env.task.ends_on_failure else
                     "The episode is a single attempt with no reset, but a fall does not end it: get back up."),
            ends=episode_ends(env, "when the time runs out", "when you call `robot.done()` / `./robot done`")), "",
        "## Start state", sheet["start"], "",
        "## Live interface (robot.py)",
        LIVE_PY.format(tick_ms=tick_ms, command_example=("{\"q\": q, \"kp\": kp, \"kd\": kd, \"tau\": tau}"
                                                         if humanoid and control == "motor" else
                                                         "{\"base_velocity\": [0.3, 0, 0], \"arm_q\": arm_q}"
                                                         if humanoid else
                                                         "{\"joint_positions\": q, \"gripper\": {\"width\": 0.0}}"
                                                         ).ljust(28),
                       see_args="depth=True".ljust(10),
                       see_doc="a camera image now (robot.cameras): v.image, v.depth (m), v.calib, v.path"),
        "Lower level: `robot_client.py` (request/response, standard library only) - its loop looks like this:",
        MOTOR_CLIENT.format(rate=rate, objects_py=objects_py), "",
        commands, "",
        "## Sensors", sheet["sensors"], "",
        "## Joint axes and sign conventions", sheet["axes"], "",
        "## Motors: limits and gains", sheet["motors"], "",
        "## Body: kinematics, dimensions and masses", kinematics + sheet["body"], "",
        "## Coordinates", coords, "",
        "## Cameras", s["cameras"], "", "Mounting and intrinsics (default image size):", sheet["cameras"], "",
        CAMERA_CALIB,
        s["perception"].replace("list_objects()", "robot.objects() / ./robot objects"),
        "## API reference",
        API_REFERENCE.format(rate=rate,
                             objects_api="- `robot.objects()` - task objects with exact positions.\n"
                             if allow_objects else "",
                             objects_cli=", `./robot objects`" if allow_objects else ""), "",
        "## Rules",
        "- Only interact with the robot through `robot_client.py` / `./robot` (or its HTTP API). Do not try to "
        "access, inspect or modify the simulator, the referee process, or anything outside this directory, and do "
        "not download controllers or policies: write the control yourself. Violations disqualify the run.",
        "- Scoring: 100% for success; otherwise partial credit for the milestones reached (see the task status in "
        "each reply). " + failure_rule(env),
        "",
    ]
    return "\n".join(parts)


SKILL_TEXT_NOTE = ("NOTE - MOTOR CONTROL: the task text below may mention high-level skills of the robot's standard "
                   "software (walk_to, walk, face, set_height, arm_pose, reach, grasp, release, move_to, "
                   "close_gripper, open_gripper, list_objects, ...). They are NOT available in this run - they "
                   "describe what the stock robot can be made to do. You command the motors yourself (see "
                   "ROBOT.md); the physical facts in the text (sizes, heights, forces, thresholds) still apply.")


def motor_goal_markdown(env, control: str, allow_objects: bool, realtime: bool = False) -> str:
    """GOAL.md for the motor-control modes."""
    instr = env.task.instruction_for(not allow_objects)
    if env.task.allow_object_list and not allow_objects:
        instr += "\n\n" + VISION_NOTE
    limit = float(env.task.time_limit)
    if realtime:
        time_line = (f"- TIME: {limit / 60:g} minutes ({limit:.0f} s) of REAL time in total, and the clock is already "
                     "running - it started the moment you were launched. Reading, understanding the robot, writing "
                     "code and doing the task all count; the robot never waits for you. Be quick.")
    else:
        time_line = (f"- Budget: {limit:.0f} s of simulated time (time only passes when you step). The episode is a "
                     + ("single attempt: a fall or broken rule ends it - test risky controllers in your own simulation "
                        "first." if env.task.ends_on_failure else
                        "single attempt with no reset - test risky controllers in your own simulation first."))
    return "\n".join([
        f"# Task: {env.task.name}", "",
        SKILL_TEXT_NOTE, "",
        instr, "",
        "## How to work",
        time_line,
        "- The robot is live: `from robot import robot` (run with `./python`) gives its senses (`robot.state`, "
        "refreshed every tick) and motors (`robot.send`, `robot.loop`) at once. `ROBOT.md` documents everything "
        "about the robot: the motor commands, the sensors, joint axes and sign conventions, limits, dimensions, "
        "cameras and the API; `robot_spec.json` has the same data as JSON.",
        "- You have no motion primitives: write your own controllers (Python scripts using `robot_client.py`) that "
        "run a whole behaviour in a closed loop, and check the sensors and camera images.",
        ("- The episode is a single attempt: a fall or a broken rule ends it for good." if env.task.ends_on_failure
         else "- The episode is a single attempt, but nothing ends it early except success: after a fall, get back "
              "up and carry on; a broken task rule rules out success, but partial credit still counts."),
        "- When the task is complete, call `robot.done()` (or `./robot done`).",
        "",
    ])


# --------------------------------------------------------------------------- launch brief (motor control)
LIVE_PY = """```python
from robot import robot                    # connected: its senses refresh every {tick_ms:.0f} ms in the background
s = robot.state                            # latest sensors (keys below; vectors are numpy arrays)
robot.send({command_example})   # motor command: applied at the next tick, persists until changed
robot.wait(n)                              # wait n ticks of {tick_ms:.0f} ms, returns the newest state
robot.loop(control)                        # control(state) -> command or None, called every tick
robot.loop("controller.py")                # same, with control() re-loaded from that file whenever you edit it
v = robot.see({see_args})                # {see_doc}
robot.done()                               # declare the task complete
```"""


def motor_brief(env, control: str, spec: dict[str, Any], cameras: tuple[str, ...], allow_objects: bool,
                realtime: bool = False) -> str:
    """The launch prompt for motor-control coding agents: everything needed to act at once - the task, the live
    robot interface, the command format, the joints with their sign conventions, the start state, the body's key
    numbers and the cameras - so nothing has to be looked up first (ROBOT.md / robot_spec.json have the rest)."""
    humanoid = _is_humanoid(env)
    tick_ms, rate = 1000 * spec["tick_s"], spec["control_rate_hz"]
    limit = float(env.task.time_limit)
    if humanoid:
        who = "a Unitree G1 humanoid robot (1.32 m, 34.4 kg, 29 body motors, two 3-finger Dex3 hands)"
        how = ("You command every motor yourself: there is no balance, walking or grasping software."
               if control == "motor" else "Its legs follow body-velocity commands; the arms and hands are yours.")
    else:
        who = "a Franka Emika Panda 7-joint arm with a parallel gripper, mounted on a table"
        how = ("You command the joint positions and the gripper yourself: there is no inverse kinematics, motion "
               "planning or grasping software.")
    clock = (f"REAL TIME: the clock started the moment you were launched - you have {limit / 60:g} minutes "
             f"({limit:.0f} s) in total, and the robot never waits for you, so act quickly."
             if realtime else f"Time only passes when you step the simulation; budget {limit:.0f} s of simulated time.")
    instr = env.task.instruction_for(not allow_objects)
    if env.task.allow_object_list and not allow_objects:
        instr += "\n\n" + VISION_NOTE
    sensors = "; ".join(f"`{s['key']}`" + (f"[{s['shape'][0]}]" if s["shape"] else "") +
                        (f" {s['units']}" if s["units"] not in ("-", "") else "") +
                        (f" ({s['frame']})" if s["frame"] and s["frame"] not in ("joint",) else "")
                        for s in spec["sensors"] if s["key"] not in ("tick",))
    if humanoid and control == "motor":
        example = '{"q": q, "kp": kp, "kd": kd, "tau": tau}'
    elif humanoid:
        example = '{"base_velocity": [0.3, 0, 0], "arm_q": arm_q}'
    else:
        example = '{"joint_positions": q, "gripper": {"width": 0.0, "force": 40}}'
    cam0 = cameras[0] if cameras else "head"
    see_doc = (f"{cam0} camera now: v.image, v.depth (m), v.calib, saved as v.path" if len(cameras) == 1 else
               f"a camera image now (default '{cam0}'; see(camera='...')): v.image, v.depth (m), v.calib, v.path")
    live = LIVE_PY.format(tick_ms=tick_ms, command_example=example.ljust(28), see_args="depth=True".ljust(10),
                          see_doc=see_doc)
    if humanoid:
        commands = (G1_COMMANDS if control == "motor" else G1_VELOCITY_COMMANDS).format(
            physics_hz=spec["physics_rate_hz"], hands=G1_HANDS)
        cols = [("index", "#"), ("name", "joint"), ("plus", "+0.1 rad moves"), ("range", "range (rad)"),
                ("torque_limit", "torque (N m)")]
        if control == "motor":
            cols += [("q_start", "q_start"), ("start_gains", "kp/kd start"), ("max_gains", "kp/kd max")]
    else:
        commands = PANDA_COMMANDS
        cols = [("index", "#"), ("name", "joint"), ("plus", "+0.1 rad moves the TCP"), ("range", "range (rad)"),
                ("max_speed", "max speed (rad/s)"), ("start", "start")]
    rows = []
    for j in spec["joints"]:
        r = dict(j)
        r["plus"] = (j["plus_0p1rad_text"] if j["plus_0p1rad_moves"] == "TCP" else
                     f"{j['plus_0p1rad_moves']} {j['plus_0p1rad_text']}")
        r["range"] = f"{j['min']:.2f} .. {j['max']:.2f}"
        if humanoid:
            r["start_gains"] = f"{j.get('kp_start', 0):g} / {j.get('kd_start', 0):g}"
            r["max_gains"] = f"{j.get('kp_max', 0):g} / {j.get('kd_max', 0):g}"
        rows.append(r)
    dims = spec["dimensions"]
    if humanoid:
        imu = dims["imu"]["imu_pelvis"]
        body = "\n".join([
            f"- Robot frame below: origin on the floor under the pelvis at the start, x forward, y left, z up. "
            f"Pelvis origin at the start {dims['pelvis_origin_at_start_m']}; whole-body centre of mass "
            f"{dims['com_at_start_m']}.",
            f"- Legs: hip pitch joints at y = +-{dims['hip_spacing_m'] / 2:.4f}; thigh (hip pitch -> knee) "
            f"{dims['thigh_length_m (hip pitch -> knee)']} m, shank (knee -> ankle pitch) "
            f"{dims['shank_length_m (knee -> ankle pitch)']} m, ankle pitch axis "
            f"{dims['ankle_pitch_axis_height_above_sole_m']} m above the sole.",
            f"- Feet: each sole touches the ground through 4 small spheres (radius "
            f"{dims['sole_contact_sphere_radius_m']} m) at {dims['sole_contact_points_in_ankle_roll_link_m']} in "
            f"the ankle roll link frame: a {dims['sole_contact_patch_m (length x width)'][0]} x "
            f"{dims['sole_contact_patch_m (length x width)'][1]} m patch; sole centres at the start "
            f"{dims['sole_centre_left_right_at_start_m']} (stance width {dims['stance_width_at_start_m (sole centres)']} m).",
            f"- Arms: shoulder pitch joints at {dims['shoulder_pitch_joint_left_right_m']}; upper arm "
            f"{dims['upper_arm_length_m (shoulder pitch -> elbow)']} m, forearm "
            f"{dims['forearm_length_m (elbow -> wrist roll)']} m, wrist roll -> palm centre "
            f"{dims['wrist_roll_to_palm_centre_m']} m.",
            f"- IMUs: pelvis IMU at {imu['offset_in_body']} in the pelvis frame; torso IMU at "
            f"{dims['imu']['imu_torso']['offset_in_body']} in the torso frame; both aligned with their link.",
            f"- {dims['fall_rule'][0].upper()}{dims['fall_rule'][1:]}.",
        ])
    else:
        body = (PANDA_KINEMATICS + "\n- Joint anchors at the start: " +
                "; ".join(f"{k} {v}" for k, v in dims["joint_anchors_at_start_m"].items()) + ".")
    cams = []
    for c in spec["cameras"]:
        line = f"- `{c['name']}`: {c['kind']}"
        if c.get("kind") == "onboard":
            line += (f" on {c['mount_body']}, at {c['position_at_start_m']} at the start, looking along "
                     f"{c['looks_along_at_start']} ({c['pitch_down_deg_at_start']} deg down)")
        if c.get("fovy_deg"):
            line += f", vertical field of view {c['fovy_deg']:g} deg"
        intr = next((v for k, v in c.items() if k.startswith("intrinsics_")), None)
        if intr:
            line += f", {env.width} x {env.height} default (fx = fy = {intr['fx']:g}, cx = {intr['cx']:g}, cy = {intr['cy']:g})"
        cams.append(line + ".")
    parts = [
        f"You are the controller of {who} in a live physics simulation (MuJoCo). {how} {clock} Everything you need to "
        "start is below; this directory also holds GOAL.md (the task), ROBOT.md (the full datasheet: every sensor "
        "frame, joint axis, link mass and the API), robot_spec.json (the same as JSON) and robot_model.xml (your "
        "body's MuJoCo model).",
        "",
        "## Task",
        SKILL_TEXT_NOTE,
        "",
        instr,
        "",
        "## Your senses and motors are live",
        "Run Python with `./python` (numpy, scipy, mujoco and pillow installed) from any directory:",
        live,
        f"Sensors (`robot.state`, exact, every {tick_ms:.0f} ms): {sensors}.",
        ("While you think, the robot keeps executing the last command: run behaviours as `robot.loop(...)` scripts "
         "in the background (e.g. `./python walk.py > walk.log 2>&1 &`) and steer them by editing their "
         "controller file. Shell: `./robot sensors`, `./robot step ticks=N command='{...}'`, `./robot look --depth`."
         if realtime else
         "Shell: `./robot sensors`, `./robot step ticks=N command='{...}'`, `./robot look --depth`."),
        "",
        "## Start state",
        spec["start"],
        "",
        commands,
        "",
        "## Joints (+ rotation = counter-clockwise about the joint axis; motion of the named point at the start "
        "posture, robot frame: x forward, y left, z up)",
        _table(rows, cols),
        "",
        "## Body",
        body,
        "",
        "## Cameras",
        *cams,
        "Every image comes with its camera's pose and pinhole intrinsics (u = cx + fx * c_x / (-c_z), "
        "v = cy - fy * c_y / (-c_z) in the camera frame: x right, y up, looking along -z); depth on request.",
        "",
        "## Rules",
        "Use only robot.py / robot_client.py / ./robot (the robot's interface). Do not read or modify anything "
        "outside this directory and write the control yourself. " +
        ("A broken task rule ends the episode at once. " if env.task.ends_on_failure else failure_rule(env) + " ") +
        "Call `robot.done()` when the task is complete.",
    ]
    return "\n".join(parts)
