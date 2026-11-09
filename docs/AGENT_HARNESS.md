# Coding-agent harness

How RoboticsBench runs agents such as Claude Code or Codex, and the contract any other harness can build on.

```
  referee (root, owns MuJoCo)  <--HTTP 127.0.0.1-->  ./robot CLI / robot_client.py  <--  coding agent (unprivileged)
        |                                                    workspace: GOAL.md ROBOT.md robot robot_client.py views/
        +-- referee_result.json (signed) + referee_trace.json + video.mp4   (ledger, root-only)
```

## Episode life cycle

1. `python -m roboticsbench.harness.referee --task T --seed S --port P --ledger DIR [--vision] [--video]
   [--control motor] [--realtime] [--time-limit SECONDS]` starts one episode. The task and seed are fixed; the agent cannot reset or choose seeds. Operator endpoints
   require `X-Operator-Token` = `$RBENCH_OPERATOR_TOKEN` (read at start-up and removed from the environment).
2. `roboticsbench.harness.workspace.make_workspace(dir, GET /info, url, python)` writes the agent's working
   directory (motor modes: plus `robot_model.xml`, the datasheet `robot_spec.json`, the live robot module `robot.py`
   - `from robot import robot`: senses streamed every tick, `send`, `wait`, `loop` with hot-reloaded controller
   files, `see` - and a `./python` launcher for an interpreter with numpy, scipy, mujoco and pillow that puts the
   workspace on the module path).
3. The agent runs with the workspace as cwd. Motor modes: the prompt is `/info`'s `brief_md` - the task, the live
   interface, the command format, the joint table with sign conventions, the start state and the body's key numbers,
   so the agent can act without looking anything up. Skills mode: `workspace.AGENT_PROMPT` (read GOAL.md/ROBOT.md,
   use `./robot`, finish with `./robot done`). Real-time runs: the orchestrator calls `POST /operator/start` as it
   launches the agent, so the clock - and the time budget, by default 3 minutes - starts the moment the agent gets
   in; from then on the world runs at 1x in a background thread whether or not the agent sends commands.
4. The episode ends on success, failure, the simulated time limit, the action budget, `./robot done`, or when
   the operator calls `POST /operator/finalize` (the agent exited or hit the wall-clock limit): the referee
   settles for the task's hold time and closes the episode as `incomplete` if it has not ended.
5. The ledger is written once, when the episode ends: `referee_trace.json` (instruction, every skill call with
   arguments and simulated times, events) and `referee_result.json` (status, score, milestones, budgets used,
   `trace_sha256`, and an HMAC `sig` when `RBENCH_LEDGER_SECRET` is set).

`roboticsbench.harness.run.run_agent_episode` does all of this for one episode (`rbench agent`). Official runs do
the same in a container where the agent runs as an unprivileged user: the simulator code, the referee and the
ledger are closed to it, and only its provider API key is passed.

## Control modes

`--control motor` (the default for coding agents): joint-level motor commands only (`roboticsbench/core/motor.py`).
Panda: `{"joint_positions": [7], "gripper": {"width", "force"}}`; G1: `{"q"|"dq"|"kp"|"kd"|"tau": [29],
"joints": {name: {...}}, "hand_q": [14]}`. `--control velocity` (G1): `{"base_velocity": [vx, vy, wz],
"pelvis_height", "arm_q": [14], "hand_q": [14]}` with the legs run by the GR00T walking policy. `--control skills`:
the high-level skill API. The workspace additionally gets `robot_model.xml` (the robot's MuJoCo model without
meshes) in the motor modes; in official runs the pretrained policy files are unreadable for the agent.

## Agent-facing API (referee)

| endpoint | purpose |
|---|---|
| `GET /info` | task, robot, control mode, cameras, budgets, skill schemas or `motor_spec` + `robot_model_xml`, `goal_md`, `manual_md`, `variant` |
| `GET /state` | simulated time / time left, robot state (skills) or `sensors` (motor modes), public task status, episode status |
| `GET /stream` | motor modes: the `/state` body once per control tick, one JSON line each, until the episode ends (not counted as observations) |
| `POST /act {skill, args}` | skills mode: run one skill; returns `{result: {ok, message, data?}, state}` |
| `POST /step {command?, ticks}` | motor modes: apply a (persistent) motor command, advance `ticks` x 20 ms (0-250); returns `{ok, events, sensors, ...}` |
| `POST /objects` | task objects with positions (when the task and variant allow it) |
| `POST /look {cameras?, width?, height?, depth?}` | current camera images (base64 JPEG) with each camera's calibration, optional 16-bit depth PNGs (mm); rendered off the physics thread |
| `POST /done` | declare completion |

Budgets: skills mode - robot actions (every skill except `look` / `list_objects`; invalid calls are free);
motor modes - up to 200k `/step` calls of at most 250 ticks. Always: observation requests, the time limit (the
task's own, or `--time-limit`: 180 s by default for real-time agent runs) and whatever wall-clock / spend limits
the launcher applies. Turn-based runs: simulated time only advances inside skills / steps. Real-time runs: it
advances at 1x from the clock start; `/step` waits for its ticks on that clock and requests from several agent
processes are served concurrently.

## Scoring and integrity

Score = 1.0 on success, else the latched milestone credit (< 1). Results carry `infra_ok`; infrastructure
failures (agent could not start, API outage before any robot call) are excluded from scores.

`rbench verify <episode_dir>` re-simulates the recorded skill calls or motor commands from the seed and checks that status, success,
progress, milestones and simulated time are reproduced exactly (MuJoCo, the IK and the single-threaded ONNX walking
policy are deterministic), so a result cannot be edited without detection.

## Variants

`--vision` removes `list_objects()` (and says so in GOAL.md / the system prompt): the agent has to localise
everything from camera images. Results are reported under `task@vision`.
