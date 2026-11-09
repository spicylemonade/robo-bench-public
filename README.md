# Robo-bench: public test set

The three public tasks of [Robo-bench](https://www.vals.ai/benchmarks/robo_bench), Vals AI's benchmark for controlling
a robot arm and a humanoid in a real-time physics simulation. The simulator, referee, robot interface and agent
launchers are the ones the full benchmark runs on; the other 21 tasks are held out. An integration that works here
runs unchanged on the held-out set, which Vals runs with it.

Shared for integration work: please don't redistribute this repository.

| task | robot | description |
|---|---|---|
| `arm/balance_scale` | Franka Panda arm | A beam balance carries a sealed box of unknown mass on its left pan; find its mass (wrist force sensor or trial and error) and put the right labelled calibrated weights (10-200 g) on the right pan so that the beam settles level within 2 deg for 2 s - the tolerance is finer than the smallest weight. |
| `arm/lego_manual` | Franka Panda arm | Build a 5-6 brick model on a studded baseplate following an illustrated step-by-step manual (PNG pages); bricks click together through real stud geometry and must be pressed in place. |
| `humanoid/ninja_course` | Unitree G1 humanoid | Ninja-warrior course: climb 5 stairs, time a step onto a motor-driven swinging platform, ride it over a water pit (leaning with the swing) and step off at the far deck, walk down a ramp, crawl on hands and knees through a 0.8 m high crawl space, get up and reach the finish. |

## Install

Python 3.11 or newer.

```bash
uv venv && uv pip install -e ".[llm,dev]"
rbench fetch-assets      # robot models, pinned: Unitree G1 + GR00T policies (NVlabs), Panda (MuJoCo Menagerie)
rbench list
rbench validate public --seeds 0-4     # each task's reference solution, and a do-nothing baseline
pytest -q
```

Dependencies are pinned to the official runs' simulator stack (MuJoCo 3.14.0; official runs use Python 3.12 on Linux
with OSMesa rendering). The reference solutions prove each task can be solved through the agent's own interface; the
ninja course's skips the stairs, the swing and the crawl space (it reports them as `reference_skipped`).

Watch a task live with its reference solution: `mjpython -m roboticsbench.cli view arm/lego_manual --oracle` (on
macOS the viewer needs `mjpython`, which ships with the `mujoco` package).

## How an episode runs

```
  referee (owns MuJoCo)  <--HTTP 127.0.0.1-->  ./robot CLI / robot.py  <--  agent (your CLI, in the workspace)
        +-- referee_result.json + referee_trace.json + video.mp4   (the ledger, closed to the agent)
```

* **Real time.** The clock starts when the agent is launched and the world runs at 1x whether or not the agent sends
  commands. The agent has 3 minutes in total (`--time-limit`, default 180 s) to read the documentation, write its
  controllers and do the task.
* **Sensors and cameras only.** The arm sees through a fixed `top` camera and its `wrist` camera, the humanoid only
  through its `head` camera, plus each robot's own sensors: joint positions, velocities and torques, the arm's gripper
  and wrist force/torque sensor, the humanoid's two IMUs, foot contact forces and base position and velocity. In the
  leaderboard setting (`--vision`) the referee never reveals where the objects are.
* **Motor commands only.** Panda: 7 joint position targets and the gripper (width, force limit). G1: per-motor
  `q, dq, kp, kd, tau` for its 29 joints and 14 hand joints. There is no inverse kinematics, grasping, walking or
  balance controller to call: the agent writes its own.
* **The workspace** holds `GOAL.md`, `ROBOT.md` (a datasheet generated from the simulation model: sensor table, joint
  axes and sign conventions, limits, gains, masses, camera mounts and intrinsics, API reference), `robot_spec.json`,
  `robot_model.xml`, the `./robot` CLI, `robot.py` (`from robot import robot`: `robot.state`, `robot.send(command)`,
  `robot.loop(control)` with hot-reloaded controller files, `robot.see(depth=True)`) and `./python` (numpy, scipy,
  mujoco, pillow). Task documents come along too, such as the manual pages of `arm/lego_manual`.
* **The launch prompt** carries the task, the live interface, the command format, the joint table and the body's key
  numbers, so the agent can act without looking anything up.
* **Scoring.** 1.0 on success, otherwise the milestone credit reached (below 1). An arm episode ends at once on a
  broken rule; a humanoid episode runs to the time limit (the robot may get back up after a fall) and a broken rule
  only rules out success.

The full contract (episode life cycle, referee endpoints, budgets, integrity checks) is in
[docs/AGENT_HARNESS.md](docs/AGENT_HARNESS.md).

## Running an agent

```bash
rbench agent arm/balance_scale --harness claude-code --model claude-opus-5-5 --effort max --vision --video
rbench agent humanoid/ninja_course --harness codex --model gpt-6.1-sol --effort max --vision
rbench agents --harness claude-code --model claude-opus-5-5 --effort max --vision --tasks public --seeds 0-2 --workers 3
rbench verify runs/agent-*/*/      # re-simulate episodes from their recorded motor commands
```

The leaderboard setting is motor control (the default), real time (the default for motor control), 180 s and
`--vision`; each model runs in its vendor's own CLI where one exists, at its highest reasoning setting. Each episode
directory holds `result.json` (score, status, milestones, usage), the referee's ledger (`referee_result.json`,
`referee_trace.json` with every motor command), the agent's `agent_stdout.jsonl` and `agent_stderr.log`, its
workspace and, with `--video`, `video.mp4`. Claude Code runs stop themselves at $25 per episode (`--budget-usd`).

Local runs are for development: the agent runs as your user and could read files outside its workspace. Official runs
put it in a container as an unprivileged user that cannot read the simulator, the referee or the ledger, and pass it
only its provider API key.

## Integrating your model

**Your agent CLI** (the leaderboard setting). Add a `Harness` subclass to
[roboticsbench/harness/agents.py](roboticsbench/harness/agents.py) and register it in `HARNESSES`:

* `launch(prompt, workspace, *, model, effort, budget_usd, isolated)` returns a `Launch`: the non-interactive command,
  extra environment variables, `stdin` and `files` to write into the workspace first. The command runs with the
  workspace as its working directory. Pass the prompt on stdin or in a workspace file, not on the command line (an
  agent's own `pkill -f` can match it). `isolated=True` means the command already runs inside the official container
  as an unprivileged user, so it must not start a sandbox of its own.
* `parse(stdout_path)` reads the CLI's event stream (stderr is next to it in `agent_stderr.log`) into `turns`,
  `usage` (`in_tokens`, `out_tokens`, `cache_read_tokens`, `cache_write_tokens`, optionally `reasoning_tokens` and
  `cost_usd`), `served_models`, `final_message`, and `is_error` / `error` when every model call failed.
* Match the other launchers: no web search or fetch tools, no telemetry or auto-updates, and nothing from the host
  machine's user configuration (settings, rules, memory, plugins, MCP servers).

The launchers for Claude Code, Codex, Gemini CLI, Antigravity, Grok Build, Muse Code, Kimi Code and OpenCode are
worked examples, tested in `tests/test_harness_parsers.py` and `tests/test_harness_isolation.py`. (Antigravity, and
Muse Code in isolated runs, go through metering wrappers of the official runner that are not part of this repository.)

**No code:** `--harness "cmd:<your command> ... {prompt} ..."` runs any CLI, with the placeholders `{prompt}`,
`{model}`, `{effort}` and `{workspace}`. Its output is stored but not parsed, so no usage is recorded.

**An OpenAI-compatible endpoint, without a CLI of your own:** add the provider to `OpenCode.PROVIDERS` (base URL and
the environment variable that holds the key) and run `--harness opencode --model <provider>/<model>`.

**A robot policy** (VLA, RL controller): `rbench bench --agent policy:my_pkg.my_policy:MyPolicy --tasks public`, or a
policy server over HTTP with `--agent remote:http://host:9000` (protocol in
[roboticsbench/agents/policy.py](roboticsbench/agents/policy.py)). These runs are turn-based: the world pauses while
the policy computes, unlike the real-time agent runs. `rbench bench --agent llm:<provider>/<model>` runs a built-in
tool-calling agent on the skill API, also turn-based.

To have Vals run your integration on the held-out tasks, send the harness code (or the command template), how to
install your CLI, the names of the environment variables it reads for authentication (never the keys themselves),
and the model name and reasoning setting to run.

## Licenses

Robot assets are downloaded at pinned commits: the G1 model and GR00T whole-body-control code
(NVlabs/GR00T-WholeBodyControl, Apache-2.0) and policy weights (NVIDIA Open Model License), and the Panda model
(google-deepmind/mujoco_menagerie, Apache-2.0).
