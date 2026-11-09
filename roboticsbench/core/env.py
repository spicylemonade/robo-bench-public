"""The simulation environment: one task + one robot in one MuJoCo world.

Time only advances through :meth:`Env.tick` (one control step, 50 Hz), so the world is frozen
while an agent (e.g. an LLM) is thinking -- episodes are turn-based in wall-clock time but
physically continuous in simulated time (the referee's real-time mode instead ticks from a clock thread).

All OpenGL work of an Env (renderer creation, rendering, freeing) runs on one dedicated thread (:class:`GLThread`):
a GL context belongs to the thread that made it current (Apple CGL keeps it locked, EGL refuses to make it current
anywhere else), so rendering from two threads - the real-time clock recording video, the HTTP thread serving
``look`` - would otherwise deadlock or fail. :meth:`Env.capture` copies the state (``mj_copyData``, ~0.2 ms) and
renders the copy there, so the physics never waits for graphics.
"""

from __future__ import annotations

import math
import os
import queue
import threading
import time
import weakref
from collections import deque
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Callable

import mujoco
import numpy as np

from roboticsbench.core.jsonutil import jsonable
from roboticsbench.core.scene import SceneBuilder, choose_collision_path
from roboticsbench.tasks.base import Evaluation, Task, make_task

RUNNING, SUCCESS, FAILURE, TIMEOUT = "running", "success", "failure", "timeout"


class Env:
    def __init__(self, task: str | Task, seed: int = 0, *, width: int = 640, height: int = 480,
                 obs_cameras: tuple[str, ...] | None = None, task_options: dict[str, Any] | None = None,
                 control: str = "skills"):
        self.task: Task = task if isinstance(task, Task) else make_task(task, **(task_options or {}))
        self.seed = int(seed)
        self.width, self.height = width, height
        self.control = control  # "skills" | "motor" | "velocity": how the robot is driven (see core.motor)
        self.vision_only = False  # set by the referee: no ground-truth object state may reach the agent
        self.robot = self.task.make_robot()
        if control == "motor":
            self.robot.control_mode = "motor"
        spec = self.robot.build_spec()
        self.scene = SceneBuilder(spec)
        self.scene.setup_world(**self.task.world)
        self.build_rng = np.random.default_rng([self.seed, 0xB0B])
        self.task.build(self.scene, self.build_rng)
        self.scene.finish()
        self.model = spec.compile()
        choose_collision_path(self.model)
        self.data = mujoco.MjData(self.model)
        self.robot.bind(self.model, self.data)
        self.task.bind(self)
        self.control_dt = self.robot.control_dt
        self.n_sub = max(1, int(round(self.control_dt / self.model.opt.timestep)))
        cams = tuple(obs_cameras or self.task.cameras or self.robot.default_cameras)
        if self.robot.agent_cameras is not None:
            # the robot fixes what an agent may see, whatever the task has: the humanoid only its head camera, the arm
            # only the workcell's top camera and its wrist camera - the same set in every task
            cams = tuple(self.robot.agent_cameras)
            have = {self.model.camera(i).name for i in range(self.model.ncam)}
            missing = [c for c in cams if c not in have]
            if missing:
                raise ValueError(f"{self.task.name}: the scene lacks the {self.robot.name} agent camera(s) {missing}")
        self.obs_cameras = cams  # what an agent may look through (render() itself works for any camera)
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}  # owned by the GL thread (see gl())
        self._gl_thread: GLThread | None = None
        self._gl_start = threading.Lock()
        self._snapshots: list[mujoco.MjData] = []  # recycled state copies for capture()
        self._snap_lock = threading.Lock()
        self._video: VideoRecorder | None = None
        self.caption = ""
        self.tick_callbacks: list[Callable[[Env], None]] = []
        self.reset()

    # ----------------------------------------------------------------- episode
    def reset(self, seed: int | None = None) -> None:
        if seed is not None and seed != self.seed:
            raise ValueError("layout depends on the seed; create a new Env for a new seed")
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        self.rng = np.random.default_rng([self.seed, 0xE9])
        self.robot.reset(self.rng, **self.task.robot_start(self.rng))
        self.task.milestones.reset()
        self.task.reset(self, self.rng)
        mujoco.mj_forward(m, d)
        self.status = RUNNING
        self.fail_reason = ""
        self.progress = 0.0
        self.last_eval = Evaluation()
        self._success_time = 0.0
        self.n_falls = 0  # times the robot went down (humanoid episodes run on after a fall)
        self._fallen = False
        self.ticks = 0
        self.t0 = 0.0
        self.wall_start = time.time()
        # settle: let objects come to rest and the robot find its balance
        for _ in range(int(0.6 / self.control_dt)):
            self._physics_tick()
        if hasattr(self.robot, "tare"):
            self.robot.tare()
        self.t0 = float(d.time)
        self.ticks = 0
        self.task.milestones.reset()
        self.task.step(self)
        self.last_eval = self.task.evaluate(self)

    @property
    def t(self) -> float:
        """Episode time (simulated seconds since reset)."""
        return float(self.data.time) - self.t0

    @property
    def running(self) -> bool:
        return self.status == RUNNING

    @property
    def time_left(self) -> float:
        return max(0.0, self.task.time_limit - self.t)

    def _physics_tick(self) -> None:
        self.robot.pre_tick()
        for _ in range(self.n_sub):
            self.robot.substep()
            mujoco.mj_step(self.model, self.data)

    def tick(self, n: int = 1) -> bool:
        """Advance ``n`` control ticks (stops early when the episode ends). Returns ``running``."""
        for _ in range(n):
            if self.status != RUNNING:
                return False
            t_before = float(self.data.time)
            self._physics_tick()
            self.ticks += 1
            # MuJoCo resets the whole state (and rewinds time) when accelerations blow up: never let that pass silently
            if not np.all(np.isfinite(self.data.qpos)) or float(self.data.time) < t_before:
                self.data.time = t_before  # MuJoCo reset the state: keep the episode clock where it ended
                self.status, self.fail_reason = FAILURE, "physics diverged"
                return False
            self.task.step(self)
            ev = self.task.evaluate(self)
            self.last_eval = ev
            self.progress = max(self.progress, min(ev.progress, 0.99))  # only a held success scores 1.0
            fallen = self.robot.is_fallen() and not self.task.allows_low_posture(self)
            self.n_falls += int(fallen and not self._fallen)
            self._fallen = fallen
            if ev.failed and self.task.ends_on_failure:
                self.status, self.fail_reason = FAILURE, ev.reason or "task failure"
            elif fallen and self.task.ends_on_failure:
                self.status, self.fail_reason = FAILURE, "robot fell over"
            elif ev.success and not ev.failed:
                self._success_time += self.control_dt
                if self._success_time >= self.task.success_hold - 1e-9:
                    self.status = SUCCESS
                    self.progress = 1.0
            else:
                self._success_time = 0.0
            if self.status == RUNNING and self.t >= self.task.time_limit:
                self.end(TIMEOUT, "time limit reached")
            for cb in self.tick_callbacks:
                cb(self)
            if self._video is not None:
                self._video.maybe_capture(self)
        return self.status == RUNNING

    def run_for(self, seconds: float) -> bool:
        return self.tick(max(1, int(round(seconds / self.control_dt))))

    def end(self, status: str, reason: str = "") -> None:
        if self.status == RUNNING:
            ev = self.last_eval
            if ev is not None and ev.failed:  # a task rule broken earlier (tasks that run on after one)
                status, reason = FAILURE, ev.reason or "task failure"
            self.status, self.fail_reason = status, reason

    def result(self) -> dict[str, Any]:
        ev = self.last_eval
        return {
            "task": self.task.name,
            "seed": self.seed,
            "status": self.status,
            "success": self.status == SUCCESS,
            "progress": round(float(self.progress if self.status != SUCCESS else 1.0), 4),
            "reason": self.fail_reason or ev.reason,
            "milestones": self.task.milestones.names(),
            "sim_time": round(self.t, 3),
            "falls": self.n_falls,
            "metrics": jsonable(ev.metrics),
        }

    # --------------------------------------------------------------- rendering
    def gl(self, fn: Callable[[], Any]) -> Any:
        """Run ``fn`` on this Env's GL thread and return its result. Anything that touches the renderers
        (``_renderer``) must go through here; callers still serialise access to ``data`` themselves."""
        return self._gl().submit(fn).result()

    def _gl(self) -> GLThread:
        if self._gl_thread is None:
            with self._gl_start:
                if self._gl_thread is None:
                    thread = GLThread()
                    # close() - or garbage collection of an Env nobody closed - frees the renderers on their thread
                    self._gl_free = weakref.finalize(self, _free_gl, thread, self._renderers)
                    self._gl_thread = thread
        return self._gl_thread

    def _renderer(self, w: int, h: int) -> mujoco.Renderer:
        """The renderer for this image size (GL thread only - see :meth:`gl`)."""
        key = (w, h)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, h, w)
        return self._renderers[key]

    def capture(self, camera: str | None = None, width: int | None = None, height: int | None = None, *,
                depth: bool = False, then: Callable[[np.ndarray], Any] | None = None) -> Future:
        """Photograph the scene as it is now and render it on the GL thread: returns a Future of the image (RGB, or
        depth in metres), or of ``then(image)``, which also runs on the GL thread (jobs run in submission order).
        Call it from whoever owns ``data`` at the moment (the ticking thread, or under the referee lock); the
        simulation can go on right away. ``"a+b"`` puts cameras side by side."""
        camera = camera or self.task.video_camera or self.robot.video_camera
        w, h = width or self.width, height or self.height
        views = [self._free_camera(c) or c for c in camera.split("+")]
        snap = self._snapshot()

        def job() -> Any:
            try:
                imgs = [self._draw(snap, v, w, h, depth) for v in views]
            finally:
                self._recycle(snap)
            img = imgs[0] if len(imgs) == 1 else np.concatenate(imgs, axis=1)
            return img if then is None else then(img)

        return self._gl().submit(job)

    def render(self, camera: str | None = None, width: int | None = None, height: int | None = None) -> np.ndarray:
        return self.capture(camera, width, height).result()

    def _draw(self, data: mujoco.MjData, view: str | mujoco.MjvCamera, w: int, h: int, depth: bool) -> np.ndarray:
        r = self._renderer(w, h)
        if depth:
            r.enable_depth_rendering()
        try:
            r.update_scene(data, camera=view)
            return r.render().copy()
        finally:
            if depth:
                r.disable_depth_rendering()

    def _snapshot(self) -> mujoco.MjData:
        with self._snap_lock:
            d = self._snapshots.pop() if self._snapshots else None
        if d is None:
            d = mujoco.MjData(self.model)
        mujoco.mj_copyData(d, self.model, self.data)
        return d

    def _recycle(self, d: mujoco.MjData) -> None:
        with self._snap_lock:
            if len(self._snapshots) < 8:
                self._snapshots.append(d)

    def _free_camera(self, camera: str) -> mujoco.MjvCamera | None:
        if camera == "chase" and hasattr(self.robot, "chase_view"):
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            lookat, dist, az, el = self.robot.chase_view()
            cam.lookat[:] = lookat
            cam.distance, cam.azimuth, cam.elevation = dist, az, el
            return cam
        return None

    def camera_calibration(self, camera: str, width: int | None = None, height: int | None = None) -> dict[str, Any]:
        """Pinhole calibration of a camera at this instant (what a calibrated real camera provides): world pose and
        intrinsics. Camera frame (MuJoCo / OpenGL convention): x = image right, y = image up, the camera looks along
        -z. A world point p is seen at c = R^T (p - position); pixel u = cx + fx * c_x / (-c_z),
        v = cy - fy * c_y / (-c_z) (u right, v down, origin at the top-left pixel corner)."""
        w, h = width or self.width, height or self.height
        m, d = self.model, self.data
        free = self._free_camera(camera)
        if free is not None:
            az, el = math.radians(free.azimuth), math.radians(free.elevation)
            fwd = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
            pos = np.asarray(free.lookat) - free.distance * fwd
            right = np.cross(fwd, [0.0, 0.0, 1.0])
            right /= np.linalg.norm(right)
            up = np.cross(right, fwd)
            R = np.stack([right, up, -fwd], axis=1)
            fovy, ortho = float(m.vis.global_.fovy), False
        else:
            cid = m.camera(camera).id
            pos, R = d.cam_xpos[cid].copy(), d.cam_xmat[cid].reshape(3, 3).copy()
            fovy = float(m.cam_fovy[cid])
            ortho = int(m.cam_projection[cid]) == int(mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC)
        out: dict[str, Any] = {"camera": camera, "width": int(w), "height": int(h),
                               "position": [round(float(x), 5) for x in pos],
                               "rotation": [[round(float(x), 6) for x in row] for row in R]}
        if ortho:  # MuJoCo: fovy of an orthographic camera is the view height in metres
            out.update(projection="orthographic", view_height_m=fovy, metres_per_pixel=fovy / h)
        else:
            f = (h / 2.0) / math.tan(math.radians(fovy) / 2.0)
            out.update(projection="perspective", fovy_deg=round(fovy, 4), fx=round(f, 4), fy=round(f, 4),
                       cx=w / 2.0, cy=h / 2.0)
        return out

    def render_depth(self, camera: str, width: int | None = None, height: int | None = None) -> np.ndarray:
        """Depth image (metres along the optical axis, like an RGB-D camera) for ``camera``."""
        return self.capture(camera, width, height, depth=True).result()

    def observe(self, cameras: tuple[str, ...] | None = None, images: bool = True,
                width: int | None = None, height: int | None = None) -> dict[str, Any]:
        obs: dict[str, Any] = {
            "time": round(self.t, 2),
            "time_left": round(self.time_left, 1),
            "status": self.status,
            "robot": jsonable(self.robot.proprio()),
            "task": jsonable(self.task.status(self)),
        }
        if images:
            obs["images"] = {c: self.render(c, width, height) for c in (cameras or self.obs_cameras)}
        return obs

    # ------------------------------------------------------------------- video
    def start_video(self, path: str | os.PathLike, camera: str | None = None, fps: int = 25,
                    width: int = 640, height: int = 480, hud: bool = True) -> None:
        self.stop_video()
        self._video = VideoRecorder(Path(path), camera or self.task.video_camera or self.robot.video_camera,
                                    fps, width, height, hud)
        self._video.capture(self)

    def stop_video(self) -> Path | None:
        if self._video is None:
            return None
        v, self._video = self._video, None
        v.capture(self)
        return v.close()

    def close(self) -> None:
        self.stop_video()
        if self._gl_thread is not None:
            self._gl_free()
            self._gl_thread = None


def _free_gl(thread: GLThread, renderers: dict[tuple[int, int], mujoco.Renderer]) -> None:
    def free() -> None:
        for r in renderers.values():
            r.close()
        renderers.clear()

    thread.submit(free)
    thread.stop()


class GLThread:
    """A worker thread that runs every GL call of one Env, in submission order (see the module docstring)."""

    def __init__(self) -> None:
        self._jobs: queue.SimpleQueue = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._loop, name="rbench-gl", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            fn, fut = job
            if fut.set_running_or_notify_cancel():
                try:
                    fut.set_result(fn())
                except BaseException as e:  # noqa: BLE001 - delivered through the Future
                    fut.set_exception(e)

    def submit(self, fn: Callable[[], Any]) -> Future:
        fut: Future = Future()
        if threading.current_thread() is self._thread:  # nested call from a job: run it inline
            try:
                fut.set_result(fn())
            except BaseException as e:  # noqa: BLE001
                fut.set_exception(e)
        else:
            self._jobs.put((fn, fut))
        return fut

    def stop(self) -> None:
        self._jobs.put(None)


class VideoRecorder:
    def __init__(self, path: Path, camera: str, fps: int, width: int, height: int, hud: bool):
        import imageio.v2 as imageio

        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self.camera, self.fps, self.w, self.h, self.hud = path, camera, fps, width, height, hud
        self.writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=7,
                                         macro_block_size=8, ffmpeg_log_level="error")
        self.next_t = 0.0
        self.frames = 0
        self.pending: deque[Future] = deque()

    def maybe_capture(self, env: Env) -> None:
        if env.t + 1e-9 >= self.next_t:
            self.capture(env)

    def capture(self, env: Env) -> None:
        hud = hud_info(env) if self.hud else None
        self.pending.append(env.capture(self.camera, self.w, self.h, then=lambda img: self._write(img, hud)))
        while self.pending and self.pending[0].done():
            self.pending.popleft().result()  # surfaces a failed frame
        if len(self.pending) > 24:  # at most ~2 s of video in flight
            self.pending.popleft().result()
        self.frames += 1
        # a fixed schedule (not "now + 1/fps", which rounds every interval up to whole ticks): one frame per
        # 1/fps of simulated time, so the video plays in real time
        self.next_t = max(self.next_t + 1.0 / self.fps, env.t)

    def _write(self, img: np.ndarray, hud: dict[str, Any] | None) -> None:  # GL thread, in capture order
        self.writer.append_data(draw_hud(img, hud) if hud is not None else img)

    def close(self) -> Path:
        while self.pending:
            self.pending.popleft().result()
        self.writer.close()
        return self.path


_FONTS: dict[int, Any] = {}


def _font(size: int = 15):
    if size not in _FONTS:
        from PIL import ImageFont

        font = None
        for cand in ("/System/Library/Fonts/Menlo.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                     "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
            if os.path.exists(cand):
                try:
                    font = ImageFont.truetype(cand, size)
                    break
                except OSError:
                    pass
        if font is None:
            try:  # Pillow >= 10.1: scalable built-in font
                font = ImageFont.load_default(size=size)
            except TypeError:
                font = ImageFont.load_default()
        _FONTS[size] = font
    return _FONTS[size]


def hud_info(env: Env) -> dict[str, Any]:
    """What the HUD shows, taken at capture time (the frame is drawn later on the GL thread)."""
    return {"task": env.task.name, "seed": env.seed, "t": float(env.t), "status": env.status,
            "progress": float(env.progress), "caption": env.caption}


def draw_hud(img: np.ndarray, env: Env | dict[str, Any]) -> np.ndarray:
    from PIL import Image, ImageDraw

    hud = env if isinstance(env, dict) else hud_info(env)
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im, "RGBA")
    W = im.width
    size = int(np.clip(W // 42, 10, 15))
    f = _font(size)
    bar = size + 7
    right = f"{100 * (1.0 if hud['status'] == SUCCESS else hud['progress']):3.0f}%  {hud['status'].upper()}"
    rw = dr.textlength(right, font=f)
    top = ""
    for cand in (f"{hud['task']}  seed={hud['seed']}  t={hud['t']:5.1f}s", f"{hud['task']}  t={hud['t']:5.1f}s",
                 f"t={hud['t']:5.1f}s"):
        top = cand
        if dr.textlength(cand, font=f) + rw + 18 <= W:
            break
    dr.rectangle([0, 0, W, bar], fill=(0, 0, 0, 140))
    dr.text((6, 3), top, font=f, fill=(255, 255, 255, 255))
    color = {SUCCESS: (120, 255, 120, 255), FAILURE: (255, 120, 120, 255)}.get(hud['status'], (255, 255, 255, 255))
    dr.text((W - rw - 6, 3), right, font=f, fill=color)
    if hud['caption']:
        cap = hud['caption']
        while len(cap) > 8 and dr.textlength(cap, font=f) > W - 12:
            cap = cap[:-4] + "..." if not cap.endswith("...") else cap[:-4] + "..."
        dr.rectangle([0, im.height - bar, W, im.height], fill=(0, 0, 0, 140))
        dr.text((6, im.height - bar + 3), cap, font=f, fill=(255, 235, 150, 255))
    return np.asarray(im)


def angle_diff_deg(a: float, b: float) -> float:
    return math.degrees((a - b + math.pi) % (2 * math.pi) - math.pi)
