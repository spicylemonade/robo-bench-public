"""Skills for the G1 humanoid (walking via the GR00T WBC, arm IK, Dex3 hands)."""

from __future__ import annotations

import math

import numpy as np

from roboticsbench.control.ik import SiteIK
from roboticsbench.core.geometry import wrap_angle
from roboticsbench.robots.g1 import ARM_POSES, LEFT_ARM, RIGHT_ARM
from roboticsbench.skills.base import SkillAPI, SkillError, SkillResult, skill

HAND = {"type": "string", "enum": ["left", "right"], "description": "which hand"}
HANDS = {"type": "string", "enum": ["left", "right", "both"], "description": "which hand(s)"}


class HumanoidSkills(SkillAPI):
    robot_kind = "humanoid"

    def __init__(self, env, **kw):
        super().__init__(env, **kw)
        m = env.model
        self._ik = {
            "left": SiteIK(m, "left_palm", LEFT_ARM, posture=ARM_POSES["forward"][:7], rot_weight=0.0,
                           posture_gain=0.02),
            "right": SiteIK(m, "right_palm", RIGHT_ARM, posture=ARM_POSES["forward"][7:], rot_weight=0.0,
                            posture_gain=0.02),
        }
        self._hand_geoms = {}
        for side in ("left", "right"):
            bodies = {b for b in range(m.nbody) if m.body(b).name.startswith((f"{side}_wrist", f"{side}_hand"))}
            self._hand_geoms[side] = {g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies}

    # ------------------------------------------------------------ internals
    def _stop(self, settle: float = 0.5) -> None:
        self.robot.vel_cmd[:] = 0.0
        self.env.run_for(settle)

    def _state_msg(self) -> str:
        r = self.robot
        p = r.base_pos
        return f"robot at ({p[0]:.2f}, {p[1]:.2f}) heading {math.degrees(r.yaw):.0f} deg"

    CROUCH_WALK_BELOW = 0.68  # pelvis command below which the WBC cannot turn in place or walk slowly
    CROUCH_MIN_VX = 0.5  # m/s: slower forward commands stall when crouched

    def _drive_to(self, x: float, y: float, speed: float, tol: float = 0.06, timeout: float | None = None,
                  stop: bool = True) -> tuple[bool, str]:
        r, env = self.robot, self.env
        target = np.array([x, y], float)
        dist0 = float(np.linalg.norm(target - r.base_pos[:2]))
        timeout = timeout or (6.0 + 2.2 * dist0 / max(speed, 0.1))
        t_end = env.t + timeout
        best, last_prog = dist0, env.t
        speed = float(np.clip(speed, 0.1, 0.9))
        crouched = r.height_cmd < self.CROUCH_WALK_BELOW
        reached = False
        while env.running and env.t < t_end:
            p, yaw = r.base_pos[:2], r.yaw
            e = target - p
            dist = float(np.linalg.norm(e))
            if dist < tol:
                reached = True
                break
            if dist < best - 0.03:
                best, last_prog = dist, env.t
            elif env.t - last_prog > 3.0:
                if stop:
                    self._stop()
                return False, f"blocked / not making progress ({dist:.2f} m from target); {self._state_msg()}"
            herr = wrap_angle(math.atan2(e[1], e[0]) - yaw)
            ex = math.cos(yaw) * e[0] + math.sin(yaw) * e[1]
            ey = -math.sin(yaw) * e[0] + math.cos(yaw) * e[1]
            if crouched:
                # the crouched gait only walks forward (>= 0.5 m/s) with gentle steering
                if ex < tol:  # reached / passed the target along the heading
                    reached = dist < 0.3
                    break
                if abs(herr) > 0.6 and dist > 0.3:
                    if stop:
                        self._stop()
                    return False, ("cannot turn toward the target while crouched (pelvis below "
                                   f"{self.CROUCH_WALK_BELOW:.2f} m): stand up with set_height(0.74), turn, then "
                                   f"crouch again; {self._state_msg()}")
                vx = max(self.CROUCH_MIN_VX, min(speed, 0.9))
                vy = 0.0
                wz = float(np.clip(1.2 * herr, -0.3, 0.3))
            elif dist > 0.45:
                wz = float(np.clip(1.6 * herr, -0.8, 0.8))
                if abs(herr) > 0.7:
                    vx, vy = 0.05, 0.0
                    wz = math.copysign(max(abs(wz), 0.3), herr)
                    last_prog = env.t  # turning toward the target counts as progress (targets behind the robot)
                else:
                    vx = min(speed, 0.9 * dist) * max(math.cos(herr), 0.0)
                    vy = float(np.clip(0.6 * ey, -0.12, 0.12))
            else:
                vx = float(np.clip(1.0 * ex, -0.25, 0.35))
                vy = float(np.clip(1.0 * ey, -0.25, 0.25))
                wz = 0.0
                if math.hypot(vx, vy) < 0.08:  # stay out of the balance/walk switching band
                    s = 0.08 / max(math.hypot(vx, vy), 1e-6)
                    vx, vy = vx * s, vy * s
            r.vel_cmd[:] = [vx, vy, wz]
            env.tick()
        if stop:
            self._stop()
        if not env.running:
            return reached, f"episode ended while walking ({env.status}: {env.fail_reason})"
        err = float(np.linalg.norm(target - r.base_pos[:2]))
        if not reached and not err < tol + 0.05:
            return False, f"timed out walking ({err:.2f} m from target); {self._state_msg()}"
        return True, f"arrived ({100 * err:.0f} cm from target); {self._state_msg()}"

    def _face(self, heading: float, timeout: float = 12.0) -> tuple[bool, str]:
        r, env = self.robot, self.env
        if r.height_cmd < self.CROUCH_WALK_BELOW and abs(wrap_angle(heading - r.yaw)) > math.radians(6):
            return False, (f"cannot turn in place while crouched (pelvis below {self.CROUCH_WALK_BELOW:.2f} m); "
                           f"stand up with set_height(0.74) first; heading {math.degrees(r.yaw):.0f} deg")
        t_end = env.t + timeout
        ok = False
        while env.running and env.t < t_end:
            err = wrap_angle(heading - r.yaw)
            if abs(err) < math.radians(3):
                ok = True
                break
            wz = float(np.clip(1.8 * err, -0.8, 0.8))
            wz = math.copysign(max(abs(wz), 0.2), wz)
            r.vel_cmd[:] = [0.0, 0.0, wz]
            env.tick()
        self._stop(0.4)
        err = math.degrees(wrap_angle(heading - r.yaw))
        return ok or abs(err) < 6, f"heading now {math.degrees(r.yaw):.0f} deg (error {err:+.1f})"

    # --------------------------------------------------------------- skills
    @skill("Walk to a floor point (world frame, metres) with closed-loop feedback, turning toward it first. "
           "The path is a straight line and does NOT avoid obstacles - plan waypoints yourself. "
           "Optionally finish facing heading_deg (0 = +x, 90 = +y).",
           x={"type": "number"}, y={"type": "number"},
           heading_deg={"type": "number", "optional": True},
           speed={"type": "number", "description": "max forward speed in m/s (0.1 - 0.9)"})
    def walk_to(self, x: float, y: float, heading_deg: float | None = None, speed: float = 0.6) -> SkillResult:
        ok, msg = self._drive_to(x, y, speed)
        if ok and heading_deg is not None and self.env.running:
            ok2, msg2 = self._face(math.radians(heading_deg))
            ok, msg = ok and ok2, msg + "; " + msg2
        return SkillResult(ok, msg + self._outcome_suffix())

    @skill("Walk relative to the robot's current pose: `forward` and `left` metres in the robot's frame, "
           "then turn by `turn_deg` (+ = counter-clockwise).",
           forward={"type": "number"}, left={"type": "number"}, turn_deg={"type": "number"},
           speed={"type": "number", "description": "max forward speed in m/s (0.1 - 0.9)"})
    def walk(self, forward: float = 0.0, left: float = 0.0, turn_deg: float = 0.0, speed: float = 0.6) -> SkillResult:
        r = self.robot
        yaw0 = r.yaw
        b = r.base_pos
        msg, ok = "", True
        if abs(forward) > 1e-3 or abs(left) > 1e-3:
            tx = b[0] + math.cos(yaw0) * forward - math.sin(yaw0) * left
            ty = b[1] + math.sin(yaw0) * forward + math.cos(yaw0) * left
            if forward < -0.05 and abs(left) < 0.3 and abs(forward) < 1.5:
                ok, msg = self._backstep(tx, ty, yaw0)
            else:
                ok, msg = self._drive_to(tx, ty, speed, tol=0.03)
        if abs(turn_deg) > 0.5 and self.env.running:
            ok2, msg2 = self._face(yaw0 + math.radians(turn_deg))
            ok, msg = ok and ok2, (msg + "; " if msg else "") + msg2
        return SkillResult(ok, (msg or "no motion requested") + self._outcome_suffix())

    def _backstep(self, tx: float, ty: float, yaw0: float) -> tuple[bool, str]:
        """Walk backwards (keeping heading) to (tx, ty)."""
        r, env = self.robot, self.env
        t_end = env.t + 10.0
        while env.running and env.t < t_end:
            e = np.array([tx, ty]) - r.base_pos[:2]
            if np.linalg.norm(e) < 0.1:
                break
            yaw = r.yaw
            ex = math.cos(yaw) * e[0] + math.sin(yaw) * e[1]
            ey = -math.sin(yaw) * e[0] + math.cos(yaw) * e[1]
            r.vel_cmd[:] = [float(np.clip(ex, -0.35, 0.2)), float(np.clip(ey, -0.2, 0.2)),
                            float(np.clip(1.5 * wrap_angle(yaw0 - yaw), -0.5, 0.5))]
            env.tick()
        self._stop()
        return True, f"stepped back; {self._state_msg()}"

    @skill("Turn in place by `degrees` (+ = counter-clockwise / left, - = right).", degrees={"type": "number"})
    def turn(self, degrees: float) -> SkillResult:
        ok, msg = self._face(self.robot.yaw + math.radians(degrees))
        return SkillResult(ok, msg + self._outcome_suffix())

    @skill("Turn in place to face an absolute world heading in degrees (0 = +x axis, 90 = +y axis).",
           heading_deg={"type": "number"})
    def face(self, heading_deg: float) -> SkillResult:
        ok, msg = self._face(math.radians(heading_deg))
        return SkillResult(ok, msg + self._outcome_suffix())

    @skill("Low-level locomotion: command body velocity for `duration` seconds, then stop. vx forward "
           "[-0.6, 1.0] m/s, vy left [-0.5, 0.5] m/s, yaw_rate_deg [-57, 57] deg/s. The achieved speed is ~90% of "
           "the command and ramps up over ~0.5 s.",
           vx={"type": "number"}, vy={"type": "number"}, yaw_rate_deg={"type": "number"},
           duration={"type": "number", "description": "seconds (0.1 - 10)"})
    def set_velocity(self, vx: float, vy: float = 0.0, yaw_rate_deg: float = 0.0, duration: float = 1.0) -> SkillResult:
        r = self.robot
        r.vel_cmd[:] = [vx, vy, math.radians(yaw_rate_deg)]
        self.env.run_for(float(np.clip(duration, 0.02, 10.0)))
        self._stop(0.3)
        return SkillResult(True, self._state_msg() + self._outcome_suffix())

    @skill("Stop walking and stand still.")
    def stop(self) -> SkillResult:
        self._stop(0.6)
        return SkillResult(True, self._state_msg() + self._outcome_suffix())

    @skill("Set the pelvis height (m). 0.74 = normal standing (top of the head ~1.28 m above the floor). Crouch to "
           "duck under low obstacles: at 0.62 the head top is ~1.16 m and the robot can still walk FORWARD, but only "
           "with little steering and slowly (~0.25-0.35 m/s at speed 0.6; ~0.7 m/s at speed 0.9, overshooting "
           "~0.3 m) - it cannot turn in place below 0.68 (line up first, then crouch). "
           "Below 0.60 it cannot walk at all (min 0.55, e.g. to reach low objects). Raise back to 0.74 afterwards.",
           height={"type": "number"})
    def set_height(self, height: float) -> SkillResult:
        r = self.robot
        h = float(np.clip(height, 0.55, 0.78))
        r.height_cmd = h
        t_end = self.env.t + 3.0
        while self.env.running and self.env.t < t_end and abs(r._height - h) > 1e-3:
            self.env.tick()
        self.env.run_for(0.6)
        return SkillResult(True, f"pelvis height {r.base_pos[2]:.3f} m (target {h:.2f})" + self._outcome_suffix())

    @skill("Move one hand's palm to a 3D point with arm inverse kinematics (legs keep balancing). "
           "frame='world' uses world coordinates; frame='robot' uses x forward / y left from the pelvis "
           "and z = height above the floor. The palm point reaches ~0.2-0.49 m in front of the pelvis at heights "
           "~0.7-1.2 m (open fingers stick out ~8 cm beyond it, a closed fist ~4-5 cm). The arm is compliant: if the "
           "hand touches something and stops moving it holds there and reports the contact.",
           hand=HAND, x={"type": "number"}, y={"type": "number"}, z={"type": "number"},
           frame={"type": "string", "enum": ["world", "robot"]})
    def reach(self, hand: str, x: float, y: float, z: float, frame: str = "world") -> SkillResult:
        r, env = self.robot, self.env
        if hand not in ("left", "right"):
            raise SkillError("hand must be 'left' or 'right'")
        target = np.array([x, y, z], float)
        if frame == "robot":
            target = r.robot_to_world(target)
        ik = self._ik[hand]
        sl = slice(0, 7) if hand == "left" else slice(7, 14)
        t_start, t_end = env.t, env.t + 4.0
        good = 0
        best, t_best = float("inf"), env.t
        blocked = None
        while env.running and env.t < t_end:
            sol = ik.solve(env.data.qpos, target, iters=8, q_init=r.arm_target[sl])
            r.arm_target[sl] = sol.q
            env.tick()
            err = float(np.linalg.norm(target - r.palm_pos(hand)))
            good = good + 1 if err < 0.012 else 0
            if good >= 10 and env.t - t_start > 0.5:
                break
            if err < best - 0.002:
                best, t_best = err, env.t
            elif env.t - t_best > 0.3:
                blocked = self._hand_contact(hand)
                if blocked:
                    # compliant stop: hold the arm where it is instead of shoving the obstacle (and the robot)
                    r.arm_target[sl] = env.data.qpos[r.q_adr[15:]][sl]
                    env.run_for(0.2)
                    break
        err = float(np.linalg.norm(r.palm_pos(hand) - target))
        p = r.palm_pos(hand)
        ok = err < 0.04
        msg = f"{hand} palm at ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}), {100 * err:.1f} cm from target"
        if blocked:
            msg += f" - hand stopped by contact with {blocked}"
        elif not ok:
            msg += " - target may be out of reach; step closer or adjust height"
        return SkillResult(ok, msg + self._outcome_suffix())

    def _hand_contact(self, hand: str) -> str | None:
        """Name of a non-robot object touching this arm's wrist/hand (floor excluded), if any."""
        m, d = self.env.model, self.env.data
        geoms = self._hand_geoms[hand]
        rg = self.robot.geom_ids
        for i in range(d.ncon):
            c = d.contact[i]
            for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if a in geoms and b not in rg:
                    name = m.geom(b).name or m.body(m.geom_bodyid[b]).name
                    if name != "floor":
                        return name
        return None

    @skill("Put both arms into a preset posture: 'zero' (default, forearms forward), 'ready' (relaxed), "
           "'forward' (arms reaching forward), 'push' (arms straight ahead at chest height, good for "
           "pushing objects or doors), 'tucked' (arms close to the body for narrow gaps).",
           pose={"type": "string", "enum": sorted(ARM_POSES)})
    def arm_pose(self, pose: str) -> SkillResult:
        if pose not in ARM_POSES:
            raise SkillError(f"unknown pose {pose!r}")
        self.robot.arm_target[:] = ARM_POSES[pose]
        self.env.run_for(1.2)
        return SkillResult(True, f"arms in '{pose}' posture" + self._outcome_suffix())

    @skill("Close hand(s) into a grasp.", hand=HANDS)
    def grasp(self, hand: str = "both") -> SkillResult:
        idx = {"left": [0], "right": [1], "both": [0, 1]}[hand]
        self.robot.hand_cmd[idx] = 1.0
        self.env.run_for(0.7)
        return SkillResult(True, f"{hand} hand closed" + self._outcome_suffix())

    @skill("Open hand(s).", hand=HANDS)
    def release(self, hand: str = "both") -> SkillResult:
        idx = {"left": [0], "right": [1], "both": [0, 1]}[hand]
        self.robot.hand_cmd[idx] = 0.0
        self.env.run_for(0.6)
        return SkillResult(True, f"{hand} hand opened" + self._outcome_suffix())
