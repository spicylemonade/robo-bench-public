"""The robot's datasheet for motor-control agents.

What a roboticist gets with a real robot: the exact motor and sensor interface (keys, shapes, units, frames, rates),
joint axes and sign conventions, limits, gains, masses and dimensions. Generated from the simulation model at the
start of the episode, so every number is exact; rendered into ROBOT.md (:func:`spec_markdown`) and saved as
``robot_spec.json`` in the agent's workspace. Nothing in here is about the task.
"""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

_DIRS = (("forward", "back"), ("left", "right"), ("up", "down"))


def _r(x: Any, nd: int = 4) -> Any:
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_r(v, nd) for v in np.asarray(x, float).reshape(-1)]
    return round(float(x), nd) + 0.0


class _Frame:
    """The robot frame used throughout the datasheet: G1 - origin on the floor under the pelvis at the start, x =
    the robot's heading, y = left, z = up; Panda - the base frame (identical to the world frame here)."""

    def __init__(self, env):
        r = env.robot
        if r.name == "g1":
            b = r.base_pos
            self.origin, self.yaw = np.array([b[0], b[1], 0.0]), float(r.yaw)
        else:
            self.origin, self.yaw = np.zeros(3), 0.0
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])  # robot -> world

    def point(self, p: np.ndarray) -> np.ndarray:
        return self.R.T @ (np.asarray(p, float) - self.origin)

    def vec(self, v: np.ndarray) -> np.ndarray:
        return self.R.T @ np.asarray(v, float)


def _subtree(m: mujoco.MjModel, root: int) -> list[int]:
    out = [root]
    for b in range(root + 1, m.nbody):
        p = m.body_parentid[b]
        while p > root:
            p = m.body_parentid[p]
        if p == root:
            out.append(b)
    return out


def _local_point(d: mujoco.MjData, body: int, local: np.ndarray) -> np.ndarray:
    return d.xpos[body] + d.xmat[body].reshape(3, 3) @ local


def _words(dv_cm: np.ndarray, thresh: float = 0.15) -> str:
    parts = []
    for k, (pos, neg) in enumerate(_DIRS):
        v = float(dv_cm[k])
        if abs(v) >= thresh:
            parts.append(f"{abs(v):.1f} cm {pos if v > 0 else neg}")
    return ", ".join(parts) if parts else "(rotates in place)"


def _axis_words(a: np.ndarray) -> str:
    names = ("x", "y", "z")
    k = int(np.argmax(np.abs(a)))
    sign = "+" if a[k] > 0 else "-"
    return f"{sign}{names[k]}" if abs(a[k]) > 0.985 else "[" + ", ".join(f"{v:.2f}" for v in a) + "]"


def build_spec(env, motor) -> dict[str, Any]:
    """Machine-readable datasheet. ``motor`` is the episode's :class:`~roboticsbench.core.motor.MotorInterface`."""
    m, d0 = env.model, env.data
    d = mujoco.MjData(m)
    mujoco.mj_copyData(d, m, d0)
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    fr = _Frame(env)
    robot = env.robot
    root = m.body(robot.root_body).id
    bodies = _subtree(m, root)
    spec: dict[str, Any] = {
        "robot": robot.name, "mode": motor.mode,
        "control_rate_hz": round(1.0 / env.control_dt), "tick_s": env.control_dt,
        "physics_rate_hz": round(1.0 / m.opt.timestep),
        "robot_frame": ("origin on the floor under the pelvis at the start, x = the robot's heading at the start, "
                        "y = left, z = up" if robot.name == "g1" else
                        "the base frame (= the world frame of this scene): origin at the base on the table surface, "
                        "x forward away from the base, y left, z up"),
        "mass_kg": _r(m.body_subtreemass[root], 2),
        "sensor_note": ("all sensor values are sampled at the end of every control tick and are exact: no noise, no "
                        "bias, no delay"),
    }

    def displacement(qadr: int, point_fn) -> np.ndarray:
        d2 = mujoco.MjData(m)
        mujoco.mj_copyData(d2, m, d)
        p0 = point_fn(d2)
        d2.qpos[qadr] += 0.1
        mujoco.mj_kinematics(m, d2)
        return 100.0 * fr.vec(point_fn(d2) - p0)

    def joint_geometry(name: str, point_name: str, point_fn) -> dict[str, Any]:
        j = m.joint(name).id
        da = m.jnt_dofadr[j]
        axis = fr.vec(d.xaxis[j])
        dv = displacement(m.jnt_qposadr[j], point_fn)
        return {"axis": _r(axis, 3), "axis_text": _axis_words(axis), "anchor": _r(fr.point(d.xanchor[j]), 4),
                "plus_0p1rad_moves": point_name, "plus_0p1rad_cm": _r(dv, 2), "plus_0p1rad_text": _words(dv),
                "armature": _r(m.dof_armature[da], 4), "damping": _r(m.dof_damping[da], 4),
                "frictionloss": _r(m.dof_frictionloss[da], 4)}

    if robot.name == "g1":
        _g1(spec, env, motor, m, d, fr, bodies, joint_geometry)
    else:
        _panda(spec, env, motor, m, d, fr, bodies, joint_geometry)
    spec["links"] = [{"name": m.body(b).name, "parent": m.body(m.body_parentid[b]).name, "mass_kg": _r(m.body_mass[b], 3),
                      "origin": _r(fr.point(d.xpos[b]), 4)} for b in bodies
                     if "hand_" not in m.body(b).name and "finger" not in m.body(b).name]
    spec["cameras"] = _cameras(env, m, d, fr)
    return spec


# ---------------------------------------------------------------------------------------------------- G1
def _g1(spec, env, motor, m, d, fr, bodies, joint_geometry) -> None:
    from roboticsbench.robots import g1 as G

    robot = env.robot
    sole = {}
    for side in ("left", "right"):
        b = m.body(f"{side}_ankle_roll_link").id
        pts = [m.geom_pos[g] for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_contype[g]
               and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
        rad = float(max(m.geom_size[g][0] for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_contype[g]
                        and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE))
        pts = np.array(pts)
        centre = pts.mean(axis=0) - np.array([0.0, 0.0, rad])
        sole[side] = (b, pts, rad, centre)

    def site_fn(name):
        sid = m.site(name).id
        return lambda dd: dd.site_xpos[sid].copy()

    def sole_fn(side):
        b, _, _, c = sole[side]
        return lambda dd: _local_point(dd, b, c)

    def body_point_fn(b, local):
        return lambda dd: _local_point(dd, b, local)

    def point_for(name: str):
        side = "left" if name.startswith("left") else "right" if name.startswith("right") else ""
        if "ankle_pitch" in name:  # the sole centre hardly moves: show the toe
            b, pts, rad, c = sole[side]
            return f"{side} toe", body_point_fn(b, np.array([pts[:, 0].max() + rad, 0.0, c[2]]))
        if "ankle_roll" in name:  # ... and the outer edge of the sole
            b, pts, rad, c = sole[side]
            y = pts[:, 1].max() + rad if side == "left" else pts[:, 1].min() - rad
            return f"{side} sole outer edge", body_point_fn(b, np.array([c[0], y, c[2]]))
        if any(k in name for k in ("hip", "knee")):
            return f"{side} sole centre", sole_fn(side)
        if name.startswith("waist"):
            return "head camera", site_fn("head_site")
        return f"{side} palm centre", site_fn(f"{side}_palm")

    groups = ["left leg"] * 6 + ["right leg"] * 6 + ["waist"] * 3 + ["left arm"] * 7 + ["right arm"] * 7
    start = robot.joint_target
    joints = []
    for i, n in enumerate(motor.joint_names):
        pname, pfn = point_for(n)
        row = {"index": i, "name": n, "group": groups[i], "min": _r(motor.q_lo[i]), "max": _r(motor.q_hi[i]),
               "torque_limit": _r(motor.tau_max[i], 2), "no_load_speed": _r(robot.dq_max[i], 1),
               "kp_default": _r(G.BODY_KP[i], 1),
               "kd_default": _r(G.BODY_KD[i], 2), "kp_start": _r(G.HOLD_KP[i], 1), "kd_start": _r(G.HOLD_KD[i], 2),
               "kp_max": _r(motor.kp_max[i], 1), "kd_max": _r(motor.kd_max[i], 2), "q_start": _r(start[i])}
        row.update(joint_geometry(n, pname, pfn))
        joints.append(row)
    spec["joints"] = joints
    hand = []
    for i, n in enumerate(motor.hand_names):
        a = m.actuator(f"{n}_pos")
        side = "left" if n.startswith("left") else "right"
        g = joint_geometry(n, f"{side} palm centre", site_fn(f"{side}_palm"))
        hand.append({"index": i, "name": n, "min": _r(motor.hand_lo[i]), "max": _r(motor.hand_hi[i]),
                     "torque_limit": _r(a.forcerange[1], 2), "servo_kp": _r(a.gainprm[0], 2),
                     "servo_kd": _r(-a.biasprm[2], 3), "axis": g["axis"], "axis_text": g["axis_text"]})
    spec["hand_joints"] = hand
    spec["motor_law"] = ("tau = kp * (q - q_measured) + kd * (dq - dq_measured) + tau_ff, evaluated by the motor "
                         f"drivers at the physics rate ({spec['physics_rate_hz']} Hz) with your latest targets and "
                         "gains, then clipped to the motor's torque limit")
    spec["hand_servo_law"] = "tau = servo_kp * (hand_q - q) - servo_kd * dq, clipped to the torque limit"

    def anchor(n):
        return d.xanchor[m.joint(n).id]

    lb, lpts, rad, lc = sole["left"]
    rb, _, _, rc = sole["right"]
    lsole, rsole = fr.point(_local_point(d, lb, lc)), fr.point(_local_point(d, rb, rc))
    hip_l, knee_l, ank_l = (fr.point(anchor(f"left_{k}_joint")) for k in ("hip_pitch", "knee", "ankle_pitch"))
    hip_r = fr.point(anchor("right_hip_pitch_joint"))
    sh_l, el_l, wr_l = (fr.point(anchor(f"left_{k}_joint")) for k in ("shoulder_pitch", "elbow", "wrist_roll"))
    sh_r = fr.point(anchor("right_shoulder_pitch_joint"))
    palm_l = fr.point(d.site_xpos[m.site("left_palm").id])
    pelvis = m.body("pelvis").id
    com = fr.point(d.subtree_com[pelvis])
    tree = set(_subtree(m, pelvis))
    lo, hi = _bbox(m, d, fr, tree)
    dz = mujoco.MjData(m)  # standing straight (all body joints at 0): overall height from the soles
    mujoco.mj_copyData(dz, m, d)
    dz.qpos[robot.q_adr] = 0.0
    mujoco.mj_kinematics(m, dz)
    height = _bbox(m, dz, fr, tree)[1][2] - min(_local_point(dz, b, c)[2] for b, _, _, c in sole.values())
    xs, ys = lpts[:, 0], lpts[:, 1]
    imu = {}
    for k, site in (("imu_pelvis", "imu_in_pelvis"), ("imu_torso", "imu_in_torso")):
        s = m.site(site).id
        imu[k] = {"body": m.body(m.site_bodyid[s]).name, "offset_in_body": _r(m.site_pos[s], 4),
                  "orientation_in_body": "aligned with the body frame (x forward, y left, z up)",
                  "position_at_start": _r(fr.point(d.site_xpos[s]), 4)}
    spec["dimensions"] = {
        "total_mass_kg": spec["mass_kg"],
        "height_standing_straight_m (all joints 0)": _r(height, 3),
        "bounding_box_at_start_m": {"min": _r(lo, 3), "max": _r(hi, 3)},
        "pelvis_origin_at_start_m": _r(fr.point(robot.base_pos), 4),
        "com_at_start_m": _r(com, 4),
        "hip_pitch_joint_left_right_m": [_r(hip_l, 4), _r(hip_r, 4)],
        "hip_spacing_m": _r(abs(hip_l[1] - hip_r[1]), 4),
        "thigh_length_m (hip pitch -> knee)": _r(np.linalg.norm(knee_l - hip_l), 4),
        "shank_length_m (knee -> ankle pitch)": _r(np.linalg.norm(ank_l - knee_l), 4),
        "ankle_pitch_axis_height_above_sole_m": _r(ank_l[2] - lsole[2], 4),
        "sole_contact_points_in_ankle_roll_link_m": [_r(p, 4) for p in lpts],
        "sole_contact_sphere_radius_m": _r(rad, 4),
        "sole_contact_patch_m (length x width)": [_r(xs.max() - xs.min() + 2 * rad, 4),
                                                   _r(ys.max() - ys.min() + 2 * rad, 4)],
        "sole_centre_left_right_at_start_m": [_r(lsole, 4), _r(rsole, 4)],
        "stance_width_at_start_m (sole centres)": _r(abs(lsole[1] - rsole[1]), 4),
        "shoulder_pitch_joint_left_right_m": [_r(sh_l, 4), _r(sh_r, 4)],
        "upper_arm_length_m (shoulder pitch -> elbow)": _r(np.linalg.norm(el_l - sh_l), 4),
        "forearm_length_m (elbow -> wrist roll)": _r(np.linalg.norm(wr_l - el_l), 4),
        "wrist_roll_to_palm_centre_m": _r(np.linalg.norm(palm_l - wr_l), 4),
        "palm_centre_left_at_start_m": _r(palm_l, 4),
        "imu": imu,
        "fall_rule": "the robot counts as fallen when the pelvis origin drops below 0.35 m or the pelvis tilts more "
                     "than 55 deg; a fall does not end the episode (get back up and carry on)",
    }
    spec["sensors"] = [
        _sensor("time", [], "s", "", "simulated time since the episode started"),
        _sensor("tick", [], "-", "", "control ticks since the episode started"),
        *[s for k in ("imu_pelvis", "imu_torso") for s in (
            _sensor(f"{k}.quaternion", [4], "-", "IMU -> world",
                    f"orientation [w, x, y, z] of the {k[4:]} IMU frame in the world frame (IMU on "
                    f"{imu[k]['body']} at {imu[k]['offset_in_body']} m, axes aligned with that link)"),
            _sensor(f"{k}.angular_velocity", [3], "rad/s", "IMU", "gyroscope"),
            _sensor(f"{k}.linear_acceleration", [3], "m/s^2", "IMU",
                    "accelerometer (specific force): about [0, 0, +9.81] when still and upright"))],
        _sensor("joint_positions", [29], "rad", "joint", "body joint encoders, joint order of the joint table"),
        _sensor("joint_velocities", [29], "rad/s", "joint", "body joint velocities"),
        _sensor("joint_torques", [29], "N m", "joint",
                "torque each body motor applies (after clipping to its torque limit)"),
        _sensor("hand_positions", [14], "rad", "joint", "Dex3 hand joints, hand joint order (left 7, right 7)"),
        _sensor("hand_velocities", [14], "rad/s", "joint", "Dex3 hand joint velocities"),
        _sensor("foot_contact_forces", [2], "N", "",
                "[left, right] sum of the normal forces of all contacts on each foot (~170 N each when standing "
                "still on both feet)"),
        _sensor("base_position", [3], "m", "world",
                "pelvis origin position - exact (as from a perfect state estimator / motion capture)"),
        _sensor("base_velocity", [3], "m/s", "world", "pelvis origin linear velocity - exact"),
    ]
    spec["start"] = ("Standing in the crouched start posture (q_start) with every body motor holding it with the "
                     "start gains (kp_start / kd_start, dq = 0, tau = 0) and the hands open (hand_q = 0). This "
                     "command stays in force until you send your first command; with it the robot stands still "
                     "indefinitely. Fields you never send keep these values.")
    if motor.mode == "velocity":
        spec["start"] = ("Standing, balanced by the walking policy: base_velocity [0, 0, 0], pelvis_height 0.74, "
                         "arm_q all 0 (forearms pointing forward), hands open (hand_q = 0). This stays in force until "
                         "you send your first command.")
        spec["motor_law"] = ("legs and waist: the walking policy's joint targets tracked by PD with kp_default / "
                             f"kd_default at {spec['physics_rate_hz']} Hz; arms: PD to the slew-limited arm_q "
                             "targets with kp_default / kd_default plus gravity compensation; all clipped to the "
                             "torque limits")


# ---------------------------------------------------------------------------------------------------- Panda
def _panda(spec, env, motor, m, d, fr, bodies, joint_geometry) -> None:
    from roboticsbench.robots import panda as P

    tcp = m.site("tcp").id

    def tcp_fn(dd):
        return dd.site_xpos[tcp].copy()

    joints = []
    for i, n in enumerate(motor.joint_names):
        row = {"index": i, "name": n, "min": _r(motor.q_lo[i]), "max": _r(motor.q_hi[i]),
               "max_speed": _r(P.MAX_JOINT_VEL[i], 3), "start": _r(P.HOME_Q[i])}
        a = m.actuator(f"actuator{i + 1}")
        row.update({"torque_limit": _r(a.forcerange[1], 1), "servo_kp": _r(a.gainprm[0], 1),
                    "servo_kd": _r(-a.biasprm[2], 1)})
        row.update(joint_geometry(n, "TCP", tcp_fn))
        joints.append(row)
    spec["joints"] = joints
    spec["servo_law"] = ("joint position servos: tau = servo_kp * (q_target - q) - servo_kd * dq (clipped to the "
                         "torque limit) plus exact gravity compensation of the links, evaluated at the physics rate "
                         f"({spec['physics_rate_hz']} Hz); q_target moves towards your commanded joint_positions at "
                         "each joint's max_speed")
    spec["gripper"] = {"max_width_m": P.MAX_OPENING, "force_range_n": [5.0, P.GRIP_FORCE_MAX],
                       "default_force_n": P.GRIP_FORCE_DEFAULT, "closing_speed_m_per_s": _r(2 * P.GRIP_SPEED, 3),
                       "start_width_m": P.MAX_OPENING}
    hand = m.body("hand").id
    lo, hi = _bbox(m, d, fr, set(bodies))
    spec["dimensions"] = {
        "total_mass_kg": spec["mass_kg"],
        "base_m": _r(fr.point(d.xpos[m.body("link0").id]), 4),
        "joint_anchors_at_start_m": {n: _r(fr.point(d.xanchor[m.joint(n).id]), 4) for n in motor.joint_names},
        "shoulder_height_m (joint 2 axis above the base)": _r(d.xanchor[m.joint("joint2").id][2], 4),
        "flange_to_tcp_m": _r(np.linalg.norm(d.site_xpos[tcp] - d.xpos[hand]), 4),
        "tcp_at_start_m": _r(fr.point(d.site_xpos[tcp]), 4),
        "max_reach_m (from the joint 2 axis, datasheet)": 0.855,
        "bounding_box_at_start_m": {"min": _r(lo, 3), "max": _r(hi, 3)},
        "wrist_force_torque_sensor": "between link7 and the hand (flange), values reported in the WORLD frame, "
                                     "tared at the episode start",
        "protective_stop": "an arm link (not the fingers) touching something with > 45 N, or > 80 N external force "
                           "on the hand, halts the arm (targets reset to where it is) until you send new targets",
    }
    spec["sensors"] = [
        _sensor("time", [], "s", "", "simulated time since the episode started"),
        _sensor("tick", [], "-", "", "control ticks since the episode started"),
        _sensor("joint_positions", [7], "rad", "joint", "joint encoders, joint order of the joint table"),
        _sensor("joint_velocities", [7], "rad/s", "joint", "joint velocities"),
        _sensor("joint_torques", [7], "N m", "joint", "servo torques (gravity compensation not included)"),
        _sensor("gripper_width", [], "m", "", "distance between the two fingers"),
        _sensor("gripper_force", [], "N", "", "gripper motor effort (the squeeze force once in contact)"),
        _sensor("wrist_force_torque", [6], "N, N m", "world",
                "[fx, fy, fz, tx, ty, tz] external wrench on the hand, tared at the start: holding a mass m reads "
                "fz ~ -9.81 m"),
        _sensor("protective_stop", [], "bool", "", "true after a collision reflex until you send new joint targets"),
    ]
    spec["start"] = ("At the start posture (joint 'start' column) with the servos holding it, gripper open "
                     "(0.08 m). This stays in force until you send your first command.")


# ---------------------------------------------------------------------------------------------------- shared
def _sensor(key: str, shape: list[int], units: str, frame: str, description: str) -> dict[str, Any]:
    return {"key": key, "shape": shape, "units": units, "frame": frame, "description": description}


def _bbox(m, d, fr, bodies: set[int]) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in range(m.ngeom):
        if m.geom_bodyid[g] not in bodies or m.geom_rgba[g][3] == 0:
            continue
        c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
        R = d.geom_xmat[g].reshape(3, 3)
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    p = fr.point(d.geom_xpos[g] + R @ (c + h * np.array([sx, sy, sz])))
                    lo, hi = np.minimum(lo, p), np.maximum(hi, p)
    return lo, hi


def _cameras(env, m, d, fr) -> list[dict[str, Any]]:
    robot_bodies = set(_subtree(m, m.body(env.robot.root_body).id))
    out = []
    for name in env.obs_cameras:
        if env._free_camera(name) is not None:
            out.append({"name": name, "kind": "external, follows the robot from behind (third-person view)",
                        "fovy_deg": _r(m.vis.global_.fovy, 1)})
            continue
        c = m.camera(name).id
        body = m.cam_bodyid[c]
        ortho = int(m.cam_projection[c]) == int(mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC)
        row: dict[str, Any] = {"name": name}
        mounted = (int(m.cam_mode[c]) == int(mujoco.mjtCamLight.mjCAMLIGHT_FIXED)
                   and float(np.linalg.norm(m.cam_pos[c])) < 0.6)  # tracking views hang metres off the pelvis
        if body in robot_bodies and not mounted:
            row["kind"] = "external, follows the robot (a tracking view, not a camera on the robot)"
        elif body in robot_bodies:
            fwd = fr.vec(-d.cam_xmat[c].reshape(3, 3)[:, 2])
            row.update({"kind": "onboard", "mount_body": m.body(body).name, "mount_offset_m": _r(m.cam_pos[c], 4),
                         "position_at_start_m": _r(fr.point(d.cam_xpos[c]), 4),
                         "looks_along_at_start": _r(fwd, 3),
                         "pitch_down_deg_at_start": _r(math.degrees(math.asin(max(-1.0, min(1.0, -fwd[2])))), 1)})
        else:
            row["kind"] = "fixed in the scene (pose: see each image's calibration file)"
        if ortho:
            row["projection"], row["view_height_m"] = "orthographic", _r(m.cam_fovy[c], 3)
        else:
            row["projection"], row["fovy_deg"] = "perspective", _r(m.cam_fovy[c], 1)
            f = (env.height / 2.0) / math.tan(math.radians(float(m.cam_fovy[c])) / 2.0)
            row[f"intrinsics_{env.width}x{env.height}"] = {"fx": _r(f, 2), "fy": _r(f, 2), "cx": env.width / 2.0,
                                                           "cy": env.height / 2.0}
        out.append(row)
    return out


# ---------------------------------------------------------------------------------------------------- markdown
def _md_table(rows: list[dict[str, Any]], cols: list[tuple[str, str]]) -> str:
    def cell(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.4g}"
        if isinstance(v, list):
            return "[" + ", ".join(cell(x) for x in v) + "]"
        return str(v)

    head = "| " + " | ".join(label for _, label in cols) + " |"
    sep = "|" + "---|" * len(cols)
    return "\n".join([head, sep, *("| " + " | ".join(cell(r.get(k, "")) for k, _ in cols) + " |" for r in rows)])


def _friction(joints: list[dict[str, Any]]) -> tuple[str, list[tuple[str, str]]]:
    """Joint damping / dry friction: one sentence when every joint has the same values, else table columns."""
    if len({(j["damping"], j["frictionloss"]) for j in joints}) == 1:
        j = joints[0]
        return f" Every joint has damping {j['damping']} N m s/rad and dry friction {j['frictionloss']} N m.", []
    return "", [("damping", "damping (N m s/rad)"), ("frictionloss", "dry friction (N m)")]


def _md_value(v: Any) -> str:
    if isinstance(v, dict):
        return "; ".join(f"{k}: {_md_value(x)}" for k, x in v.items())
    if isinstance(v, list):
        return "[" + ", ".join(_md_value(x) for x in v) + "]"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def spec_markdown(spec: dict[str, Any]) -> dict[str, str]:
    """ROBOT.md sections from the datasheet: sensors, joint axes, motor parameters, body, cameras."""
    g1 = spec["robot"] == "g1"
    out: dict[str, str] = {}
    out["sensors"] = "\n".join([
        f"Every step reply carries `sensors`; {spec['sensor_note']}.", "",
        _md_table(spec["sensors"], [("key", "key"), ("shape", "shape"), ("units", "units"), ("frame", "frame"),
                                    ("description", "meaning")])])
    axis_cols = [("index", "#"), ("name", "joint"), ("axis_text", "axis (robot frame)"),
                 ("plus_0p1rad_moves", "+0.1 rad moves"), ("plus_0p1rad_text", "by")]
    out["axes"] = "\n".join([
        "Positive rotation is counter-clockwise about the joint axis (right-hand rule). Axes and motions are in the "
        "robot frame (" + spec["robot_frame"] + "), at the start posture.", "",
        _md_table(spec["joints"], axis_cols)])
    if g1:
        mcols = [("index", "#"), ("name", "joint"), ("min", "min (rad)"), ("max", "max (rad)"),
                 ("torque_limit", "torque limit (N m)"), ("kp_default", "kp_default"), ("kd_default", "kd_default")]
        if spec["mode"] == "motor":
            mcols += [("kp_start", "kp_start"), ("kd_start", "kd_start"), ("kp_max", "kp_max"), ("kd_max", "kd_max"),
                      ("q_start", "q_start")]
        mcols.append(("armature", "rotor inertia (kg m^2)"))
        fric, extra = _friction(spec["joints"])
        out["motors"] = "\n".join([
            f"Motor law: {spec['motor_law']}.{fric}", "", _md_table(spec["joints"], mcols + extra), "",
            f"Hand joints (Dex3, position servos: {spec['hand_servo_law']}):", "",
            _md_table(spec["hand_joints"], [("index", "#"), ("name", "joint"), ("min", "min (rad)"),
                                            ("max", "max (rad)"), ("torque_limit", "torque limit (N m)"),
                                            ("servo_kp", "servo_kp"), ("servo_kd", "servo_kd"),
                                            ("axis_text", "axis (robot frame)")])])
    else:
        mcols = [("index", "#"), ("name", "joint"), ("min", "min (rad)"), ("max", "max (rad)"),
                 ("max_speed", "max speed (rad/s)"), ("torque_limit", "torque limit (N m)"), ("servo_kp", "servo_kp"),
                 ("servo_kd", "servo_kd"), ("start", "start (rad)"), ("armature", "rotor inertia (kg m^2)")]
        gr = spec["gripper"]
        fric, extra = _friction(spec["joints"])
        out["motors"] = "\n".join([
            f"Servo law: {spec['servo_law']}.{fric}", "", _md_table(spec["joints"], mcols + extra), "",
            f"Gripper: opening 0 - {gr['max_width_m']} m, squeeze force {gr['force_range_n'][0]:.0f} - "
            f"{gr['force_range_n'][1]:.0f} N (default {gr['default_force_n']:.0f} N), fingers close at "
            f"~{gr['closing_speed_m_per_s']} m/s, open at the start."])
    dims = spec["dimensions"]
    out["body"] = "\n".join([f"Total mass {spec['mass_kg']} kg. Robot frame: {spec['robot_frame']}; all positions "
                             "and axes in this manual are given in it, at the start posture.", ""]
                            + [f"- {k}: {_md_value(v)}" for k, v in dims.items()] + [
        "", "Links (origin positions at the start, robot frame; " + ("hand / finger links" if g1 else "finger links")
        + " omitted):", "",
        _md_table(spec["links"], [("name", "link"), ("parent", "parent"), ("mass_kg", "mass (kg)"),
                                  ("origin", "origin (m)")])])
    out["cameras"] = "\n".join(f"- `{c['name']}`: " + "; ".join(f"{k}: {_md_value(v)}" for k, v in c.items()
                                                               if k != "name") for c in spec["cameras"])
    out["start"] = spec["start"]
    return out
