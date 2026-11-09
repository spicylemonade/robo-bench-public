"""GR00T whole-body-control lower-body policy for the Unitree G1 (NVlabs/GR00T-WholeBodyControl).

Two ONNX MLPs (Balance / Walk) map a 6-frame observation history to 15 joint-position targets
(12 leg + 3 waist joints) at 50 Hz. The upper body (arms, hands) is free for manipulation and is
treated by the policy as an external disturbance, which is exactly what we want for
loco-manipulation. Observation layout (86 per frame, oldest first):

    [0:3]   base velocity command (vx, vy, wz) * (2, 2, 0.5)
    [3]     base height command (m)
    [4:7]   torso roll / pitch / yaw command (rad)
    [7:10]  pelvis angular velocity (body frame) * 0.5
    [10:13] gravity direction in pelvis frame
    [13:42] 29 body joint positions - defaults
    [42:71] 29 body joint velocities * 0.05
    [71:86] previous action
"""

from __future__ import annotations

import collections

import numpy as np

from roboticsbench.assets import G1_POLICY_BALANCE, G1_POLICY_WALK, ensure_assets
from roboticsbench.core.geometry import gravity_in_body

LEG_WAIST_DEFAULT = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                              -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                              0.0, 0.0, 0.0], dtype=np.float32)
LEG_WAIST_KP = np.array([150, 150, 150, 200, 40, 40, 150, 150, 150, 200, 40, 40, 250, 250, 250], dtype=np.float64)
LEG_WAIST_KD = np.array([2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 5, 5, 5], dtype=np.float64)
CMD_SCALE = np.array([2.0, 2.0, 0.5], dtype=np.float32)
ACTION_SCALE = 0.25
HISTORY = 6
FRAME = 86
DEFAULT_HEIGHT = 0.74

# Command envelope we expose (the policy is trained roughly on these ranges).
VX_RANGE = (-0.6, 1.0)
VY_RANGE = (-0.5, 0.5)
WZ_RANGE = (-1.0, 1.0)
HEIGHT_RANGE = (0.50, 0.80)


class GearWbcPolicy:
    def __init__(self, threads: int = 1):
        import onnxruntime as ort

        ensure_assets()
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        prov = ["CPUExecutionProvider"]
        self._balance = ort.InferenceSession(str(G1_POLICY_BALANCE), so, providers=prov)
        self._walk = ort.InferenceSession(str(G1_POLICY_WALK), so, providers=prov)
        self._in = self._walk.get_inputs()[0].name
        self.reset()

    def reset(self) -> None:
        self.history = collections.deque([np.zeros(FRAME, np.float32)] * HISTORY, maxlen=HISTORY)
        self.action = np.zeros(15, np.float32)
        self.walking = False

    def step(self, q29: np.ndarray, dq29: np.ndarray, base_quat: np.ndarray, base_angvel_body: np.ndarray,
             cmd_vel: np.ndarray, height: float, torso_rpy: np.ndarray) -> np.ndarray:
        """Returns 15 leg+waist joint-position targets."""
        o = np.empty(FRAME, np.float32)
        o[0:3] = np.asarray(cmd_vel, np.float32) * CMD_SCALE
        o[3] = height
        o[4:7] = torso_rpy
        o[7:10] = base_angvel_body * 0.5
        o[10:13] = gravity_in_body(base_quat)
        qd = np.asarray(q29, np.float32).copy()
        qd[:15] -= LEG_WAIST_DEFAULT
        o[13:42] = qd
        o[42:71] = np.asarray(dq29, np.float32) * 0.05
        o[71:86] = self.action
        self.history.append(o)
        obs = np.concatenate(self.history)[None, :]
        self.walking = float(np.linalg.norm(cmd_vel)) >= 0.05
        sess = self._walk if self.walking else self._balance
        self.action = sess.run(None, {self._in: obs})[0][0].astype(np.float32)
        return self.action * ACTION_SCALE + LEG_WAIST_DEFAULT
