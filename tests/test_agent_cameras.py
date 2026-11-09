"""What an agent may look through is fixed by the robot, not the task: arm = top + wrist, humanoid = head."""

import pytest

from roboticsbench.core.env import Env


@pytest.mark.parametrize("task", ["arm/balance_scale", "arm/lego_manual"])
def test_arm_agents_get_the_top_and_wrist_cameras_only(task):
    env = Env(task, seed=0, control="motor")
    try:
        assert env.obs_cameras == ("top", "wrist")
        for cam in env.obs_cameras:
            assert env.render(cam, 64, 48).shape == (48, 64, 3)
    finally:
        env.close()


def test_humanoid_agents_get_the_head_camera_only():
    env = Env("humanoid/ninja_course", seed=0, control="motor")
    try:
        assert env.obs_cameras == ("head",)
    finally:
        env.close()
