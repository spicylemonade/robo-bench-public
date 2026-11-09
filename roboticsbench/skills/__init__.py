from __future__ import annotations

from roboticsbench.skills.base import SkillAPI, SkillError, SkillResult, skill


def make_skills(env, **kwargs) -> SkillAPI:
    """Skill API matching the env's robot."""
    kind = env.robot.name
    if kind == "g1":
        from roboticsbench.skills.humanoid import HumanoidSkills

        return HumanoidSkills(env, **kwargs)
    if kind == "panda":
        from roboticsbench.skills.arm import ArmSkills

        return ArmSkills(env, **kwargs)
    raise KeyError(kind)


__all__ = ["SkillAPI", "SkillError", "SkillResult", "skill", "make_skills"]
