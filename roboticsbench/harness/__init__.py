"""Coding-agent harness: run Claude Code, Codex, ... on RoboticsBench tasks.

The simulator lives in a referee process that exposes only the robot skill API; the agent works in a workspace
with GOAL.md, ROBOT.md and a ``./robot`` CLI. See :mod:`roboticsbench.harness.run`.
"""
