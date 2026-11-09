"""Tool-calling LLM agent: the model sees camera images + state after every action and controls the
robot exclusively through the skill API (the same tools the scripted oracles use)."""

from __future__ import annotations

import json
import time
from typing import Any

from roboticsbench.agents.base import Agent, AgentTrace
from roboticsbench.agents.llm.prompts import observation_text, system_prompt
from roboticsbench.agents.llm.providers import ToolOutput, encode_jpeg, make_provider

NUDGE = ("You did not call any tool. Keep controlling the robot with tool calls, or call done() if the task is "
         "complete.")


class LLMAgent(Agent):
    name = "llm"

    def __init__(self, model: str, *, max_actions: int | None = None, effort: str | None = "high",
                 cameras: tuple[str, ...] | None = None, image_size: tuple[int, int] = (640, 480),
                 max_output_tokens: int | None = None, temperature: float | None = None, max_nudges: int = 2,
                 max_wall_s: float = 3600.0, show_thinking: bool = True, **provider_kw: Any):
        self.model = model
        self.max_actions = max_actions
        self.effort = effort
        self.cameras = cameras
        self.image_size = image_size
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.max_nudges = max_nudges
        self.max_wall_s = max_wall_s
        self.show_thinking = show_thinking
        self.provider_kw = provider_kw
        self.provider = None

    def describe(self) -> dict[str, Any]:
        return {"agent": self.name, "model": self.model, "effort": self.effort, "max_actions": self.max_actions,
                "image_size": list(self.image_size), "cameras": list(self.cameras) if self.cameras else None}

    def _images(self, env, cams) -> list[tuple[str, bytes]]:
        w, h = self.image_size
        return [(c, encode_jpeg(env.render(c, w, h))) for c in cams]

    @staticmethod
    def _documents(env) -> list[tuple[str, bytes]]:
        """Task documents that are images (e.g. an assembly manual), re-encoded as JPEG for the first message."""
        import io

        import numpy as np
        from PIL import Image

        out = []
        for name, data in sorted(env.task.documents(env).items()):
            if name.lower().endswith((".png", ".jpg", ".jpeg")):
                img = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
                out.append((f"doc:{name}", encode_jpeg(img, quality=88)))
        return out

    def run(self, env, sk) -> AgentTrace:
        prov = make_provider(self.model, effort=self.effort, max_output_tokens=self.max_output_tokens,
                             temperature=self.temperature, show_thinking=self.show_thinking, **self.provider_kw)
        self.provider = prov
        tr = AgentTrace(agent=self.name, model=self.model)
        max_actions = int(self.max_actions or env.task.max_actions)
        cams = tuple(self.cameras or env.obs_cameras)
        tools = sk.tools()
        system = system_prompt(env, sk, max_actions, cams)
        first = ("The episode has started. Initial observation:\n" + observation_text(env, 0, max_actions))
        docs = self._documents(env)
        if docs:
            first = (f"The task comes with {len(docs)} document image(s), attached below after the camera views "
                     f"({', '.join(n[4:] for n, _ in docs)}).\n\n" + first)
        prov.start(system, tools, first, self._images(env, cams) + docs)
        tr.notes.append(system)
        used, nudges, wall0 = 0, 0, time.time()
        while env.running:
            if used >= max_actions:
                tr.stop_reason = "tool-call budget exhausted"
                break
            if time.time() - wall0 > self.max_wall_s:
                tr.stop_reason = "wall-clock limit"
                break
            try:
                turn = prov.generate()
            except Exception as e:  # noqa: BLE001 - provider/API failure ends the episode (infra error)
                tr.stop_reason = f"model API error: {type(e).__name__}: {str(e)[:500]}"
                tr.notes.append("infra_error")
                break
            tr.add_usage(**turn.usage)
            rec = {"turn_index": len(tr.turns), "t_sim": round(env.t, 3), "stop_reason": turn.stop_reason,
                   "text": turn.text, "reasoning": turn.reasoning[:4000], "usage": turn.usage,
                   "duration_s": round(turn.duration_s, 3), "served_model": turn.served_model,
                   "tool_calls": [{"id": c.id, "name": c.name, "args": c.args} for c in turn.tool_calls]}
            tr.turns.append(rec)
            if turn.stop_details:
                rec["stop_details"] = turn.stop_details
            if turn.stop_reason == "refusal":
                cat = (turn.stop_details or {}).get("category")
                tr.stop_reason = "model refused" + (f" ({cat})" if cat else "")
                break
            if turn.stop_reason == "max_tokens" and not turn.tool_calls:
                tr.notes.append("hit max_tokens without a tool call")
            if not turn.tool_calls:
                nudges += 1
                if nudges > self.max_nudges:
                    tr.stop_reason = "model stopped calling tools"
                    break
                prov.add_user_text(NUDGE)
                continue
            outputs: list[ToolOutput] = []
            skip_reason = None
            for c in turn.tool_calls:
                if skip_reason is None and not env.running:
                    skip_reason = f"episode ended ({env.status})"
                if skip_reason is None and used >= max_actions:
                    skip_reason = "tool-call budget exhausted"
                if skip_reason is not None:
                    outputs.append(ToolOutput(c.id, f"skipped: {skip_reason}", is_error=True))
                    continue
                used += 1
                if c.parse_error:
                    res_d = {"ok": False, "message": c.parse_error}
                else:
                    res = sk.call(c.name, c.args)
                    res_d = res.to_dict()
                    if c.name == "list_objects" and res.ok:
                        res_d["message"] = json.dumps(res.data.get("objects", []), separators=(",", ":"))
                if not res_d["ok"]:
                    skip_reason = "a previous tool call in this response failed"
                outputs.append(ToolOutput(c.id, observation_text(env, used, max_actions, res_d),
                                          is_error=not res_d["ok"]))
            rec["executed"] = [o.text.split("\n", 1)[0][:300] for o in outputs]
            if not env.running:
                tr.stop_reason = f"episode ended ({env.status})"
                break
            outputs[-1].images = self._images(env, cams)
            prov.add_results(outputs)
        if not tr.stop_reason:
            tr.stop_reason = f"episode ended ({env.status})"
        tr.usage["tool_calls"] = used
        return tr
