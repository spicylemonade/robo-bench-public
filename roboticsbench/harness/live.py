"""Live view of a running episode: a small local web page that streams the simulation as it happens, the task status
and what the agent is doing (its messages and commands, read from its event stream).

The referee process creates a :class:`LiveView` next to its own server. Frames are rendered in the simulation thread
(a tick callback, every 1/fps of simulated time) and queued; the page plays them back at real-time pace, so a burst of
simulated motion is shown smoothly and the view catches up while the agent is thinking. The live server only reads
what the agent could see anyway (a task camera, public status, the agent's own output) and listens on localhost.

    open http://127.0.0.1:<port>/
"""

from __future__ import annotations

import io
import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

MAX_FRAMES = 6000  # ~6.5 min of 15 fps video buffered
MAX_LAG_S = 30.0  # a viewer that falls further behind jumps ahead
AGENT_FILES = ("agent_stdout.jsonl",)


class LiveView:
    def __init__(self, ledger: Path, port: int, *, camera: str | None = None, fps: int = 15,
                 size: tuple[int, int] = (960, 720), title: str = "", token: str = ""):
        self.ledger, self.port, self.fps = Path(ledger), int(port), int(fps)
        self.token = token  # the page lives under /<token>/ so the agent (same machine) cannot read the score
        self.camera, self.size, self.title = camera, size, title
        self.frames: deque[tuple[int, bytes]] = deque(maxlen=MAX_FRAMES)
        self.n_frames = 0
        self.cond = threading.Condition()
        self.status: dict[str, Any] = {}
        self.docs: dict[str, bytes] = {}
        self._next_t = 0.0
        self._agent_pos = 0
        self._agent_items: deque[dict[str, Any]] = deque(maxlen=300)
        self._agent_lock = threading.Lock()
        self.srv: ThreadingHTTPServer | None = None
        self._pending = None  # Future of the frame being rendered

    # ------------------------------------------------------------------ simulation side
    def attach(self, env, referee=None) -> None:
        self.env, self.referee = env, referee
        self.camera = self.camera or env.task.video_camera or env.robot.video_camera
        try:
            self.docs = dict(env.task.documents(env))
        except Exception:  # noqa: BLE001 - documents are optional decoration here
            self.docs = {}
        env.tick_callbacks.append(self._on_tick)
        self._capture(env)
        self.srv = ThreadingHTTPServer(("127.0.0.1", self.port), _handler(self))
        self.srv.daemon_threads = True
        threading.Thread(target=self.srv.serve_forever, name="rbench-live", daemon=True).start()

    def _on_tick(self, env) -> None:
        if env.t + 1e-9 >= self._next_t or not env.running:
            self._capture(env)

    def _capture(self, env) -> None:
        """Snapshot the scene and the status now; render + encode on the env's GL thread (the physics goes on)."""
        from roboticsbench.core.env import hud_info

        self._next_t = env.t + 1.0 / self.fps
        if self._pending is not None and not self._pending.done() and env.running:
            return  # the previous frame is still being rendered: drop this one (best-effort view)
        ref = self.referee
        ev = env.last_eval
        status = {"task": env.task.name, "seed": env.seed, "sim_time": round(env.t, 2),
                  "time_limit": env.task.time_limit, "status": env.status,
                  "progress": round(1.0 if env.status == "success" else float(env.progress), 3),
                  "milestones": env.task.milestones.names(), "reason": env.fail_reason or ev.reason,
                  "task_status": _jsonable(env.task.status(env)), "wall_s": round(time.time() - ref.wall0, 1)
                  if ref is not None else None,
                  "step_calls": getattr(ref, "step_calls", None), "actions": getattr(ref, "actions", None),
                  "control": getattr(ref, "control", None)}
        hud = hud_info(env)
        w, h = self.size
        self._pending = env.capture(self.camera, w, h, then=lambda img: self._publish(img, hud, status))

    def _publish(self, img, hud, status) -> None:  # GL thread
        from PIL import Image

        from roboticsbench.core.env import draw_hud

        buf = io.BytesIO()
        Image.fromarray(draw_hud(img, hud)).save(buf, "JPEG", quality=80)
        with self.cond:
            self.frames.append((self.n_frames, buf.getvalue()))
            self.n_frames += 1
            self.status = status
            self.cond.notify_all()

    def close(self) -> None:
        if self.srv is not None:
            self.srv.shutdown()

    # ------------------------------------------------------------------ agent feed
    def agent_items(self) -> list[dict[str, Any]]:
        with self._agent_lock:
            path = next((self.ledger / f for f in AGENT_FILES if (self.ledger / f).exists()), None)
            if path is not None:
                with path.open("rb") as fh:
                    fh.seek(self._agent_pos)
                    chunk = fh.read()
                cut = chunk.rfind(b"\n") + 1  # only complete lines
                self._agent_pos += cut
                for line in chunk[:cut].splitlines():
                    try:
                        item = _summarize(json.loads(line))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                    if item:
                        self._agent_items.append(item)
            return list(self._agent_items)


def _jsonable(o: Any) -> Any:
    from roboticsbench.core.jsonutil import jsonable

    return jsonable(o)


def _clip(s: Any, n: int) -> str:
    s = str(s or "")
    return s if len(s) <= n else s[:n] + " ..."


def _summarize(ev: dict[str, Any]) -> dict[str, Any] | None:
    """One feed entry from a Codex / Claude Code stream-json event (None = not interesting)."""
    t = ev.get("type")
    now = time.strftime("%H:%M:%S")
    if t == "item.completed" or t == "item.started":  # Codex
        it = ev.get("item") or {}
        kind = it.get("type")
        if kind in ("agent_message", "assistant_message"):
            return {"at": now, "kind": "message", "text": _clip(it.get("text"), 4000)}
        if kind == "reasoning":
            return {"at": now, "kind": "thinking", "text": _clip(it.get("text") or it.get("summary"), 3000)}
        if kind == "command_execution":
            if t == "item.started":
                return {"at": now, "kind": "running", "text": _clip(it.get("command"), 1500)}
            return {"at": now, "kind": "command", "text": _clip(it.get("command"), 1500),
                    "output": _clip(it.get("aggregated_output"), 1500), "exit": it.get("exit_code")}
        if kind == "file_change":
            files = [c.get("path") for c in (it.get("changes") or []) if isinstance(c, dict)]
            return {"at": now, "kind": "files", "text": ", ".join(str(f) for f in files)}
        if kind == "error":
            return {"at": now, "kind": "error", "text": _clip(it.get("message"), 1500)}
        return None
    if t in ("turn.failed", "error"):
        return {"at": now, "kind": "error", "text": _clip(ev.get("error") or ev.get("message"), 1500)}
    if t == "assistant":  # Claude Code stream-json
        out = []
        for block in (ev.get("message") or {}).get("content") or []:
            if block.get("type") == "text":
                out.append({"at": now, "kind": "message", "text": _clip(block.get("text"), 4000)})
            elif block.get("type") == "thinking":
                out.append({"at": now, "kind": "thinking", "text": _clip(block.get("thinking"), 3000)})
            elif block.get("type") == "tool_use":
                inp = block.get("input") or {}
                out.append({"at": now, "kind": "command",
                            "text": _clip(inp.get("command") or inp.get("file_path") or json.dumps(inp), 1500)})
        return out[-1] if out else None
    return None


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RoboticsBench live</title>
<style>
:root { --bg:#111317; --panel:#1b1e24; --ink:#e8e8ea; --dim:#9aa0aa; --acc:#ffd479; --ok:#7fe08a; --bad:#ff8a8a; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 -apple-system, system-ui, sans-serif; }
header { padding:10px 16px; display:flex; gap:18px; align-items:baseline; flex-wrap:wrap; border-bottom:1px solid #2a2e36; }
header h1 { font-size:16px; margin:0; color:var(--acc); }
header .kv { color:var(--dim); } header .kv b { color:var(--ink); font-weight:600; }
main { display:grid; grid-template-columns:minmax(0,1fr) 420px; gap:12px; padding:12px 16px; }
@media (max-width: 1000px) { main { grid-template-columns:1fr; } }
.video { background:#000; border-radius:8px; overflow:hidden; }
.video img { width:100%; display:block; }
.panel { background:var(--panel); border-radius:8px; padding:10px 12px; }
.bar { height:8px; background:#2a2e36; border-radius:4px; overflow:hidden; margin:6px 0 10px; }
.bar div { height:100%; background:var(--acc); width:0; transition:width .4s; }
#feed { max-height:calc(100vh - 360px); overflow:auto; }
.item { border-top:1px solid #2a2e36; padding:7px 0; white-space:pre-wrap; word-break:break-word; }
.item .k { font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--dim); }
.item.message .k { color:var(--acc); } .item.thinking .k { color:#9ecbff; } .item.error .k { color:var(--bad); }
.item code { display:block; font:12px/1.4 ui-monospace, Menlo, monospace; color:#cfd3da; }
.item .out { font:11px/1.35 ui-monospace, Menlo, monospace; color:var(--dim); max-height:9em; overflow:hidden; }
.docs a { color:var(--acc); margin-right:8px; font-size:12px; }
pre.st { font:12px/1.4 ui-monospace, Menlo, monospace; color:#cfd3da; margin:0; white-space:pre-wrap; }
</style></head><body>
<header><h1 id="title">RoboticsBench live</h1>
<span class="kv">status <b id="status">-</b></span><span class="kv">sim <b id="sim">-</b></span>
<span class="kv">wall <b id="wall">-</b></span><span class="kv">score <b id="score">-</b></span>
<span class="kv">steps <b id="steps">-</b></span></header>
<main>
<div><div class="video"><img id="vid" src="stream" alt="live simulation"></div>
<div class="panel" style="margin-top:10px"><div class="docs" id="docs"></div></div></div>
<div>
<div class="panel"><div>progress</div><div class="bar"><div id="pbar"></div></div>
<pre class="st" id="task"></pre></div>
<div class="panel" style="margin-top:10px"><div>agent activity (newest first)</div><div id="feed"></div></div>
</div></main>
<script>
const esc = s => String(s ?? "").replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
async function poll() {
  try {
    const s = await (await fetch("status.json", {cache:"no-store"})).json();
    document.getElementById("title").textContent = s.title || s.task;
    const st = document.getElementById("status"); st.textContent = s.status;
    st.style.color = s.status === "success" ? "var(--ok)" : (s.status === "running" ? "var(--ink)" : "var(--bad)");
    document.getElementById("sim").textContent = `${s.sim_time}s / ${s.time_limit}s`;
    document.getElementById("wall").textContent = s.wall_s == null ? "-" : `${Math.floor(s.wall_s/60)}m ${Math.round(s.wall_s%60)}s`;
    document.getElementById("score").textContent = `${Math.round(100*s.progress)}%`;
    document.getElementById("steps").textContent = s.step_calls ?? s.actions ?? "-";
    document.getElementById("pbar").style.width = `${Math.round(100*s.progress)}%`;
    document.getElementById("task").textContent = (s.reason ? s.reason + "\\n" : "") + JSON.stringify(s.task_status, null, 1);
    const f = await (await fetch("agent.json", {cache:"no-store"})).json();
    document.getElementById("feed").innerHTML = f.slice().reverse().slice(0, 80).map(it =>
      `<div class="item ${it.kind}"><div class="k">${esc(it.at)} ${esc(it.kind)}${it.exit != null ? " (exit " + esc(it.exit) + ")" : ""}</div>` +
      (it.kind === "command" || it.kind === "running" ? `<code>${esc(it.text)}</code>` : esc(it.text)) +
      (it.output ? `<div class="out">${esc(it.output)}</div>` : "") + `</div>`).join("");
  } catch (e) {}
  setTimeout(poll, 1000);
}
fetch("docs.json").then(r => r.json()).then(d => {
  document.getElementById("docs").innerHTML = d.length ? "documents: " + d.map(n => `<a href="doc/${n}" target="_blank">${esc(n.split("/").pop())}</a>`).join("") : "";
});
poll();
</script></body></html>
"""


def _handler(live: LiveView):
    class Handler(BaseHTTPRequestHandler):
        server_version = "rbench-live"

        def log_message(self, fmt: str, *args: Any) -> None:
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if live.token:
                prefix = f"/{live.token}"
                if not (path == prefix or path.startswith(prefix + "/")):
                    return self._send(404, b"not found", "text/plain")
                path = path[len(prefix):] or "/"
            if path == "/":
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if path == "/status.json":
                with live.cond:
                    st = dict(live.status)
                st["title"] = live.title or st.get("task")
                return self._send(200, json.dumps(st).encode(), "application/json")
            if path == "/agent.json":
                return self._send(200, json.dumps(live.agent_items()).encode(), "application/json")
            if path == "/docs.json":
                return self._send(200, json.dumps(sorted(live.docs)).encode(), "application/json")
            if path.startswith("/doc/"):
                data = live.docs.get(path[len("/doc/"):])
                if data is None:
                    return self._send(404, b"not found", "text/plain")
                return self._send(200, data, "image/png" if path.endswith(".png") else "application/octet-stream")
            if path == "/frame.jpg":
                with live.cond:
                    data = live.frames[-1][1] if live.frames else b""
                return self._send(200, data, "image/jpeg")
            if path == "/stream":
                return self._stream()
            self._send(404, b"not found", "text/plain")

        def _stream(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            period = 1.0 / live.fps
            with live.cond:  # start ~2 s behind the newest frame
                cursor = max(live.frames[0][0], live.frames[-1][0] - 2 * live.fps) if live.frames else 0
            try:
                self.wfile.write(b"--frame\r\n")
                while True:
                    t0 = time.time()
                    with live.cond:
                        if not live.frames or live.frames[-1][0] < cursor:
                            live.cond.wait(timeout=1.0)
                        if live.frames and live.frames[-1][0] >= cursor:
                            first, last = live.frames[0][0], live.frames[-1][0]
                            if last - cursor > MAX_LAG_S * live.fps:  # too far behind: jump ahead
                                cursor = last - int(5 * live.fps)
                            cursor = max(cursor, first)
                            data = live.frames[cursor - first][1]
                            cursor += 1
                        else:
                            # idle (the agent is thinking): repeat the newest frame - browsers only paint a part
                            # once the next one starts, so without this the last frame would never show
                            data = live.frames[-1][1] if live.frames else b""
                    if data:
                        self.wfile.write(b"Content-Type: image/jpeg\r\nContent-Length: " + str(len(data)).encode() +
                                         b"\r\n\r\n" + data + b"\r\n--frame\r\n")
                    time.sleep(max(0.0, period - (time.time() - t0)))
            except (BrokenPipeError, ConnectionResetError):
                return

    return Handler
