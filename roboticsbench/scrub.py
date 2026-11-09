"""Secret scrubbing for everything an episode leaves behind (traces, logs, the agent's workspace copy).

Coding agents run with provider API keys in their environment, and whatever they print or write ends up in the
episode record. Vals security rule: scrub every injected key from trajectories and logs before saving or publishing
them, including keys that sit in non-string fields. Two detectors:

* exact values of secret-looking environment variables (plus any extra mapping passed in), and
* well-known key formats (Anthropic, OpenAI, OpenRouter, xAI, Google, AWS, GitHub, Hugging Face, Slack, Modal).

Scrub the in-memory object before hashing / signing a trace, and run ``scrub_tree`` over the episode directory
once the agent has exited (its stdout log and workspace copy are written outside our control).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

_SECRET_NAME = re.compile(r"(API_KEY|_KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?)$|^AWS_SESSION_TOKEN$", re.I)
# per-run capability tokens that die with the referee; redacting them only hurts debugging
_NOT_SECRET = {"RBENCH_LIVE_TOKEN", "RBENCH_OPERATOR_TOKEN", "GPG_KEY", "SSH_AUTH_SOCK"}
_MIN_LEN = 12

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("anthropic", re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")),
    ("openrouter", re.compile(r"sk-or-v1-[A-Za-z0-9]{20,}")),
    ("openai", re.compile(r"sk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_\-]{32,}")),
    ("xai", re.compile(r"xai-[A-Za-z0-9]{32,}")),
    ("google", re.compile(r"AIza[0-9A-Za-z_\-]{35}")),
    ("aws-access-key-id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{50,})\b")),
    ("huggingface", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    ("slack", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}\b")),
    ("modal", re.compile(r"\b(?:ak|as)-[A-Za-z0-9]{20,}\b")),
]

TEXT_SUFFIXES = {".json", ".jsonl", ".log", ".txt", ".md", ".py", ".html", ".csv", ".yaml", ".yml", ".toml",
                 ".sh", ".out", ".err", ".ini", ".cfg", ".env", ".xml", ""}
MAX_FILE_BYTES = 256 * 1024 * 1024
# hashed + HMAC-signed by their writers (scrubbed in memory first); rewriting them would break verification
SIGNED_FILES = frozenset({"referee_trace.json", "referee_result.json", "trace.json"})


def secret_values(extra: Mapping[str, str] | None = None, environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """value -> label for every secret-looking variable in ``environ`` (default ``os.environ``) and ``extra``."""
    out: dict[str, str] = {}
    env = os.environ if environ is None else environ
    for source in (env, extra or {}):
        for name, value in source.items():
            if name in _NOT_SECRET or not value or len(value) < _MIN_LEN:
                continue
            if _SECRET_NAME.search(name):
                out[value] = name
    return out


def scrub_text(text: str, secrets: Mapping[str, str] | None = None) -> tuple[str, int]:
    """Redact known secret values (longest first) and key-shaped strings. Returns (text, n_redactions)."""
    n = 0
    for value in sorted(secrets or {}, key=len, reverse=True):
        if value in text:
            n += text.count(value)
            text = text.replace(value, f"[REDACTED:{secrets[value]}]")
    for label, pat in PATTERNS:
        text, k = pat.subn(f"[REDACTED:{label}]", text)
        n += k
    return text, n


def scrub_obj(obj: Any, secrets: Mapping[str, str] | None = None) -> tuple[Any, int]:
    """Scrub a JSON-like object: every string anywhere (keys included) and every non-string leaf whose text form
    contains a secret (it is replaced by the redacted string)."""
    if isinstance(obj, str):
        return scrub_text(obj, secrets)
    if isinstance(obj, dict):
        out, n = {}, 0
        for k, v in obj.items():
            k2, a = scrub_obj(k, secrets) if isinstance(k, str) else (k, 0)
            v2, b = scrub_obj(v, secrets)
            out[k2] = v2
            n += a + b
        return out, n
    if isinstance(obj, (list, tuple)):
        items, n = [], 0
        for v in obj:
            v2, a = scrub_obj(v, secrets)
            items.append(v2)
            n += a
        return (items if isinstance(obj, list) else tuple(items)), n
    if isinstance(obj, bytes):
        try:
            text, n = scrub_text(obj.decode("utf-8"), secrets)
        except UnicodeDecodeError:
            return obj, 0
        return (text.encode() if n else obj), n
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj, 0
    text, n = scrub_text(str(obj), secrets)
    return (text if n else obj), n


def scrub_file(path: Path, secrets: Mapping[str, str] | None = None) -> int:
    try:
        if path.is_symlink() or path.stat().st_size > MAX_FILE_BYTES:
            return 0
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return 0
    new, n = scrub_text(text, secrets)
    if n:
        tmp = path.with_name(path.name + ".scrub-tmp")
        tmp.write_text(new, encoding="utf-8")
        os.replace(tmp, path)
    return n


def scrub_tree(root: str | os.PathLike, extra: Mapping[str, str] | None = None,
               skip: frozenset[str] = SIGNED_FILES, literal: Mapping[str, str] | None = None) -> dict[str, int]:
    """Scrub every text file under ``root`` in place (except ``skip``: signed files, which their writers scrub
    before hashing). ``extra``: more variables, redacted when their names look secret; ``literal``: {label: value}
    redacted whatever the name. Returns {relative path: redactions} for changed files."""
    secrets = secret_values(extra)
    secrets.update({v: k for k, v in (literal or {}).items() if v and len(v) >= _MIN_LEN})
    root = Path(root)
    changed: dict[str, int] = {}
    if not root.exists():
        return changed
    for p in root.rglob("*"):
        if p.name in skip:
            continue
        if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES:
            n = scrub_file(p, secrets)
            if n:
                changed[str(p.relative_to(root))] = n
    return changed


def dumps_scrubbed(obj: Any, extra: Mapping[str, str] | None = None, **kw: Any) -> str:
    """json.dumps after scrub_obj (convenience for writers that hash/sign the bytes)."""
    clean, _ = scrub_obj(obj, secret_values(extra))
    return json.dumps(clean, **kw)
