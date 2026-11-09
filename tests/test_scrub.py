"""Injected API keys must not survive in anything an episode writes (Vals security rule)."""

import json

from roboticsbench.scrub import scrub_obj, scrub_text, scrub_tree, secret_values

FAKE = "zzfakeprovidersecret0123456789abcdef"  # not key-shaped: only caught through the environment


def test_env_values_and_key_shapes_are_redacted():
    secrets = secret_values(environ={"OPENROUTER_API_KEY": FAKE, "HOME": "/Users/someone-with-a-long-home"})
    text = (f"export OPENROUTER_API_KEY={FAKE}; key=sk-ant-api03-{'a' * 40} aws=AKIAABCDEFGHIJKLMNOP "
            "home=/Users/someone-with-a-long-home")
    out, n = scrub_text(text, secrets)
    assert FAKE not in out and "sk-ant-api03" not in out and "AKIAABCDEFGHIJKLMNOP" not in out
    assert "[REDACTED:OPENROUTER_API_KEY]" in out and "[REDACTED:anthropic]" in out
    assert "/Users/someone-with-a-long-home" in out  # non-secret variables stay
    assert n == 3


def test_scrub_obj_covers_keys_nested_and_non_string_fields():
    class Opaque:
        def __str__(self):
            return f"token={FAKE}"

    secrets = {FAKE: "ANTHROPIC_API_KEY"}
    obj = {FAKE: [1, 2.5, None, {"cmd": f"curl -H 'x-api-key: {FAKE}'"}], "blob": FAKE.encode(), "o": Opaque()}
    clean, n = scrub_obj(obj, secrets)
    assert FAKE not in json.dumps(clean, default=str)
    assert n == 4 and clean["[REDACTED:ANTHROPIC_API_KEY]"][:3] == [1, 2.5, None]


def test_scrub_tree_rewrites_logs_but_not_signed_files(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE)
    (tmp_path / "workspace").mkdir()
    (tmp_path / "agent_stdout.jsonl").write_text(json.dumps({"text": f"env says {FAKE}"}) + "\n")
    (tmp_path / "workspace" / "notes.md").write_text(f"saved my key {FAKE}\n")
    (tmp_path / "referee_trace.json").write_text(json.dumps({"x": FAKE}))
    (tmp_path / "video.mp4").write_bytes(b"\x00\x01" + FAKE.encode())
    changed = scrub_tree(tmp_path)
    assert set(changed) == {"agent_stdout.jsonl", "workspace/notes.md"}
    assert FAKE not in (tmp_path / "agent_stdout.jsonl").read_text()
    assert FAKE in (tmp_path / "referee_trace.json").read_text()  # its writer scrubs before signing
