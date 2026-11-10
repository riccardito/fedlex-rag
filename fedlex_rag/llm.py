"""Claude calls, recorded: every request is keyed by a hash of its content and its answer stored.

`make eval` replays `data/recordings.json` and needs no key. `--live` records what is missing,
with the key from ANTHROPIC_API_KEY or, failing that, the OS keyring (service `anthropic`,
user `portfolio`).

Two notes on the request shape:
- `fallbacks: "default"` (beta `server-side-fallback-2026-07-01`): if a safety classifier
  declines a request, the API re-runs it on the model Anthropic recommends for that refusal
  category instead of returning the refusal. The serving model is recorded.
- No prompt caching: the system prompts are far below the minimum cacheable prefix, and the
  article passages -- the bulk of each request -- differ from question to question.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .loader import DATA

MODEL = "claude-opus-5-5"
EFFORT = "medium"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
RECORDINGS = DATA / "recordings.json"


class MissingRecording(LookupError):
    """A request with no recorded answer, and no client to ask."""


class Refused(RuntimeError):
    """Every model in the fallback chain declined the request."""


def request_key(request: dict) -> str:
    return hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def api_key() -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    try:
        import keyring
    except ImportError:
        return None
    return keyring.get_password("anthropic", "portfolio")


def live_client():
    """An Anthropic client, or a clear error. Needs `pip install -e '.[llm]'`."""
    import anthropic

    key = api_key()
    if not key:
        raise RuntimeError("no API key: set ANTHROPIC_API_KEY, or store one with "
                           "`python -m keyring set anthropic portfolio`")
    return anthropic.Anthropic(api_key=key)


def call_claude(client, request: dict) -> tuple[dict, str]:
    """One structured call: (the parsed JSON, the model that served it)."""
    response = client.beta.messages.create(
        model=request["model"],
        max_tokens=16000,
        betas=[FALLBACK_BETA],
        fallbacks="default",
        system=request["system"],
        output_config={"effort": request["effort"],
                       "format": {"type": "json_schema", "schema": request["schema"]}},
        messages=[{"role": "user", "content": request["user"]}],
    )
    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        raise Refused(getattr(details, "category", None) or "unknown")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("the response hit max_tokens before the JSON was complete")
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text), response.model


class Recorder:
    """Replays recorded answers; with a client, asks for the misses and records them."""

    def __init__(self, path: Path = RECORDINGS, client=None, call=call_claude):
        self.path = path
        self.client = client
        self.call = call
        self.entries: dict[str, dict] = {}
        if path.exists():
            self.entries = json.loads(path.read_text(encoding="utf-8")).get("entries", {})
        self.added = 0
        self.used: set[str] = set()

    def ask(self, request: dict) -> dict:
        key = request_key(request)
        self.used.add(key)
        if key in self.entries:
            return self.entries[key]["output"]
        if self.client is None:
            raise MissingRecording(f"no recording for {request.get('task', '?')} request {key[:12]} -- "
                                   "run with --live to record it")
        output, served_by = self.call(self.client, request)
        self.entries[key] = {"task": request.get("task"), "served_by": served_by, "output": output}
        self.added += 1
        return output

    def save(self, prune: bool = False) -> None:
        """Write the recordings; `prune` (a full live eval) keeps only the ones this run asked for."""
        if prune:
            stale = set(self.entries) - self.used
            for key in stale:
                del self.entries[key]
            self.added += len(stale)
        if not self.added:
            return
        body = {"model": MODEL, "effort": EFFORT,
                "note": "Recorded Claude answers, keyed by the SHA-256 of each request. "
                        "`make eval` replays them; `--live` records new ones.",
                "entries": dict(sorted(self.entries.items()))}
        self.path.write_text(json.dumps(body, ensure_ascii=False, indent=1) + "\n", encoding="utf-8",
                             newline="\n")
        self.added = 0
