import json
import sys
import types
from types import SimpleNamespace as NS

import pytest

from fedlex_rag import llm
from fedlex_rag.llm import MissingRecording, Recorder, Refused, call_claude, request_key

REQ = {"task": "rerank", "model": "m", "effort": "medium", "system": "s", "schema": {"a": 1}, "user": "u"}


def test_request_key_ignores_dict_order_and_sees_content():
    shuffled = dict(reversed(list(REQ.items())))
    assert request_key(REQ) == request_key(shuffled)
    assert request_key(REQ) != request_key({**REQ, "user": "v"})


def test_replay_without_a_client_refuses_a_miss(tmp_path):
    with pytest.raises(MissingRecording):
        Recorder(tmp_path / "r.json").ask(REQ)


def test_record_save_and_replay(tmp_path):
    calls = []

    def fake_call(client, request):
        calls.append(request)
        return {"ranking": ["A"]}, "claude-x"

    rec = Recorder(tmp_path / "r.json", client=object(), call=fake_call)
    assert rec.ask(REQ) == {"ranking": ["A"]}
    assert rec.ask(REQ) == {"ranking": ["A"]} and len(calls) == 1
    rec.save()
    stored = json.loads((tmp_path / "r.json").read_text("utf-8"))
    assert stored["entries"][request_key(REQ)]["served_by"] == "claude-x"
    assert Recorder(tmp_path / "r.json").ask(REQ) == {"ranking": ["A"]}   # keyless replay


def test_save_without_news_writes_nothing(tmp_path):
    Recorder(tmp_path / "r.json").save()
    assert not (tmp_path / "r.json").exists()


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def client_for(response):
    messages = FakeMessages(response)
    return NS(beta=NS(messages=messages)), messages


def test_call_claude_sends_fallbacks_and_schema_and_parses():
    resp = NS(stop_reason="end_turn", model="claude-opus-5-5",
              content=[NS(type="thinking"), NS(type="text", text='{"ranking": ["B"]}')])
    client, messages = client_for(resp)
    assert call_claude(client, REQ) == ({"ranking": ["B"]}, "claude-opus-5-5")
    sent = messages.kwargs
    assert sent["fallbacks"] == "default" and sent["betas"] == [llm.FALLBACK_BETA]
    assert sent["output_config"] == {"effort": "medium", "format": {"type": "json_schema", "schema": {"a": 1}}}
    assert sent["messages"] == [{"role": "user", "content": "u"}]


def test_call_claude_raises_on_refusal_and_truncation():
    with pytest.raises(Refused, match="bio"):
        call_claude(client_for(NS(stop_reason="refusal", stop_details=NS(category="bio"), content=[]))[0], REQ)
    with pytest.raises(RuntimeError, match="max_tokens"):
        call_claude(client_for(NS(stop_reason="max_tokens", content=[]))[0], REQ)


def test_api_key_from_env_then_keyring(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    assert llm.api_key() == "env-key"
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    fake = types.ModuleType("keyring")
    fake.get_password = lambda service, user: f"{service}/{user}"
    monkeypatch.setitem(sys.modules, "keyring", fake)
    assert llm.api_key() == "anthropic/portfolio"
    monkeypatch.setitem(sys.modules, "keyring", None)      # not installed
    assert llm.api_key() is None


def test_live_client_without_a_key_says_how(monkeypatch):
    monkeypatch.setattr(llm, "api_key", lambda: None)
    monkeypatch.setitem(sys.modules, "anthropic", types.ModuleType("anthropic"))
    with pytest.raises(RuntimeError, match="keyring set anthropic portfolio"):
        llm.live_client()
