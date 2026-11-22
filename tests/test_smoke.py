"""These pass on day one. They check the plumbing so you notice quickly if you break it."""
import os

from app import llm
def test_model_settings_default_to_a_local_ollama(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)                  # no .env here
    for name in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    base, key, model = llm.settings()
    assert base == "http://localhost:11434/v1" and key == "" and model == ""


def test_optional_llm_client_sends_openai_compatible_request(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1/")
    monkeypatch.setenv("LLM_API_KEY", "unit-test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    calls = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    def fake_post(url, **kwargs):
        calls["url"] = url
        calls.update(kwargs)
        return Response()

    monkeypatch.setattr(llm.httpx, "post", fake_post)
    result = llm.chat([{"role": "user", "content": "ping"}], temperature=0)
    assert result == "ok"
    assert calls["url"] == "https://example.test/v1/chat/completions"
    assert calls["headers"] == {"Authorization": "Bearer unit-test-key"}
    assert calls["json"] == {"model": "test-model", "messages": [{"role": "user", "content": "ping"}], "temperature": 0}


def test_the_command_line_runs(capsys):
    from app.cli import main
    main(["hello"])
    assert "is ready" in capsys.readouterr().out

