from __future__ import annotations

from types import SimpleNamespace

from backend.fastapi_app import _run_llm_capability_startup_probe
from backend.llm_capability_probe import LLMCapabilityProbeResult, LLMCapabilityProbeRunner


class _FakeProviders:
    def __init__(self, registry):
        self._registry = registry

    def get_registry(self):
        return self._registry


class _FakeFactory:
    def __init__(self, model):
        self._model = model

    def _build(self, provider, model_name):
        return self._model


def _make_core_config(provider):
    return SimpleNamespace(
        llm=SimpleNamespace(
            roles=SimpleNamespace(chat=object(), vision=object(), auxiliary=object()),
            get_role_provider_chain_with_models=lambda role_name, _lang: [
                ("lmstudio", provider, "openai/liquid/lfm2-24b-a2b")
            ] if role_name in {"chat", "vision", "auxiliary"} else [],
        ),
        fork=SimpleNamespace(language="en"),
    )


def test_probe_runner_native_success():
    class _Model:
        def bind_tools(self, tools):
            return self

    provider = SimpleNamespace(
        tool_calling="native",
        api_base="http://host.docker.internal:1234/v1",
        models=SimpleNamespace(en="openai/liquid/lfm2-24b-a2b"),
    )
    runner = LLMCapabilityProbeRunner(core_config=_make_core_config(provider), profile_id="starter")
    runner._factory = _FakeFactory(_Model())

    results = runner.run()

    assert len(results) == 1
    assert results[0].status == "ok"
    assert results[0].supports_bind_tools is True
    assert results[0].supports_tool_schema is True
    assert results[0].capability_mismatch is False


def test_probe_runner_native_bind_failure_marks_mismatch():
    class _Model:
        def bind_tools(self, tools):
            raise RuntimeError("bind failed")

    provider = SimpleNamespace(
        tool_calling="native",
        api_base="http://host.docker.internal:1234/v1",
        models=SimpleNamespace(en="openai/liquid/lfm2-24b-a2b"),
    )
    runner = LLMCapabilityProbeRunner(core_config=_make_core_config(provider), profile_id="starter")
    runner._factory = _FakeFactory(_Model())

    results = runner.run()

    assert len(results) == 1
    assert results[0].status == "warning"
    assert results[0].supports_bind_tools is False
    assert results[0].supports_tool_schema is False
    assert results[0].capability_mismatch is True
    assert "bind_tools_failed" in str(results[0].error_message)


def test_probe_runner_non_native_mode_is_config_only_ok():
    provider = SimpleNamespace(
        tool_calling="structured",
        api_base="http://host.docker.internal:1234/v1",
        models=SimpleNamespace(en="openai/liquid/lfm2-24b-a2b"),
    )
    runner = LLMCapabilityProbeRunner(core_config=_make_core_config(provider), profile_id="starter")

    results = runner.run()

    assert len(results) == 1
    assert results[0].status == "ok"
    assert results[0].supports_bind_tools is None
    assert results[0].supports_tool_schema is None
    assert results[0].capability_mismatch is False


def test_startup_probe_persists_results(monkeypatch):
    stored = []

    class _ProbeStore:
        def upsert_probe_result(self, result):
            stored.append(result)
            return result

    class _Runner:
        def __init__(self, profile_id):
            self._profile_id = profile_id

        def run(self):
            return [
                LLMCapabilityProbeResult(
                    profile_id=self._profile_id,
                    provider_id="lmstudio",
                    model_name="openai/liquid/lfm2-24b-a2b",
                    configured_tool_calling_mode="native",
                    supports_bind_tools=True,
                    supports_tool_schema=True,
                    capability_mismatch=False,
                    status="ok",
                )
            ]

    monkeypatch.setattr("backend.llm_capability_probe.LLMCapabilityProbeRunner", _Runner)
    monkeypatch.setenv("LLM_CAPABILITY_PROBE_ENABLED", "true")
    monkeypatch.setenv("LLM_CAPABILITY_PROBE_ON_STARTUP", "true")
    monkeypatch.setenv("PROFILE_ID", "starter")

    app = SimpleNamespace(state=SimpleNamespace(llm_capability_probe_store=_ProbeStore()))
    _run_llm_capability_startup_probe(app)

    assert len(stored) == 1
    assert stored[0]["provider_id"] == "lmstudio"
