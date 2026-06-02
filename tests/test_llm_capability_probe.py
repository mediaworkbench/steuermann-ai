from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from backend.fastapi_app import _run_llm_capability_startup_probe
from backend.llm_capability_probe import (
    LLMCapabilityProbeResult,
    LLMCapabilityProbeRunner,
    _detect_vision_from_model_entry,
    _fetch_model_metadata,
)


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
        profile=SimpleNamespace(language="en"),
    )


class TestDetectVisionFromModelEntry:
    def test_lmstudio_vlm_type(self):
        assert _detect_vision_from_model_entry({"type": "vlm"}) is True

    def test_vision_type(self):
        assert _detect_vision_from_model_entry({"type": "vision"}) is True

    def test_multimodal_type(self):
        assert _detect_vision_from_model_entry({"type": "multimodal"}) is True

    def test_capabilities_list_with_vision(self):
        assert _detect_vision_from_model_entry({"capabilities": ["vision", "tools"]}) is True

    def test_capabilities_list_without_vision(self):
        assert _detect_vision_from_model_entry({"capabilities": ["tools"]}) is False

    def test_capabilities_dict_vision_true(self):
        assert _detect_vision_from_model_entry({"capabilities": {"vision": True, "tools": True}}) is True

    def test_capabilities_dict_no_vision(self):
        assert _detect_vision_from_model_entry({"capabilities": {"tools": True}}) is False

    def test_direct_vision_bool(self):
        assert _detect_vision_from_model_entry({"vision": True}) is True

    def test_modality_image_string(self):
        assert _detect_vision_from_model_entry({"modality": "text+image->text"}) is True

    def test_text_only_type_returns_none(self):
        assert _detect_vision_from_model_entry({"type": "llm"}) is None

    def test_empty_dict_returns_none(self):
        assert _detect_vision_from_model_entry({}) is None


class TestFetchModelMetadata:
    def _make_client(self, native_response=None, compat_response=None, raise_on_native=False):
        """Return a mock httpx.Client that serves different responses per URL."""
        native_mock = MagicMock()
        if raise_on_native:
            native_mock.raise_for_status.side_effect = Exception("not found")
        else:
            native_mock.json.return_value = native_response or {"data": []}

        compat_mock = MagicMock()
        compat_mock.json.return_value = compat_response or {"data": []}

        client = MagicMock()
        client.__enter__ = MagicMock(return_value=client)
        client.__exit__ = MagicMock(return_value=False)

        def _get(url):
            return native_mock if "/api/v0/" in url else compat_mock

        client.get.side_effect = _get
        return client

    def test_native_endpoint_vlm_type(self):
        client = self._make_client(native_response={
            "data": [{"id": "google/gemma-4-e4b", "type": "vlm", "loaded_context_length": 16384}]
        })
        with patch("backend.llm_capability_probe.httpx.Client", return_value=client):
            result = _fetch_model_metadata("http://localhost:1234/v1", "openai/google/gemma-4-e4b")

        assert result["supports_vision"] is True
        assert result["context_window_tokens"] == 16384

    def test_fallback_to_compat_when_native_fails(self):
        client = self._make_client(
            raise_on_native=True,
            compat_response={"data": [{"id": "google/gemma-4-e4b", "context_length": 8192, "type": "vlm"}]},
        )
        with patch("backend.llm_capability_probe.httpx.Client", return_value=client):
            result = _fetch_model_metadata("http://localhost:1234/v1", "openai/google/gemma-4-e4b")

        assert result["supports_vision"] is True
        assert result["context_window_tokens"] == 8192

    def test_omits_vision_when_unknown(self):
        client = self._make_client(native_response={
            "data": [{"id": "google/gemma-4-e4b", "max_context_length": 4096}]
        })
        with patch("backend.llm_capability_probe.httpx.Client", return_value=client):
            result = _fetch_model_metadata("http://localhost:1234/v1", "openai/google/gemma-4-e4b")

        assert result["context_window_tokens"] == 4096
        assert "supports_vision" not in result

    def test_returns_empty_dict_on_network_error(self):
        with patch("backend.llm_capability_probe.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__ = MagicMock(side_effect=Exception("connection refused"))
            result = _fetch_model_metadata("http://localhost:1234/v1", "openai/some/model")

        assert result == {}


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
