from __future__ import annotations

from subprocess import CompletedProcess
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.routers.settings import router
from backend.version import get_framework_version


def test_system_config_includes_active_profile_object(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    profiles_dir = config_dir / "profiles" / "medical"
    config_dir.mkdir(parents=True)
    profiles_dir.mkdir(parents=True)

    config_dir.joinpath("core.yaml").write_text(
        """
fork:
  name: starter
  language: en
database:
  url: sqlite:///base.db
memory:
  vector_store:
    host: localhost
    collection_prefix: base
  embeddings:
    model: embed
    dimension: 384
  retention:
    session_memory_days: 90
    user_memory_days: 365
checkpointing:
  enabled: false
        """,
        encoding="utf-8",
    )
    profiles_dir.joinpath("core.yaml").write_text(
        """
llm:
  providers:
    lmstudio:
      api_key: test-key
      models:
        en: openai/base-model
  roles:
    chat:
      providers:
        - provider_id: lmstudio
    embedding:
      providers:
        - provider_id: lmstudio
      config_only: true
    vision:
      providers:
        - provider_id: lmstudio
      config_only: true
    auxiliary:
      providers:
        - provider_id: lmstudio
      config_only: true
rag:
  collection_name: framework
  top_k: 5
tokens:
  default_budget: 10000
ingestion:
  collection_name: framework
        """,
        encoding="utf-8",
    )
    config_dir.joinpath("tools.yaml").write_text(
        """
tools:
  - name: datetime_tool
    path: universal_agentic_framework/tools/datetime
    enabled: true
        """,
        encoding="utf-8",
    )
    profiles_dir.joinpath("profile.yaml").write_text(
        "profile_id: medical\ndisplay_name: Medical Assistant\ndescription: Clinical profile\n",
        encoding="utf-8",
    )
    profiles_dir.joinpath("ui.yaml").write_text(
        """
branding:
  role_label: Clinical Assistant
  app_name: Med Console
theme:
  colors:
    primary: '#008080'
        """,
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PROFILE_ID", "medical")
    monkeypatch.setenv("AUTH_USERNAME", "u1")
    monkeypatch.delenv("CHAT_ACCESS_TOKEN", raising=False)

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    response = client.get("/api/system-config")

    assert response.status_code == 200
    body = response.json()
    assert body["framework_version"] == get_framework_version()
    assert body["profile"]["id"] == "medical"
    assert body["profile"]["display_name"] == "Medical Assistant"
    assert body["profile"]["role_label"] == "Clinical Assistant"
    assert body["profile"]["app_name"] == "Med Console"
    assert body["available_tools"][0]["id"] == "datetime_tool"


def test_system_config_supported_languages_fallback_order(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AUTH_USERNAME", "u1")
    monkeypatch.delenv("CHAT_ACCESS_TOKEN", raising=False)
    monkeypatch.setenv("PROFILE_ID", "starter")

    def _profile_metadata(*args, **kwargs):
        return SimpleNamespace(display_name="Starter", description="Starter profile")

    def _profile_ui(*args, **kwargs):
        return SimpleNamespace(
            branding=SimpleNamespace(role_label="Assistant", app_name="Starter App"),
            theme=SimpleNamespace(colors={}, fonts={}, radius={}, custom_css_vars={}),
        )

    monkeypatch.setattr(
        "backend.routers.settings.load_tools_config",
        lambda: SimpleNamespace(tools=[SimpleNamespace(enabled=True, name="datetime_tool")]),
    )
    monkeypatch.setattr("backend.routers.settings.load_profile_metadata", _profile_metadata)
    monkeypatch.setattr("backend.routers.settings.load_profile_ui_config", _profile_ui)

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    class _ProviderRegistry(dict):
        def get_registry(self):
            return self

    def _core_config(prompt_languages):
        provider = SimpleNamespace(models=SimpleNamespace(en="base-model", model_dump=lambda: {"en": "base-model"}))
        providers = _ProviderRegistry({"lmstudio": provider})
        roles = SimpleNamespace(
            chat=SimpleNamespace(providers=[SimpleNamespace(provider_id="lmstudio")]),
            embedding=SimpleNamespace(providers=[SimpleNamespace(provider_id="lmstudio")]),
            vision=SimpleNamespace(providers=[SimpleNamespace(provider_id="lmstudio")]),
            auxiliary=SimpleNamespace(providers=[SimpleNamespace(provider_id="lmstudio")]),
            model_dump=lambda: {
                "chat": {"providers": [{"provider_id": "lmstudio"}]},
                "embedding": {"providers": [{"provider_id": "lmstudio"}]},
                "vision": {"providers": [{"provider_id": "lmstudio"}]},
                "auxiliary": {"providers": [{"provider_id": "lmstudio"}]},
            },
        )

        llm = SimpleNamespace(
            providers=providers,
            roles=roles,
            get_role_provider=lambda _role: provider,
        )

        return SimpleNamespace(
            rag=SimpleNamespace(collection_name="framework", top_k=5),
            llm=llm,
            fork=SimpleNamespace(language="en", supported_languages=[]),
            prompts=SimpleNamespace(languages=prompt_languages),
        )

    monkeypatch.setattr(
        "backend.routers.settings.load_core_config",
        lambda: _core_config({"de": object(), "en": object()}),
    )
    response = client.get("/api/system-config")
    assert response.status_code == 200
    body = response.json()
    assert body["framework_version"] == get_framework_version()
    assert body["supported_languages"] == ["de", "en"]

    monkeypatch.setattr(
        "backend.routers.settings.load_core_config",
        lambda: _core_config({}),
    )
    response = client.get("/api/system-config")
    assert response.status_code == 200
    body = response.json()
    assert body["framework_version"] == get_framework_version()
    assert body["supported_languages"] == ["en"]


def test_reingest_all_documents_success(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AUTH_USERNAME", "u1")
    monkeypatch.delenv("CHAT_ACCESS_TOKEN", raising=False)
    monkeypatch.setenv("RAG_DATA_PATH", str(tmp_path))
    monkeypatch.setenv("INGEST_COLLECTION", "framework")
    monkeypatch.setenv("INGEST_LANGUAGE", "de")

    def _mock_run(*args, **kwargs):
        return CompletedProcess(
            args=kwargs.get("args") or args[0],
            returncode=0,
            stdout="REINDEX COMPLETE\nFiles processed:  4\nTotal chunks:     17\n",
            stderr="",
        )

    monkeypatch.setattr("backend.routers.settings.subprocess.run", _mock_run)

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    response = client.post("/api/ingestion/reingest-all")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["source"] == str(tmp_path)
    assert body["collection"] == "framework"
    assert body["language"] == "de"
    assert body["processed"] == 4
    assert body["total_chunks"] == 17


def test_reingest_all_documents_missing_source(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AUTH_USERNAME", "u1")
    monkeypatch.delenv("CHAT_ACCESS_TOKEN", raising=False)
    missing_path = tmp_path / "does-not-exist"
    monkeypatch.setenv("RAG_DATA_PATH", str(missing_path))

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    response = client.post("/api/ingestion/reingest-all")
    assert response.status_code == 400
    assert "Ingestion source path does not exist" in response.json()["detail"]


def test_llm_capabilities_includes_probe_details(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AUTH_USERNAME", "u1")
    monkeypatch.delenv("CHAT_ACCESS_TOKEN", raising=False)
    monkeypatch.setenv("PROFILE_ID", "starter")

    class _Provider:
        def __init__(self):
            self.models = SimpleNamespace(model_dump=lambda: {"en": "openai/test-model"})

        def get_tool_calling_mode(self, _model_name: str) -> str:
            return "native"

    class _Registry:
        def get(self, provider_id: str):
            return _Provider() if provider_id == "primary" else None

    core_config = SimpleNamespace(
        llm=SimpleNamespace(
            providers=SimpleNamespace(get_registry=lambda: _Registry()),
            roles=SimpleNamespace(chat=SimpleNamespace(providers=[SimpleNamespace(provider_id="primary")])),
        )
    )

    probe_rows = [
        {
            "provider_id": "primary",
            "model_name": "openai/test-model",
            "configured_tool_calling_mode": "native",
            "supports_bind_tools": False,
            "supports_tool_schema": False,
            "capability_mismatch": True,
            "status": "warning",
            "error_message": "bind_tools_failed: test-error",
            "api_base": "http://host.docker.internal:1234/v1",
            "probed_at": "2026-05-11T12:34:56+00:00",
            "metadata": {"capabilities": {"supports_json_mode": False}, "probe_kind": "native_bind_tools"},
        }
    ]

    monkeypatch.setattr("backend.routers.settings.get_active_profile_id", lambda: "starter")
    monkeypatch.setattr("backend.routers.settings.load_core_config", lambda: core_config)

    app = FastAPI()
    app.state.llm_capability_probe_store = SimpleNamespace(
        list_probe_results=lambda profile_id, limit=500: probe_rows
    )
    app.include_router(router)
    client = TestClient(app)

    response = client.get("/api/llm/capabilities")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["profile_id"] == "starter"
    assert len(payload["items"]) == 1

    item = payload["items"][0]
    assert item["provider_id"] == "primary"
    assert item["model_name"] == "openai/test-model"
    assert item["configured_tool_calling_mode"] == "native"
    assert item["api_base"] == "http://host.docker.internal:1234/v1"
    assert item["error_message"] == "bind_tools_failed: test-error"
    assert item["metadata"]["probe_kind"] == "native_bind_tools"
    assert item["capabilities"]["supports_json_mode"] is False
