from __future__ import annotations

from subprocess import CompletedProcess
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.routers.settings import router


def test_system_config_includes_active_profile_object(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    profiles_dir = tmp_path / "profiles" / "medical"
    config_dir.mkdir(parents=True)
    profiles_dir.mkdir(parents=True)

    config_dir.joinpath("core.yaml").write_text(
        """
fork:
  name: starter
  language: en
llm:
  providers:
    primary:
      api_key: test-key
      models:
        en: openai/base-model
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
rag:
  collection_name: framework
  top_k: 5
tokens:
  default_budget: 10000
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

    # Fallback 1: derive from prompt language files when supported_languages is absent.
    monkeypatch.setattr(
        "backend.routers.settings.load_core_config",
        lambda: SimpleNamespace(
            rag=SimpleNamespace(collection_name="framework", top_k=5),
            llm=SimpleNamespace(providers=SimpleNamespace(primary=SimpleNamespace(models=SimpleNamespace(en="base-model")))),
            fork=SimpleNamespace(language="en", supported_languages=[]),
            prompts=SimpleNamespace(languages={"de": object(), "en": object()}),
        ),
    )
    response = client.get("/api/system-config")
    assert response.status_code == 200
    body = response.json()
    assert body["supported_languages"] == ["de", "en"]

    # Fallback 2: derive from fork.language when no prompt files are available.
    monkeypatch.setattr(
        "backend.routers.settings.load_core_config",
        lambda: SimpleNamespace(
            rag=SimpleNamespace(collection_name="framework", top_k=5),
            llm=SimpleNamespace(providers=SimpleNamespace(primary=SimpleNamespace(models=SimpleNamespace(en="base-model")))),
            fork=SimpleNamespace(language="en", supported_languages=[]),
            prompts=SimpleNamespace(languages={}),
        ),
    )
    response = client.get("/api/system-config")
    assert response.status_code == 200
    body = response.json()
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