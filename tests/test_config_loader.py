from pathlib import Path

import pytest

from universal_agentic_framework.config import (
    get_active_profile_id,
    load_agents_config,
    load_core_config,
    load_features_config,
    load_profile_metadata,
    load_tools_config,
)


def _create_profile_dir(profiles_dir: Path, profile_id: str) -> Path:
    profile_dir = profiles_dir / profile_id
    profile_dir.mkdir(parents=True)
    return profile_dir


def _write_profile_metadata(profile_dir: Path, profile_id: str, display_name: str = "Medical Assistant") -> None:
    profile_dir.joinpath("profile.yaml").write_text(
        f"profile_id: {profile_id}\ndisplay_name: {display_name}\n",
        encoding="utf-8",
    )


def test_load_core_config_env_substitution() -> None:
    env = {
        "PROFILE_ID": "starter",
        "POSTGRES_USER": "app",
        "POSTGRES_PASSWORD": "pw",
        "POSTGRES_HOST": "localhost",
        "POSTGRES_PORT": "5432",
        "POSTGRES_DB": "framework",
        "INGEST_COLLECTION": "framework",
        "RAG_DATA_PATH": "./data/rag-data",
        "LLM_PROVIDERS_LMSTUDIO_API_BASE": "http://localhost:1234/v1",
        "LLM_PROVIDERS_OLLAMA_API_BASE": "http://localhost:11434/v1",
        "LLM_PROVIDERS_OPENROUTER_API_BASE": "https://openrouter.ai/api/v1",
        "LLM_PROVIDERS_OPENROUTER_API_KEY": "test-key",
        "QDRANT_HOST": "localhost",
        "EMBEDDING_SERVER": "http://localhost:8000/v1",
        "CHECKPOINTER_POSTGRES_DSN": "postgresql://app:pw@localhost:5432/framework",
    }

    core = load_core_config(env=env)
    registry = core.llm.providers.get_registry()

    assert core.fork.name == "starter"
    assert core.database.url == "postgresql://app:pw@localhost:5432/framework"
    assert core.rag is not None
    assert core.rag.collection_name == "framework"
    assert core.ingestion.collection_name == "framework"
    assert core.ingestion.source_path == "./data/rag-data"
    assert core.llm.roles.chat.providers[0].provider_id == "lmstudio"
    assert registry["lmstudio"].models.en == "openai/liquid/lfm2-24b-a2b"
    assert "localhost:1234" in str(registry["lmstudio"].api_base)


def test_load_agents_config_defaults(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    profiles_dir = tmp_path / "profiles"
    config_dir.mkdir()
    config_dir.joinpath("agents.yaml").write_text("crews: {}\n", encoding="utf-8")
    _create_profile_dir(profiles_dir, "starter")

    agents = load_agents_config(
        config_dir=config_dir,
        profiles_dir=profiles_dir,
        env={"PROFILE_ID": "starter"},
    )

    assert agents.crews == {}


def test_load_tools_config_defaults(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    profiles_dir = tmp_path / "profiles"
    config_dir.mkdir()
    config_dir.joinpath("tools.yaml").write_text("tools: []\n", encoding="utf-8")
    _create_profile_dir(profiles_dir, "starter")

    tools = load_tools_config(
        config_dir=config_dir,
        profiles_dir=profiles_dir,
        env={"PROFILE_ID": "starter"},
    )

    assert tools.tools == []


def test_load_features_config_defaults(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    profiles_dir = tmp_path / "profiles"
    config_dir.mkdir()
    config_dir.joinpath("features.yaml").write_text(
        """
        multi_agent_crews: true
        long_term_memory: false
        ingestion_service: true
        ui_tool_visualization: true
        ui_token_counter: false
        ui_export_chat: true
        """,
        encoding="utf-8",
    )
    _create_profile_dir(profiles_dir, "starter")

    features = load_features_config(
        config_dir=config_dir,
        profiles_dir=profiles_dir,
        env={"PROFILE_ID": "starter"},
    )

    assert features.multi_agent_crews is True
    assert features.ingestion_service is True
    assert features.ui_export_chat is True


def test_get_active_profile_id_requires_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PROFILE_ID", raising=False)

    with pytest.raises(ValueError, match="PROFILE_ID must be set"):
        get_active_profile_id()


def test_get_active_profile_id_rejects_base() -> None:
    with pytest.raises(ValueError, match="no longer a valid runtime profile"):
        get_active_profile_id({"PROFILE_ID": "base"})


def test_load_profile_metadata_valid_profile(tmp_path: Path) -> None:
    profiles_dir = tmp_path / "profiles"
    profile_dir = _create_profile_dir(profiles_dir, "medical")
    _write_profile_metadata(profile_dir, "medical")

    metadata = load_profile_metadata(profiles_dir=profiles_dir, profile_id="medical")

    assert metadata is not None
    assert metadata.profile_id == "medical"
    assert metadata.display_name == "Medical Assistant"


def test_load_core_config_applies_profile_overlay_for_allowed_fields(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    profiles_dir = tmp_path / "profiles"
    profile_dir = _create_profile_dir(profiles_dir, "medical")
    config_dir.mkdir()

    config_dir.joinpath("core.yaml").write_text(
        """
fork:
  name: $PROFILE_ID
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
        """,
        encoding="utf-8",
    )
    _write_profile_metadata(profile_dir, "medical")
    profile_dir.joinpath("core.yaml").write_text(
        """
fork:
  language: de
  locale: de_DE
llm:
  providers:
    lmstudio:
      api_base: http://localhost:11434/v1
      models:
        en: openai/base-model
        de: openai/base-model-de
  roles:
    chat:
      providers:
        - provider_id: lmstudio
      config_only: false
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
tokens:
  default_budget: 20000
rag:
  enabled: true
  collection_name: medical-rag
ingestion:
  collection_name: medical-rag
prompts:
  response_system:
    en: Profile prompt
        """,
        encoding="utf-8",
    )

    core = load_core_config(
        config_dir=config_dir,
        profiles_dir=profiles_dir,
        env={"PROFILE_ID": "medical"},
    )

    assert core.fork.language == "de"
    assert core.tokens.default_budget == 20000
    assert core.ingestion.collection_name == "medical-rag"
    assert core.database.url == "sqlite:///base.db"


def test_load_core_config_rejects_disallowed_profile_override(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    profiles_dir = tmp_path / "profiles"
    profile_dir = _create_profile_dir(profiles_dir, "medical")
    config_dir.mkdir()

    config_dir.joinpath("core.yaml").write_text(
        """
fork:
  name: $PROFILE_ID
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
tokens:
  default_budget: 10000
        """,
        encoding="utf-8",
    )
    _write_profile_metadata(profile_dir, "medical")
    profile_dir.joinpath("core.yaml").write_text(
        "database:\n  url: sqlite:///override.db\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="deployment-global core setting"):
        load_core_config(
            config_dir=config_dir,
            profiles_dir=profiles_dir,
            env={"PROFILE_ID": "medical"},
        )


def test_load_core_config_exposes_provider_registry_without_legacy_aliases(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    profiles_dir = tmp_path / "profiles"
    profile_dir = _create_profile_dir(profiles_dir, "starter")
    config_dir.mkdir()

    config_dir.joinpath("core.yaml").write_text(
        """
fork:
  name: $PROFILE_ID
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
        """,
        encoding="utf-8",
    )
    _write_profile_metadata(profile_dir, "starter", display_name="Starter")
    profile_dir.joinpath("core.yaml").write_text(
        """
fork:
  language: en
llm:
  providers:
    lmstudio:
      api_base: http://localhost:1234/v1
      models:
        en: openai/liquid/lfm2-24b-a2b
    openrouter:
      api_base: https://openrouter.ai/api/v1
      api_key: test-key
      models:
        en: openrouter/openai/gpt-4o-mini
  roles:
    chat:
      providers:
        - provider_id: lmstudio
        - provider_id: openrouter
      config_only: false
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
tokens:
  default_budget: 10000
ingestion:
  collection_name: starter-rag
        """,
        encoding="utf-8",
    )

    core = load_core_config(
        config_dir=config_dir,
        profiles_dir=profiles_dir,
        env={"PROFILE_ID": "starter"},
    )
    registry = core.llm.providers.get_registry()

    assert [ref.provider_id for ref in core.llm.roles.chat.providers] == ["lmstudio", "openrouter"]
    assert registry["lmstudio"].models.en == "openai/liquid/lfm2-24b-a2b"
    assert registry["openrouter"].models.en == "openrouter/openai/gpt-4o-mini"
    with pytest.raises(AttributeError):
        _ = core.llm.providers.primary


def test_load_tools_config_name_aware_merge(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    profiles_dir = tmp_path / "profiles"
    profile_dir = _create_profile_dir(profiles_dir, "medical")
    config_dir.mkdir()

    config_dir.joinpath("tools.yaml").write_text(
        "loading_mode: explicit\n"
        "tools:\n"
        "  - name: web_search_mcp\n"
        "    path: base/path\n"
        "    enabled: true\n"
        "    config:\n"
        "      server_url: http://base\n"
        "      default_tool: search\n",
        encoding="utf-8",
    )
    _write_profile_metadata(profile_dir, "medical")
    profile_dir.joinpath("tools.yaml").write_text(
        "tools:\n"
        "  - name: web_search_mcp\n"
        "    config:\n"
        "      default_tool: fetch_content\n"
        "  - name: patient_lookup\n"
        "    path: profiles/medical/tools/patient_lookup\n"
        "    enabled: true\n",
        encoding="utf-8",
    )

    tools = load_tools_config(
        config_dir=config_dir,
        profiles_dir=profiles_dir,
        env={"PROFILE_ID": "medical"},
    )

    assert tools.tools[0].name == "web_search_mcp"
    assert tools.tools[0].model_dump()["config"]["server_url"] == "http://base"
    assert tools.tools[0].model_dump()["config"]["default_tool"] == "fetch_content"
    assert tools.tools[1].name == "patient_lookup"