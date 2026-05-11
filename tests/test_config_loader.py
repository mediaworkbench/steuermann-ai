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


def test_load_core_config_env_substitution(tmp_path: Path, monkeypatch):
    # copy sample core.yaml into temp dir
    config_dir = tmp_path
    sample = Path("config/core.yaml").read_text()
    config_dir.joinpath("core.yaml").write_text(sample)

    env = {
        "PROFILE_ID": "starter",
        "POSTGRES_USER": "app",
        "POSTGRES_PASSWORD": "pw",
        "POSTGRES_HOST": "localhost",
        "POSTGRES_PORT": "5432",
        "POSTGRES_DB": "framework",
        "INGEST_COLLECTION": "framework",
        # LLM and Qdrant env vars
        "LLM_PROVIDERS_LMSTUDIO_API_BASE": "http://localhost:1234/v1",
        "LLM_PROVIDERS_OLLAMA_API_BASE": "http://localhost:11434/v1",
        "LLM_PROVIDERS_OPENROUTER_API_BASE": "https://openrouter.ai/api/v1",
        "QDRANT_HOST": "localhost",
        "WEB_SEARCH_MCP_URL": "http://localhost:9100",
    }

    core = load_core_config(config_dir=config_dir, base_dir=None, env=env)

    assert core.fork.name == "starter"
    assert core.rag.collection_name == "framework"
    assert "postgresql://app:pw@localhost:5432/framework" in core.database.url
    assert core.llm.providers.primary.models.en == "openai/liquid/lfm2-24b-a2b"
    # Pydantic HttpUrl may normalize, so compare using host/port fragment.
    assert "localhost:1234" in str(core.llm.providers.primary.api_base)
    assert core.tokens.default_budget == 10000


def test_load_agents_config_defaults(tmp_path: Path):
    config_dir = tmp_path
    config_dir.joinpath("agents.yaml").write_text("crews: {}\n")

    agents = load_agents_config(config_dir=config_dir, env={"PROFILE_ID": "base"})
    assert agents.crews == {}


def test_load_tools_config_defaults(tmp_path: Path):
    config_dir = tmp_path
    config_dir.joinpath("tools.yaml").write_text("tools: []\n")

    tools = load_tools_config(config_dir=config_dir, env={"PROFILE_ID": "base"})
    assert tools.tools == []


def test_load_features_config_defaults(tmp_path: Path):
    config_dir = tmp_path
    config_dir.joinpath("features.yaml").write_text(
        """
        multi_agent_crews: true
        long_term_memory: false
        ingestion_service: true
        ui_tool_visualization: true
        ui_token_counter: false
        ui_export_chat: true
        """
    )

    features = load_features_config(config_dir=config_dir, env={"PROFILE_ID": "base"})
    assert features.multi_agent_crews is True
    assert features.ingestion_service is True
    assert features.ui_export_chat is True


def test_get_active_profile_id_defaults_to_base(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PROFILE_ID", raising=False)
        assert get_active_profile_id() == "base"


def test_load_profile_metadata_valid_profile(tmp_path: Path) -> None:
        profiles_dir = tmp_path / "profiles"
        profile_dir = profiles_dir / "medical"
        profile_dir.mkdir(parents=True)
        profile_dir.joinpath("profile.yaml").write_text(
                "profile_id: medical\ndisplay_name: Medical Assistant\n",
                encoding="utf-8",
        )

        metadata = load_profile_metadata(profiles_dir=profiles_dir, profile_id="medical")

        assert metadata is not None
        assert metadata.profile_id == "medical"
        assert metadata.display_name == "Medical Assistant"


def test_load_core_config_applies_profile_overlay_for_allowed_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_dir = tmp_path / "config"
    profiles_dir = tmp_path / "profiles"
    profile_dir = profiles_dir / "medical"
    config_dir.mkdir()
    profile_dir.mkdir(parents=True)

    config_dir.joinpath("core.yaml").write_text(
        """
fork:
    name: $PROFILE_ID
    language: en
llm:
    providers:
        primary:
            api_base: $LLM_PROVIDERS_OLLAMA_API_BASE
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
tokens:
    default_budget: 10000
prompts:
    response_system:
        en: Base prompt
        """,
        encoding="utf-8",
    )
    profile_dir.joinpath("profile.yaml").write_text(
        "profile_id: medical\ndisplay_name: Medical Assistant\n",
        encoding="utf-8",
    )
    profile_dir.joinpath("core.yaml").write_text(
        """
fork:
    language: de
prompts:
    response_system:
        en: Profile prompt
tokens:
    default_budget: 20000
        """,
        encoding="utf-8",
    )
    monkeypatch.setenv("PROFILE_ID", "medical")

    core = load_core_config(
        config_dir=config_dir,
        profiles_dir=profiles_dir,
        env={"PROFILE_ID": "medical", "LLM_PROVIDERS_OLLAMA_API_BASE": "http://localhost:11434/v1"},
    )

    assert core.fork.language == "de"
    assert core.prompts.response_system["en"] == "Profile prompt"
    assert core.tokens.default_budget == 20000
    assert core.database.url == "sqlite:///base.db"


def test_load_core_config_rejects_disallowed_profile_override(tmp_path: Path) -> None:
        config_dir = tmp_path / "config"
        profiles_dir = tmp_path / "profiles"
        profile_dir = profiles_dir / "medical"
        config_dir.mkdir()
        profile_dir.mkdir(parents=True)
        config_dir.joinpath("core.yaml").write_text(
                """
fork:
    name: starter
    language: en
llm:
    providers:
        primary:
            api_key: profile-key
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
tokens:
    default_budget: 10000
                """,
                encoding="utf-8",
        )
        profile_dir.joinpath("profile.yaml").write_text(
                "profile_id: medical\ndisplay_name: Medical Assistant\n",
                encoding="utf-8",
        )
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


def test_load_core_config_role_based_llm_shape_keeps_legacy_compat(tmp_path: Path) -> None:
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_dir.joinpath("core.yaml").write_text(
                """
fork:
    name: starter
    language: en
llm:
    providers:
        lmstudio:
            api_base: http://localhost:1234/v1
            models:
                en: lm_studio/liquid/lfm2-24b-a2b
            model_tool_calling:
                openai/liquid/lfm2-24b-a2b: native
        openrouter:
            api_base: https://openrouter.ai/api/v1
            api_key: test-key
            models:
                en: openrouter/openai/gpt-4o-mini
            model_tool_calling:
                openrouter/openai/gpt-4o-mini: native
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

        core = load_core_config(config_dir=config_dir, env={"PROFILE_ID": "base"})

        assert core.llm.roles is not None
        assert [ref.provider_id for ref in core.llm.roles.chat.providers] == ["lmstudio", "openrouter"]
        assert core.llm.providers.primary is not None
        assert core.llm.providers.primary.models.en == "lm_studio/liquid/lfm2-24b-a2b"
        assert core.llm.providers.fallback is not None
        assert core.llm.providers.fallback.models.en == "openrouter/openai/gpt-4o-mini"


def test_load_tools_config_name_aware_merge(tmp_path: Path) -> None:
        config_dir = tmp_path / "config"
        profiles_dir = tmp_path / "profiles"
        profile_dir = profiles_dir / "medical"
        config_dir.mkdir()
        profile_dir.mkdir(parents=True)
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
        profile_dir.joinpath("profile.yaml").write_text(
                "profile_id: medical\ndisplay_name: Medical Assistant\n",
                encoding="utf-8",
        )
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
