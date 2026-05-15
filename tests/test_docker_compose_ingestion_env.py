from __future__ import annotations

from pathlib import Path

import yaml


def test_ingestion_service_includes_required_core_config_env_vars() -> None:
    compose_path = Path(__file__).resolve().parents[1] / "docker-compose.yml"
    compose = yaml.safe_load(compose_path.read_text(encoding="utf-8"))

    ingestion = compose["services"]["ingestion"]
    environment = ingestion.get("environment") or []
    assert isinstance(environment, list)

    env_keys = {
        item.split("=", 1)[0].strip()
        for item in environment
        if isinstance(item, str) and "=" in item
    }

    # These variables are required so `load_core_config()` can substitute placeholders
    # in config/core.yaml without falling back to defaults in ingestion startup.
    assert "PROFILE_ID" in env_keys
    assert "LLM_PROVIDERS_LMSTUDIO_API_BASE" in env_keys
    assert "LLM_PROVIDERS_OLLAMA_API_BASE" in env_keys
    assert "LLM_PROVIDERS_OPENROUTER_API_BASE" in env_keys
