import os
import pytest


@pytest.fixture(autouse=True)
def _hermetic_auth_env(monkeypatch):
    """Keep tests independent of the deployment's `.env`.

    The langsmith pytest plugin auto-loads `.env` into the process, so a real
    `AUTH_ENABLED=true` / `CHAT_ACCESS_TOKEN` would otherwise leak into unit tests that
    assume the single-user dev bypass. Default both off here; tests that exercise auth set
    them explicitly (via their own ``monkeypatch.setenv`` / ``patch.dict``), which runs
    after this fixture and takes precedence.
    """
    monkeypatch.delenv("AUTH_ENABLED", raising=False)
    monkeypatch.delenv("CHAT_ACCESS_TOKEN", raising=False)


# Ensure config placeholder substitution succeeds in tests that call load_core_config() directly.
os.environ.setdefault("LLM_PROVIDERS_OLLAMA_API_BASE", "http://localhost:11434/v1")
os.environ.setdefault("LLM_PROVIDERS_LMSTUDIO_API_BASE", "http://localhost:1234/v1")
os.environ.setdefault("LLM_PROVIDERS_OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("QDRANT_HOST", "localhost")

# Normalize embedding endpoint: EMBEDDING_SERVER in .env already includes /v1,
# but test code previously appended /v1 again.  Canonicalize here so test files
# can safely do `os.environ.get("EMBEDDING_SERVER", "http://localhost:1234") + "/v1"`.
_raw_embedding_server = os.environ.get("EMBEDDING_SERVER", "http://localhost:1234")
if _raw_embedding_server.endswith("/v1"):
    # Strip /v1 so the + "/v1" in test files produces a correct single-suffix URL.
    os.environ["EMBEDDING_SERVER"] = _raw_embedding_server[: -len("/v1")]

# Canonical embedding endpoint for fixtures and integration tests.
EMBEDDING_ENDPOINT = os.environ.get("EMBEDDING_SERVER", "http://localhost:1234") + "/v1"


@pytest.fixture(scope="session")
def _embedding_provider_session():
    """Session-scoped probe: builds the provider and encodes once to verify LM Studio is up.

    Returns the live provider on success, or None if unreachable — never raises.
    """
    from universal_agentic_framework.embeddings import (
        build_embedding_provider,
        EmbeddingProviderUnavailableError,
    )
    try:
        provider = build_embedding_provider(
            model_name="text-embedding-granite-embedding-278m-multilingual",
            dimension=768,
            provider_type="remote",
            remote_endpoint=EMBEDDING_ENDPOINT,
        )
        provider.encode("probe")
        return provider
    except (EmbeddingProviderUnavailableError, ValueError):
        return None


@pytest.fixture
def live_embedding_provider(_embedding_provider_session):
    """Function-scoped fixture: skips the test if the session probe could not reach LM Studio."""
    if _embedding_provider_session is None:
        pytest.skip(f"Embedding provider not reachable at {EMBEDDING_ENDPOINT}")
    return _embedding_provider_session


@pytest.fixture(autouse=True)
def _clear_config_cache_each_test():
    """Reset the loader's merged-config cache around every test.

    load_*_config() caches the env-substituted config dict keyed on PROFILE_ID + paths
    (not on individual env vars). Tests routinely monkeypatch substitution-affecting env
    vars or PROFILE_ID, so a value cached by one test would otherwise leak into the next.
    Clearing before and after keeps each test's config build hermetic.
    """
    try:
        from universal_agentic_framework.config import clear_config_cache
    except (ImportError, ModuleNotFoundError):
        yield
        return
    # Tool discovery is also cached per (profile, language, dir); clear it too so a test
    # that patches tool config/discovery isn't served a prior test's tool list.
    try:
        from universal_agentic_framework.orchestration.graph_builder import (
            clear_tool_registry_cache,
        )
    except (ImportError, ModuleNotFoundError):
        clear_tool_registry_cache = None

    clear_config_cache()
    if clear_tool_registry_cache:
        clear_tool_registry_cache()
    yield
    clear_config_cache()
    if clear_tool_registry_cache:
        clear_tool_registry_cache()


@pytest.fixture(autouse=True)
def _disable_redis_for_tests(monkeypatch):
    """Remove REDIS_URL so performance_nodes uses MemoryCacheBackend silently.

    In the Docker Compose environment REDIS_URL=redis://redis:6379/0 is set, but
    the 'redis' hostname does not resolve outside the compose network.  Without this
    fixture every test session logs an ERROR from manager.py followed by a WARNING
    from performance_nodes.py before falling back to the in-memory cache.
    """
    monkeypatch.delenv("REDIS_URL", raising=False)


@pytest.fixture(autouse=True)
def _disable_checkpointer_for_tests(monkeypatch):
    """Disable checkpointing globally in tests.

    With conditional edges from START, LangGraph enforces thread_id when a checkpointer is
    attached.  Integration tests invoke the graph without a configurable thread_id, and tool
    instances stored in GraphState are not msgpack-serialisable anyway.  Patching
    build_checkpointer to return None keeps tests self-contained.
    """
    try:
        monkeypatch.setattr(
            "universal_agentic_framework.orchestration.graph_builder.build_checkpointer",
            lambda *_, **__: None,
        )
    except (ImportError, ModuleNotFoundError):
        pass  # langgraph not available in this environment (e.g. ingestion container)


@pytest.fixture(autouse=True)
def _disable_crews_for_tests(monkeypatch):
    """Disable multi-agent crews during tests to prevent real LLM initialization."""
    try:
        from universal_agentic_framework.orchestration import crew_nodes
    except (ImportError, ModuleNotFoundError):
        return  # orchestration deps not available in this environment (e.g. ingestion container)
    original_load = crew_nodes.load_features_config
    
    def mock_load_features(*args, **kwargs):
        cfg = original_load(*args, **kwargs)
        # Disable multi_agent_crews in the config
        cfg.multi_agent_crews = False
        return cfg
    
    monkeypatch.setattr(crew_nodes, "load_features_config", mock_load_features)
