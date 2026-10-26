from universal_agentic_framework.config.schemas import (
    CoreConfig,
    DatabaseSettings,
    EmbeddingSettings,
    ProfileSettings,
    IngestionSettings,
    LLMRoleSettings,
    LLMRoles,
    LLMSettings,
    LLMProviders,
    ProviderModelMap,
    ProviderSettings,
    RetentionSettings,
    TokensSettings,
    VectorStoreSettings,
    MemorySettings,
)
from universal_agentic_framework.llm.factory import LLMFactory
from universal_agentic_framework.llm.provider_registry import normalize_model_id


class DummyModel:
    def __init__(self, label: str):
        self.label = label

    def invoke(self, text: str):  # pragma: no cover - not used
        return f"{self.label}:{text}"


def _core_config(primary_model: str = "local-model") -> CoreConfig:
    return CoreConfig(
        profile=ProfileSettings(name="starter", language="en"),
        llm=LLMSettings(
            providers=LLMProviders(),
            roles=LLMRoles(
                chat=LLMRoleSettings(
                    provider_id="lmstudio",
                    api_base="http://localhost:1234/v1",
                    model=primary_model,
                    temperature=0.1,
                    max_tokens=256,
                    timeout=30,
                ),
                embedding=LLMRoleSettings(
                    provider_id="lmstudio",
                    api_base="http://localhost:1234/v1",
                    model="text-embedding-granite-embedding-278m-multilingual",
                ),
                vision=LLMRoleSettings(
                    provider_id="lmstudio",
                    api_base="http://localhost:1234/v1",
                    model=primary_model,
                ),
                auxiliary=LLMRoleSettings(
                    provider_id="lmstudio",
                    api_base="http://localhost:1234/v1",
                    model=primary_model,
                ),
            ),
        ),
        database=DatabaseSettings(url="postgresql://user:pass@localhost:5432/db", pool_size=10, echo=False),
        memory=MemorySettings(
            vector_store=VectorStoreSettings(type="mem0", host="qdrant", port=6333, collection_prefix="starter"),
            embeddings=EmbeddingSettings(dimension=384, batch_size=32),
            retention=RetentionSettings(session_memory_days=90, user_memory_days=365),
        ),
        tokens=TokensSettings(default_budget=10000, per_node_budgets={}),
        ingestion=IngestionSettings(collection_name="starter-rag"),
    )


def test_llm_factory_primary_selection():
    factory = LLMFactory(
        config=_core_config(primary_model="local-model"),
        builders={"openai_compatible": lambda _provider, model: DummyModel(f"mock:{model}")},
    )
    model = factory.get_model(language="en")
    assert isinstance(model, DummyModel)
    assert model.label == "mock:local-model"


def test_llm_factory_uses_primary_for_any_language():
    factory = LLMFactory(
        config=_core_config(primary_model="remote-model"),
        builders={"openai_compatible": lambda _provider, model: DummyModel(f"mock:{model}")},
    )
    model = factory.get_model(language="de")
    assert model.label == "mock:remote-model"


def test_llm_factory_model_selection_metadata():
    factory = LLMFactory(
        config=_core_config(primary_model="local-model"),
        builders={"openai_compatible": lambda _provider, model: DummyModel(f"mock:{model}")},
    )

    selection = factory.get_model_selection(language="en")
    assert isinstance(selection.model, DummyModel)
    assert selection.provider_type == "lmstudio"  # the role's provider_id
    assert selection.model_name == "local-model"
    assert selection.source == "primary_default"


def test_llm_factory_candidate_order_with_preferred_model():
    factory = LLMFactory(
        config=_core_config(primary_model="local-model"),
        builders={"openai_compatible": lambda _provider, model: DummyModel(f"mock:{model}")},
    )

    candidates = factory.get_model_candidates(
        language="en",
        preferred_model="preferred-model",
    )

    assert [candidate.source for candidate in candidates] == [
        "primary_preferred",
        "primary_default",
    ]
    assert [candidate.model_name for candidate in candidates] == [
        "preferred-model",
        "local-model",
    ]


def test_llm_factory_candidate_order_with_preferred_model_strict():
    factory = LLMFactory(
        config=_core_config(primary_model="local-model"),
        builders={"openai_compatible": lambda _provider, model: DummyModel(f"mock:{model}")},
    )

    candidates = factory.get_model_candidates(
        language="en",
        preferred_model="preferred-model",
        include_default_when_preferred=False,
    )

    assert [candidate.source for candidate in candidates] == [
        "primary_preferred",
    ]
    assert [candidate.model_name for candidate in candidates] == [
        "preferred-model",
    ]


def test_llm_factory_prefers_role_level_model_override():
    config = _core_config(primary_model="provider-default")
    config.llm.roles.chat.model = "chat-role-override"

    factory = LLMFactory(
        config=config,
        builders={"openai_compatible": lambda _provider, model: DummyModel(f"mock:{model}")},
    )

    model = factory.get_model(language="en")
    assert isinstance(model, DummyModel)
    assert model.label == "mock:chat-role-override"


def test_normalize_model_id_keeps_ids_verbatim():
    """Model IDs are the endpoint's own: no prefix is required, stripped or case-folded."""
    assert normalize_model_id("openai/gpt-5") == "openai/gpt-5"  # OpenRouter ID
    assert normalize_model_id("  Qwen/Qwen3-8B ") == "Qwen/Qwen3-8B"
    assert normalize_model_id("llama3.1:8b") == "llama3.1:8b"


def test_normalize_model_id_rejects_empty():
    import pytest

    with pytest.raises(ValueError):
        normalize_model_id("   ")


# ---------------------------------------------------------------------------
# OpenAI-compatible chat model contract
# ---------------------------------------------------------------------------

def test_get_model_builds_openai_compatible_chat():
    """The default builder returns OpenAICompatibleChat configured from the chat role."""
    from universal_agentic_framework.llm.chat_model import OpenAICompatibleChat

    model = LLMFactory(config=_core_config(primary_model="primary-model")).get_model(language="en")

    assert isinstance(model, OpenAICompatibleChat)
    assert model.model_name == "primary-model"
    assert model.openai_api_base == "http://localhost:1234/v1"
    assert model.max_retries == 3
    assert model.streaming is True
    assert model.stream_usage is True
    assert model.use_responses_api is False


def test_get_model_sends_max_tokens_not_max_completion_tokens():
    """OpenAI-compatible servers document ``max_tokens``; ChatOpenAI would rename it."""
    from langchain_core.messages import HumanMessage

    model = LLMFactory(config=_core_config(primary_model="primary-model")).get_model(language="en")
    payload = model._get_request_payload([HumanMessage(content="hi")])

    assert payload["max_tokens"] == 256
    assert "max_completion_tokens" not in payload
    assert "tool_choice" not in payload


def test_get_model_uses_llm_max_retries():
    config = _core_config(primary_model="primary-model")
    config.llm.max_retries = 7

    assert LLMFactory(config=config).get_model(language="en").max_retries == 7


def test_model_id_is_sent_verbatim():
    """A gateway ID that starts with ``openai/`` (OpenRouter) reaches the endpoint unchanged."""
    model = LLMFactory(config=_core_config(primary_model="openai/gpt-5")).get_model(language="en")

    assert model.model_name == "openai/gpt-5"


def test_get_model_with_preferred_model_builds_that_model():
    factory = LLMFactory(
        config=_core_config(primary_model="local-model"),
        builders={"openai_compatible": lambda _provider, model: DummyModel(f"mock:{model}")},
    )

    assert factory.get_model(language="en", preferred_model="preferred-model").label == (
        "mock:preferred-model"
    )


def test_missing_builder_for_api_type_fails():
    import pytest

    factory = LLMFactory(
        config=_core_config(primary_model="local-model"),
        builders={"other": lambda _provider, model: DummyModel(model)},
    )
    with pytest.raises(RuntimeError, match="No LLM builder registered for api_type 'openai_compatible'"):
        factory.get_model(language="en")


def test_llm_router_section_is_rejected():
    import pytest
    from pydantic import ValidationError

    roles = _core_config(primary_model="local-model").llm.roles.model_dump()
    with pytest.raises(ValidationError, match="llm.router is no longer supported"):
        LLMSettings.model_validate({"roles": roles, "router": {"num_retries": 3}})


def test_llm_max_retries_defaults_to_three():
    assert _core_config(primary_model="local-model").llm.max_retries == 3


def test_llm_client_imports_are_confined():
    """No litellm anywhere; langchain_openai (internal hooks) only in llm/chat_model.py."""
    import ast
    import os

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    always_forbidden = {"litellm", "langchain_litellm", "langchain_ollama"}
    allowed_openai_module = os.path.join(repo_root, "universal_agentic_framework", "llm", "chat_model.py")
    violations = []
    scanned = 0

    for top_dir in ("universal_agentic_framework", "backend"):
        for root, _dirs, files in os.walk(os.path.join(repo_root, top_dir)):
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                path = os.path.join(root, fname)
                scanned += 1
                try:
                    tree = ast.parse(open(path).read(), filename=path)
                except SyntaxError:
                    continue
                for node in ast.walk(tree):
                    if not isinstance(node, (ast.Import, ast.ImportFrom)):
                        continue
                    module = node.module if isinstance(node, ast.ImportFrom) else None
                    for alias in node.names:
                        top = (module or alias.name).split(".")[0]
                        if top in always_forbidden:
                            violations.append(f"{path}: imports {top}")
                        elif top == "langchain_openai" and path != allowed_openai_module:
                            violations.append(f"{path}: imports langchain_openai outside llm/chat_model.py")

    assert scanned > 50, f"import guard scanned only {scanned} files — wrong root?"
    assert not violations, "Disallowed LLM client imports found:\n" + "\n".join(violations)
