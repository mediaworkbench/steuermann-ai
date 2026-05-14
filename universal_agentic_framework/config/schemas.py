"""Pydantic schemas for framework configuration."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, HttpUrl, PositiveInt, conint, confloat, ConfigDict, field_validator, model_validator

from universal_agentic_framework.llm.provider_registry import normalize_model_id


PROFILE_ID_PATTERN = r"^[a-z0-9_-]+$"


class ForkSettings(BaseModel):
    name: str
    language: str = Field(..., min_length=2, max_length=5)
    locale: Optional[str] = None
    timezone: Optional[str] = None
    supported_languages: List[str] = Field(default_factory=list)


class ProviderModelMap(BaseModel):
    en: Optional[str] = None
    de: Optional[str] = None
    fr: Optional[str] = None
    es: Optional[str] = None
    # allow other language keys dynamically
    model_config = ConfigDict(extra="allow")


class ProviderSettings(BaseModel):
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    models: ProviderModelMap
    temperature: Optional[confloat(ge=0.0, le=2.0)] = 0.7
    max_tokens: Optional[PositiveInt] = None
    timeout: Optional[PositiveInt] = None
    model_tool_calling: Dict[str, Literal["native", "structured", "react"]] = Field(default_factory=dict)
    # LiteLLM router-friendly metadata.
    order: Optional[conint(ge=1)] = None
    weight: Optional[conint(ge=1)] = None
    rpm: Optional[PositiveInt] = None
    tpm: Optional[PositiveInt] = None
    max_parallel_requests: Optional[PositiveInt] = None
    region_name: Optional[str] = None
    model_config = ConfigDict(extra="forbid")

    @field_validator("api_key")
    @classmethod
    def _validate_api_key(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.strip():
            raise ValueError("api_key must not be empty when provided")
        return value

    @model_validator(mode="after")
    def _validate_models_are_litellm_strings(self) -> "ProviderSettings":
        for language, model in self.models.model_dump().items():
            if not model:
                continue
            model_value = normalize_model_id(str(model))
            setattr(self.models, language, model_value)
        if self.model_tool_calling:
            normalized_modes: Dict[str, Literal["native", "structured", "react"]] = {}
            for model_name, mode in self.model_tool_calling.items():
                normalized_modes[normalize_model_id(str(model_name))] = mode
            self.model_tool_calling = normalized_modes
        return self

    def get_tool_calling_mode(self, model_name: Optional[str]) -> Literal["native", "structured", "react"]:
        if not model_name:
            return "structured"
        normalized = normalize_model_id(str(model_name))
        return self.model_tool_calling.get(normalized, "structured")


class LLMProviders(BaseModel):
    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def _coerce_extra_provider_entries(self) -> "LLMProviders":
        if not self.model_extra:
            return self
        for provider_id, raw_value in list(self.model_extra.items()):
            if isinstance(raw_value, ProviderSettings):
                continue
            if isinstance(raw_value, dict):
                self.model_extra[provider_id] = ProviderSettings.model_validate(raw_value)
        return self

    def get_registry(self) -> Dict[str, ProviderSettings]:
        registry: Dict[str, ProviderSettings] = {}
        if self.model_extra:
            for provider_id, provider in self.model_extra.items():
                if isinstance(provider, ProviderSettings):
                    registry[str(provider_id)] = provider
        return registry


class RoleProviderRef(BaseModel):
    provider_id: str
    model: Optional[str] = None
    models: Optional[ProviderModelMap] = None

    @model_validator(mode="after")
    def _normalize_models(self) -> "RoleProviderRef":
        if self.model:
            self.model = normalize_model_id(str(self.model))
        if self.models:
            for language, model_name in self.models.model_dump().items():
                if not model_name:
                    continue
                setattr(self.models, language, normalize_model_id(str(model_name)))
        return self


class LLMRoleSettings(BaseModel):
    provider_id: str
    api_base: str
    api_key: Optional[str] = None
    model: str
    temperature: Optional[confloat(ge=0.0, le=2.0)] = None
    max_tokens: Optional[PositiveInt] = None
    timeout: Optional[PositiveInt] = None

    @model_validator(mode="after")
    def _normalize_models(self) -> "LLMRoleSettings":
        self.model = normalize_model_id(str(self.model))
        return self


class LLMRoles(BaseModel):
    chat: LLMRoleSettings
    embedding: LLMRoleSettings
    vision: LLMRoleSettings
    auxiliary: LLMRoleSettings


class LLMRouterRoutingGroupSettings(BaseModel):
    group_name: str = Field(..., min_length=1)
    models: List[str] = Field(default_factory=list)
    routing_strategy: Optional[str] = None
    routing_strategy_args: Dict[str, Any] = Field(default_factory=dict)


class LLMRouterSettings(BaseModel):
    routing_strategy: str = "simple-shuffle"
    num_retries: PositiveInt = 3
    retry_after: PositiveInt = 1
    allowed_fails: Optional[PositiveInt] = None
    cooldown_time: Optional[PositiveInt] = None
    disable_cooldowns: bool = False
    enable_pre_call_checks: bool = False
    default_max_parallel_requests: Optional[PositiveInt] = None
    set_verbose: bool = False
    debug_level: Optional[str] = None
    fallbacks: List[Dict[str, List[str]]] = Field(default_factory=list)
    default_fallbacks: List[str] = Field(default_factory=list)
    context_window_fallbacks: List[Dict[str, List[str]]] = Field(default_factory=list)
    content_policy_fallbacks: List[Dict[str, List[str]]] = Field(default_factory=list)
    routing_groups: List[LLMRouterRoutingGroupSettings] = Field(default_factory=list)


class LLMSettings(BaseModel):
    providers: LLMProviders = Field(default_factory=LLMProviders)
    roles: LLMRoles
    router: LLMRouterSettings = Field(default_factory=LLMRouterSettings)

    @model_validator(mode="after")
    def _validate_roles(self) -> "LLMSettings":
        role_entries: Dict[str, LLMRoleSettings] = {}
        for role_name in ("chat", "embedding", "vision", "auxiliary"):
            role_cfg = getattr(self.roles, role_name)
            role_entries[role_name] = role_cfg

        # Build runtime provider registry directly from roles.
        provider_payload: Dict[str, Dict[str, Any]] = {}
        for role_name, role_cfg in role_entries.items():
            provider_payload[f"{role_name}:{role_cfg.provider_id}"] = {
                "api_base": role_cfg.api_base,
                "api_key": role_cfg.api_key,
                "models": {"en": role_cfg.model},
                "temperature": role_cfg.temperature if role_cfg.temperature is not None else 0.7,
                "max_tokens": role_cfg.max_tokens,
                "timeout": role_cfg.timeout,
            }

        self.providers = LLMProviders.model_validate(provider_payload)
        return self

    def get_provider(self, provider_id: str) -> ProviderSettings:
        registry = self.providers.get_registry()
        try:
            return registry[provider_id]
        except KeyError as exc:
            raise KeyError(f"Unknown provider_id '{provider_id}'") from exc

    def get_role_provider_chain(self, role_name: str) -> List[tuple[str, ProviderSettings]]:
        chain = self.get_role_provider_chain_with_models(role_name, language="en")
        return [(provider_id, provider) for provider_id, provider, _model_name in chain]

    def get_role_provider(self, role_name: str) -> ProviderSettings:
        chain = self.get_role_provider_chain_with_models(role_name, language="en")
        if not chain:
            raise ValueError(f"llm.roles.{role_name} has no resolvable provider chain")
        return chain[0][1]

    @staticmethod
    def _select_model_from_map(model_map: ProviderModelMap, language: str) -> Optional[str]:
        model = getattr(model_map, language, None)
        if model:
            return str(model)
        for value in model_map.model_dump().values():
            if value:
                return str(value)
        return None

    def get_role_model_name(self, role_name: str, language: str, provider_index: int = 0) -> str:
        role = getattr(self.roles, role_name)
        if provider_index != 0:
            raise IndexError(f"provider_index {provider_index} out of range for role '{role_name}'")
        return str(role.model)

    def get_role_provider_chain_with_models(
        self,
        role_name: str,
        language: str,
    ) -> List[tuple[str, ProviderSettings, str]]:
        role = getattr(self.roles, role_name)
        provider = ProviderSettings.model_validate(
            {
                "api_base": role.api_base,
                "api_key": role.api_key,
                "models": {"en": role.model, language: role.model},
                "temperature": role.temperature if role.temperature is not None else 0.7,
                "max_tokens": role.max_tokens,
                "timeout": role.timeout,
            }
        )
        return [(str(role.provider_id), provider, str(role.model))]

    def get_embedding_provider_type(self) -> str:
        """Embedding transport type used by the embedding provider factory."""
        return "remote"

    def get_embedding_remote_endpoint(self) -> Optional[str]:
        """Resolve embedding endpoint directly from the embedding role config."""
        endpoint = getattr(self.roles.embedding, "api_base", None)
        if endpoint is None:
            return None
        return str(endpoint)


class DatabaseSettings(BaseModel):
    url: str
    pool_size: Optional[PositiveInt] = 10
    echo: bool = False


class VectorStoreSettings(BaseModel):
    type: Literal["mem0"] = "mem0"
    host: str
    port: conint(gt=0) = 6333
    collection_prefix: str


class EmbeddingSettings(BaseModel):
    model: Optional[str] = None
    dimension: PositiveInt
    batch_size: Optional[PositiveInt] = 32
    provider: Literal["remote"] = "remote"
    remote_endpoint: Optional[str] = None  # e.g., http://localhost:8000/v1


class RetentionSettings(BaseModel):
    session_memory_days: PositiveInt = 90
    user_memory_days: PositiveInt = 365


class Mem0Settings(BaseModel):
    search_limit: PositiveInt = 10
    custom_instructions: Optional[str] = None
    llm_provider: str = "openai"  # "lmstudio" for LM Studio servers (uses json_schema response format)


class MemorySettings(BaseModel):
    vector_store: VectorStoreSettings
    embeddings: EmbeddingSettings
    retention: RetentionSettings
    mem0: Mem0Settings = Field(default_factory=Mem0Settings)


class ToolRoutingSettings(BaseModel):
    similarity_threshold: confloat(ge=0.0, le=1.0) = 0.55
    embedding_model: Optional[str] = None
    top_k: Optional[PositiveInt] = 5
    intent_boost: confloat(ge=0.0, le=0.5) = 0.2
    max_retries: PositiveInt = 2
    min_top_score: confloat(ge=0.0, le=1.0) = 0.7
    min_spread: confloat(ge=0.0, le=1.0) = 0.10


class RagSettings(BaseModel):
    enabled: bool = True
    collection_name: Optional[str] = None
    top_k: PositiveInt = 5
    score_threshold: Optional[confloat(ge=0.0)] = None
    with_payload: Union[bool, List[str]] = True
    with_vectors: bool = False
    timeout_seconds: PositiveInt = 30


class TokensSettings(BaseModel):
    default_budget: PositiveInt = 10000
    conversation_budget: Optional[PositiveInt] = None
    per_turn_budget: Optional[PositiveInt] = None
    per_turn_budget_ratio: confloat(gt=0.0, le=1.0) = 0.4
    response_reserve_ratio: confloat(ge=0.0, le=0.5) = 0.15
    enforce_per_node_hard_limit: bool = True
    per_node_budgets: Dict[str, PositiveInt] = Field(default_factory=dict)


class CheckpointingSettings(BaseModel):
    enabled: bool = False
    backend: Literal["sqlite", "postgres"] = "sqlite"
    sqlite_path: str = "./data/checkpoints/langgraph_checkpoints.sqlite"
    postgres_dsn: Optional[str] = None


class IngestionSettings(BaseModel):
    source_path: Optional[str] = None
    language: str = "en"
    language_threshold: confloat(ge=0.0, le=1.0) = 0.8
    embedding_batch_size: PositiveInt = 32
    upsert_batch_size: PositiveInt = 128
    file_concurrency: PositiveInt = 1
    incremental_mode: bool = True
    phase_timing: bool = True
    reingest_timeout_seconds: PositiveInt = 1800


class LanguagePrompts(BaseModel):
    """Prompt templates for a single language."""

    response_system: str = ""
    synthesis: str = ""
    synthesis_with_sources: str = ""
    language_enforcement: str = ""


class PromptsSettings(BaseModel):
    """Optional prompt templates for fork customization.

    Supports both legacy inline format (response_system as str/dict) and
    the new per-language prompt file format (languages dict).
    """

    response_system: Optional[Union[str, Dict[str, str]]] = None
    languages: Dict[str, LanguagePrompts] = Field(default_factory=dict)

    def get_prompt(self, lang: str, prompt_type: str, fallback_lang: str = "en") -> Optional[str]:
        """Resolve a prompt by language with fallback."""
        # New format: per-language prompt files
        if self.languages:
            lang_prompts = self.languages.get(lang) or self.languages.get(fallback_lang)
            if lang_prompts:
                value = getattr(lang_prompts, prompt_type, None)
                if value:
                    return value
        # Legacy format: inline response_system dict
        if prompt_type == "response_system" and self.response_system:
            if isinstance(self.response_system, dict):
                return self.response_system.get(lang) or self.response_system.get(fallback_lang)
            return self.response_system
        return None


class CoreConfig(BaseModel):
    fork: ForkSettings
    llm: LLMSettings
    database: DatabaseSettings
    memory: MemorySettings
    tokens: TokensSettings
    ingestion: IngestionSettings
    checkpointing: Optional[CheckpointingSettings] = None
    prompts: Optional[PromptsSettings] = None
    tool_routing: Optional[ToolRoutingSettings] = None
    rag: Optional[RagSettings] = None


class ProfileMetadata(BaseModel):
    profile_id: str = Field(..., pattern=PROFILE_ID_PATTERN)
    display_name: str = Field(..., min_length=1)
    description: Optional[str] = None
    version: Optional[str] = None
    tool_names: List[str] = Field(default_factory=list)
    plugin_modules: List[str] = Field(default_factory=list)
    labels: Dict[str, str] = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid")


class ProfileBrandingSettings(BaseModel):
    app_name: Optional[str] = None
    role_label: Optional[str] = None
    display_name: Optional[str] = None
    email: Optional[str] = None
    logo_text: Optional[str] = None


class ProfileThemeSettings(BaseModel):
    colors: Dict[str, str] = Field(default_factory=dict)
    fonts: Dict[str, str] = Field(default_factory=dict)
    radius: Dict[str, str] = Field(default_factory=dict)
    custom_css_vars: Dict[str, str] = Field(default_factory=dict)


class ProfileUIConfig(BaseModel):
    branding: ProfileBrandingSettings = Field(default_factory=ProfileBrandingSettings)
    theme: ProfileThemeSettings = Field(default_factory=ProfileThemeSettings)
    model_config = ConfigDict(extra="forbid")


# Agents / Crews
class AgentDefinition(BaseModel):
    role: Optional[str] = None
    goal: Optional[str] = None
    backstory: Optional[str] = None
    tools: Optional[List[str]] = None
    llm_override: Optional[str] = None
    model_config = ConfigDict(extra="allow")


class CrewDefinition(BaseModel):
    enabled: bool = True
    process: Optional[str] = None
    agents: Dict[str, AgentDefinition] = Field(default_factory=dict)
    max_iterations: Optional[PositiveInt] = None
    timeout_seconds: Optional[PositiveInt] = None
    max_retries: Optional[PositiveInt] = 2
    retry_backoff_base: Optional[confloat(ge=1.0)] = 2.0
    model_config = ConfigDict(extra="allow")


class CrewChainStep(BaseModel):
    """One step in a crew chain pipeline."""
    crew_name: str
    input_key: str = "topic"
    output_key: str = "result"
    input_from: Optional[str] = None  # key from previous step's output
    transform: Optional[str] = None   # optional transform function path


class CrewChainDefinition(BaseModel):
    """Declarative crew chain configuration."""
    name: str
    enabled: bool = True
    steps: List[CrewChainStep] = Field(min_length=2)
    fail_fast: bool = True


class AgentsConfig(BaseModel):
    crews: Dict[str, CrewDefinition] = Field(default_factory=dict)
    crew_chains: List[CrewChainDefinition] = Field(default_factory=list)


# Tools
class ToolDefinition(BaseModel):
    name: str
    path: str
    enabled: bool = True
    model_config = ConfigDict(extra="allow")


class ToolsConfig(BaseModel):
    loading_mode: Literal["explicit", "auto_discover"] = "explicit"
    tools: List[ToolDefinition] = Field(default_factory=list)


# Features
class FeaturesConfig(BaseModel):
    multi_agent_crews: bool = False
    long_term_memory: bool = False
    ingestion_service: bool = False
    rag_retrieval: bool = True
    authentication: bool = False
    monitoring: bool = False
    ui_tool_visualization: bool = False
    ui_token_counter: bool = False
    ui_export_chat: bool = False
    crew_result_caching: bool = False
    crew_cache_ttl_seconds: PositiveInt = 3600
    crew_chaining: bool = False
    crew_parallel_execution: bool = False
    crew_result_validation: bool = True
    model_config = ConfigDict(extra="allow")
