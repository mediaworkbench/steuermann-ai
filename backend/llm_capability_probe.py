from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Literal, Optional
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field

from universal_agentic_framework.config import get_active_profile_id, load_core_config
from universal_agentic_framework.llm.factory import LLMFactory

logger = logging.getLogger(__name__)

_REASONING_MODEL_PATTERNS: list[str] = [
    "deepseek-r1", "r1-lite", "r1-zero", "deepseek-r2",
    "qwq", "qwen3",
    "phi-4-reasoning", "phi4-reasoning", "phi-4-mini-reasoning",
    "lfm2", "lfm-2", "lfm2.5", "lfm-2.5",
    "gemma-4", "gemma4",
    "magistral",
    "reflection",
    "thinking", "reasoner", "marco-o1", "skywork-o1",
]


def _detect_vision_from_model_entry(m: dict) -> "bool | None":
    """Return True/False/None from a single /models entry.

    None means no recognisable vision field was present — caller treats as unknown.
    Checks multiple shapes used by LM Studio, Ollama, and OpenRouter.
    """
    # LM Studio: "type": "vlm"
    model_type = str(m.get("type") or "").lower()
    if model_type in ("vlm", "vision", "multimodal"):
        return True

    caps = m.get("capabilities")

    if isinstance(caps, list):
        caps_lower = [str(c).lower() for c in caps]
        if any(v in caps_lower for v in ("vision", "image", "multimodal", "visual")):
            return True
        return False  # field present but vision not listed

    if isinstance(caps, dict):
        if caps.get("vision") or caps.get("image") or caps.get("multimodal"):
            return True
        if caps:  # dict has entries but none indicate vision
            return False

    if "vision" in m:
        return bool(m["vision"])

    modality = str(m.get("modality") or "").lower()
    if modality and ("image" in modality or "vision" in modality):
        return True

    return None  # field absent — unknown


def _fetch_model_metadata(api_base: str, model_name: str) -> "dict[str, Any]":
    """Fetch model metadata (context window, vision support) from the provider's /models endpoint.

    Tries the LM Studio native API (/api/v0/models) first — it returns rich fields like
    `type`, `loaded_context_length`, and `capabilities`. Falls back to the standard
    OpenAI-compat path ({api_base}/models) for other providers.
    Missing or unknown fields are omitted from the returned dict.
    """
    raw_id = model_name.split("/", 1)[-1] if "/" in model_name else model_name

    parsed = urlparse(api_base)
    server_root = f"{parsed.scheme}://{parsed.netloc}"
    candidates = [
        (f"{server_root}/api/v0/models", True),   # LM Studio native — rich metadata
        (f"{api_base.rstrip('/')}/models", False),  # OpenAI-compat fallback
    ]

    result: Dict[str, Any] = {}
    try:
        with httpx.Client(timeout=3.0) as client:
            for url, is_native in candidates:
                try:
                    resp = client.get(url)
                    resp.raise_for_status()
                except Exception:
                    continue
                for m in resp.json().get("data", []):
                    if str(m.get("id", "")) in (raw_id, model_name):
                        # Native API: prefer loaded_context_length, then max_context_length
                        # Compat API: prefer context_length, then max_context_length
                        if is_native:
                            ctx = m.get("loaded_context_length") or m.get("max_context_length")
                        else:
                            ctx = m.get("context_length") or m.get("max_context_length")
                        if ctx:
                            result["context_window_tokens"] = int(ctx)
                        vision = _detect_vision_from_model_entry(m)
                        if vision is not None:
                            result["supports_vision"] = vision
                        return result  # found in this endpoint — done
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to fetch model metadata for %s: %s", model_name, exc)
    return result


@dataclass(frozen=True)
class ProbeTarget:
    provider_id: str
    provider: Any
    model_name: str


class LLMCapabilityProbeResult(BaseModel):
    profile_id: str
    provider_id: str
    model_name: str
    api_base: Optional[str] = None
    configured_tool_calling_mode: str
    supports_bind_tools: Optional[bool] = None
    supports_tool_schema: Optional[bool] = None
    capability_mismatch: bool = False
    status: Literal["ok", "warning", "error"]
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LLMCapabilityProbeRunner:
    """Run low-cost capability probes for configured chat providers."""

    def __init__(self, core_config: Any | None = None, profile_id: Optional[str] = None) -> None:
        self._core_config = core_config or load_core_config()
        self._profile_id = profile_id or get_active_profile_id()
        self._factory = LLMFactory(self._core_config)

    def run(self) -> list[LLMCapabilityProbeResult]:
        results: list[LLMCapabilityProbeResult] = []
        for target in self._collect_targets():
            results.append(self._probe_target(target))
        return results

    def reprobe_for_model(self, model_name: str) -> list[LLMCapabilityProbeResult]:
        """Reprobe only the targets whose model_name matches (curated reprobe).

        Returns an empty list when the model is not found in the current config.
        """
        from universal_agentic_framework.llm.provider_registry import normalize_model_id
        normalized = normalize_model_id(model_name)
        matching = [t for t in self._collect_targets() if normalize_model_id(t.model_name) == normalized]
        return [self._probe_target(t) for t in matching]

    def reprobe_model(self, provider_id: str, model_name: str) -> LLMCapabilityProbeResult:
        """Reprobe a specific provider+model combination by explicit provider_id."""
        llm_cfg = self._core_config.llm
        role_chain = llm_cfg.get_role_provider_chain_with_models("chat", self._core_config.profile.language)
        for pid, provider, _ in role_chain:
            if str(pid) == provider_id:
                target = ProbeTarget(provider_id=provider_id, provider=provider, model_name=model_name)
                return self._probe_target(target)
        raise ValueError(f"Unknown provider_id: {provider_id}")

    def _collect_targets(self) -> list[ProbeTarget]:
        """Collect all unique (provider_id, model_name) pairs across all roles except embedding."""
        llm_cfg = self._core_config.llm
        targets: list[ProbeTarget] = []
        seen: set[tuple[str, str]] = set()  # Track (provider_id, model_name) pairs

        # Iterate through all roles except embedding (chat, vision, auxiliary)
        role_names = ["chat", "vision", "auxiliary"]
        for role_name in role_names:
            # Check if role exists in config (may not exist in all test fixtures)
            role_cfg = getattr(llm_cfg.roles, role_name, None)
            if role_cfg is None:
                logger.debug(f"Role {role_name} not found in config, skipping")
                continue

            try:
                role_chain = llm_cfg.get_role_provider_chain_with_models(role_name, self._core_config.profile.language)
            except Exception as exc:
                logger.warning(
                    "Skipping role during probe target collection",
                    extra={"role": role_name, "error": str(exc)},
                )
                continue

            for provider_id, provider, model_name in role_chain:
                pair_key = (str(provider_id), str(model_name))
                if pair_key in seen:
                    continue
                targets.append(ProbeTarget(provider_id=str(provider_id), provider=provider, model_name=str(model_name)))
                seen.add(pair_key)

        # Log probe coverage for observability
        if seen:
            role_coverage = {}
            for provider_id, model_name in seen:
                for role_name in role_names:
                    role_cfg = getattr(llm_cfg.roles, role_name, None)
                    if role_cfg and str(getattr(role_cfg, "provider_id", "")) == provider_id:
                        role_coverage.setdefault(role_name, []).append((provider_id, model_name))

            logger.info(
                f"Probe coverage: {len(targets)} models across {len(set(p for p, _ in seen))} providers",
                role_summary={role: len(models) for role, models in role_coverage.items()},
            )

        return targets

    def _probe_target(self, target: ProbeTarget) -> LLMCapabilityProbeResult:
        getter = getattr(target.provider, "get_tool_calling_mode", None)
        if callable(getter):
            tool_mode = str(getter(target.model_name))
        else:
            tool_mode = str(getattr(target.provider, "tool_calling", "structured"))
        api_base_raw = getattr(target.provider, "api_base", None)
        api_base = str(api_base_raw) if api_base_raw else None
        metadata: Dict[str, Any] = {
            "probe_kind": "native_bind_tools" if tool_mode == "native" else "non_native_mode",
            "max_output_tokens": getattr(target.provider, "max_tokens", None),
            "supports_reasoning": any(p in target.model_name.lower() for p in _REASONING_MODEL_PATTERNS),
        }

        if api_base:
            model_meta = _fetch_model_metadata(api_base, target.model_name)
            metadata.update(model_meta)

        if tool_mode != "native":
            return LLMCapabilityProbeResult(
                profile_id=self._profile_id,
                provider_id=target.provider_id,
                model_name=target.model_name,
                api_base=api_base,
                configured_tool_calling_mode=tool_mode,
                supports_bind_tools=None,
                supports_tool_schema=None,
                capability_mismatch=False,
                status="ok",
                metadata=metadata,
            )

        try:
            model = self._factory._build(target.provider, target.model_name)
        except Exception as exc:
            return LLMCapabilityProbeResult(
                profile_id=self._profile_id,
                provider_id=target.provider_id,
                model_name=target.model_name,
                api_base=api_base,
                configured_tool_calling_mode=tool_mode,
                supports_bind_tools=False,
                supports_tool_schema=False,
                capability_mismatch=True,
                status="error",
                error_message=f"model_init_failed: {exc}",
                metadata=metadata,
            )

        try:
            self._probe_native_bind_tools(model)
            return LLMCapabilityProbeResult(
                profile_id=self._profile_id,
                provider_id=target.provider_id,
                model_name=target.model_name,
                api_base=api_base,
                configured_tool_calling_mode=tool_mode,
                supports_bind_tools=True,
                supports_tool_schema=True,
                capability_mismatch=False,
                status="ok",
                metadata=metadata,
            )
        except Exception as exc:
            return LLMCapabilityProbeResult(
                profile_id=self._profile_id,
                provider_id=target.provider_id,
                model_name=target.model_name,
                api_base=api_base,
                configured_tool_calling_mode=tool_mode,
                supports_bind_tools=False,
                supports_tool_schema=False,
                capability_mismatch=True,
                status="warning",
                error_message=f"bind_tools_failed: {exc}",
                metadata=metadata,
            )

    @staticmethod
    def _select_model(provider: Any, language: str) -> str:
        model = getattr(provider.models, language, None)
        if model:
            return str(model)

        model_values: Iterable[Any]
        if hasattr(provider.models, "model_dump"):
            model_values = provider.models.model_dump().values()
        else:
            model_values = vars(provider.models).values()

        for value in model_values:
            if value:
                return str(value)

        raise ValueError("No model configured for provider")

    @staticmethod
    def _collect_all_models(provider: Any) -> set[str]:
        """Collect all distinct models configured for a provider across all languages."""
        models: set[str] = set()
        
        if hasattr(provider.models, "model_dump"):
            model_dict = provider.models.model_dump()
        else:
            model_dict = vars(provider.models)

        for value in model_dict.values():
            if value:
                models.add(str(value))

        return models

    @staticmethod
    def _probe_native_bind_tools(model: Any) -> None:
        from langchain_core.tools import tool
        from pydantic import BaseModel, Field

        class ProbeArgs(BaseModel):
            text: str = Field(description="Probe text")

        @tool(args_schema=ProbeArgs)
        def probe_echo(text: str) -> str:
            """Echo probe input."""
            return text

        bind_tools = getattr(model, "bind_tools", None)
        if not callable(bind_tools):
            raise RuntimeError("Model does not implement bind_tools")

        bound = bind_tools([probe_echo])
        if bound is None:
            raise RuntimeError("bind_tools returned None")
