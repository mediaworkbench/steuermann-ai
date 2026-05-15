from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Literal, Optional

from pydantic import BaseModel, Field

from universal_agentic_framework.config import get_active_profile_id, load_core_config
from universal_agentic_framework.llm.factory import LLMFactory

logger = logging.getLogger(__name__)


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

    def reprobe_model(self, provider_id: str, model_name: str) -> LLMCapabilityProbeResult:
        """Reprobe a specific provider+model combination (curated reprobe)."""
        llm_cfg = self._core_config.llm
        registry = llm_cfg.providers.get_registry()
        provider = registry.get(provider_id)
        if provider is None:
            raise ValueError(f"Unknown provider_id: {provider_id}")

        target = ProbeTarget(provider_id=provider_id, provider=provider, model_name=model_name)
        return self._probe_target(target)

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
                role_chain = llm_cfg.get_role_provider_chain_with_models(role_name, self._core_config.fork.language)
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
        metadata = {
            "probe_kind": "native_bind_tools" if tool_mode == "native" else "non_native_mode",
            "capabilities": {
                "max_output_tokens": getattr(target.provider, "max_tokens", None),
                "supports_streaming": None,
                "supports_json_mode": None,
            },
            "confidence": {
                "max_output_tokens": "medium",
                "supports_streaming": "low",
                "supports_json_mode": "low",
            },
            "origin": {
                "max_output_tokens": "config",
                "supports_streaming": "unknown",
                "supports_json_mode": "unknown",
            },
        }

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
