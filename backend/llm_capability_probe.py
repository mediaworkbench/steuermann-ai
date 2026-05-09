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
        llm_cfg = self._core_config.llm
        registry = llm_cfg.providers.get_registry()
        role_chain = llm_cfg.roles.chat.providers
        targets: list[ProbeTarget] = []
        seen: set[str] = set()

        for ref in role_chain:
            provider_id = str(ref.provider_id)
            if provider_id in seen:
                continue
            provider = registry.get(provider_id)
            if provider is None:
                logger.warning("Skipping unknown probe provider", provider_id=provider_id)
                continue
            model_name = self._select_model(provider, self._core_config.fork.language)
            targets.append(ProbeTarget(provider_id=provider_id, provider=provider, model_name=model_name))
            seen.add(provider_id)

        return targets

    def _probe_target(self, target: ProbeTarget) -> LLMCapabilityProbeResult:
        tool_mode = str(getattr(target.provider, "tool_calling", "structured"))
        api_base_raw = getattr(target.provider, "api_base", None)
        api_base = str(api_base_raw) if api_base_raw else None

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
                metadata={"probe_kind": "non_native_mode"},
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
                metadata={"probe_kind": "native_bind_tools"},
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
                metadata={"probe_kind": "native_bind_tools"},
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
                metadata={"probe_kind": "native_bind_tools"},
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
