"""Tool registry supporting LangChain tools and MCP servers."""

from __future__ import annotations

import asyncio
import json
import hashlib
import importlib
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml
from langchain_core.tools import BaseTool

try:  # Optional import to avoid hard dependency at import time
    from universal_agentic_framework.config.schemas import ToolsConfig
except Exception:  # pragma: no cover - fallback when schema not available
    ToolsConfig = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class ToolManifest:
    """Normalized tool manifest."""

    name: str
    type: str  # langchain_tool | mcp_server
    entry_point: Optional[str]
    description: Optional[str]
    config: Dict[str, Any]
    path: Path


class MCPServerTool(BaseTool):
    """MCP server wrapper using the official MCP Python SDK.

    Connects to an MCP server via streamable-http transport and invokes
    tools using the standard MCP protocol (ClientSession.call_tool).
    """

    name: str
    description: str = "Execute requests against an MCP server"
    server_url: str
    default_tool: str = "web_search"
    rag_data_path: str = ""

    def _normalize_tool_query(self, tool_name: str, query: Any, kwargs: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """Normalize model-produced argument shapes into expected MCP inputs."""
        normalized_kwargs = dict(kwargs)
        normalized_query = "" if query is None else str(query)

        def _find_url(value: Any) -> Optional[str]:
            if isinstance(value, str):
                candidate = value.strip().strip('"\'')
                if re.match(r"^(https?://|www\.)", candidate):
                    return candidate
                return None
            if isinstance(value, dict):
                for v in value.values():
                    found = _find_url(v)
                    if found:
                        return found
                return None
            if isinstance(value, (list, tuple)):
                for item in value:
                    found = _find_url(item)
                    if found:
                        return found
                return None
            return None

        if tool_name != "fetch_content":
            return normalized_query, normalized_kwargs

        # Models may emit URL under alternate keys instead of query.
        for key in ("request_url", "url", "link", "path", "webpage", "website"):
            value = normalized_kwargs.pop(key, None)
            if value:
                normalized_query = str(value)
                break

        # Some model/tool wrappers pass a JSON object as the single query string.
        stripped = normalized_query.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = json.loads(stripped)
                for key in ("request_url", "url", "link", "path", "webpage", "website", "query"):
                    value = parsed.get(key)
                    if value:
                        normalized_query = str(value)
                        break
                # Preserve supported optional fetch parameters if provided.
                for key in ("max_length", "start_index"):
                    if key in parsed and key not in normalized_kwargs:
                        normalized_kwargs[key] = parsed[key]
            except Exception:
                pass

        # Native tool-calling can produce key=value strings instead of JSON.
        kv_match = re.search(
            r'(?:request_url|url|link)\s*=\s*["\']?(https?://[^"\'\s\)\]]+|www\.[^"\'\s\)\]]+)',
            stripped,
        )
        if kv_match:
            normalized_query = kv_match.group(1)

        # If query text is not a URL, prefer a URL-looking value from remaining kwargs.
        if normalized_query and not re.match(r"^(https?://|www\.)", normalized_query):
            for candidate in normalized_kwargs.values():
                found = _find_url(candidate)
                if found:
                    normalized_query = found
                    break

        normalized_query = normalized_query.strip().strip("\"'")
        if normalized_query and not normalized_query.startswith(("http://", "https://")):
            if normalized_query.startswith("www."):
                normalized_query = f"https://{normalized_query}"

        return normalized_query, normalized_kwargs

    def _run(self, query: str = "", *args: Any, **kwargs: Any) -> str:  # type: ignore[override]
        """Execute MCP tool via the official MCP SDK (per-call session)."""
        if not self.server_url.rstrip("/").endswith("/mcp"):
            return self._run_legacy_http(query, *args, **kwargs)

        import concurrent.futures

        def _run_in_new_loop() -> str:
            new_loop = asyncio.new_event_loop()
            try:
                return new_loop.run_until_complete(self._arun(query, *args, **kwargs))
            finally:
                new_loop.close()

        # Run in a separate thread with its own event loop to avoid
        # conflicts with the already-running uvloop.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_run_in_new_loop)
            return future.result(timeout=60)

    def _run_legacy_http(self, query: str = "", *args: Any, **kwargs: Any) -> str:
        """Backward-compatible HTTP invocation for legacy MCP adapters."""
        try:
            import httpx

            tool_name = kwargs.pop("tool", self.default_tool)
            save_to_rag_flag = kwargs.pop("save_to_rag", False)
            query, kwargs = self._normalize_tool_query(tool_name, query, kwargs)
            parameters = {"url": query, **kwargs} if tool_name == "fetch_content" else {"query": query, **kwargs}
            payload = {
                "tool": tool_name,
                "parameters": parameters,
            }
            response = httpx.post(f"{self.server_url.rstrip('/')}/execute", json=payload, timeout=30.0)
            response.raise_for_status()
            body = response.json()

            if not body.get("success", False):
                error_text = body.get("error") or body.get("message") or "unknown error"
                return f"MCP tool error: {error_text}"

            result_text = body.get("result", "")
            if save_to_rag_flag:
                self._save_to_rag(query, str(result_text), tool_name)
            return str(result_text)
        except Exception as e:
            error_msg = f"MCP server '{self.name}' invocation failed: {str(e)}"
            logger.error("MCP invocation failed", extra={
                "server": self.name, "error": str(e),
            })
            return error_msg

    async def _arun(self, query: str = "", *args: Any, **kwargs: Any) -> str:  # type: ignore[override]
        """Async MCP tool execution via streamable-http transport."""
        if not self.server_url.rstrip("/").endswith("/mcp"):
            return await self._arun_legacy_http(query, *args, **kwargs)

        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamablehttp_client

            tool_name = kwargs.pop("tool", self.default_tool)
            save_to_rag_flag = kwargs.pop("save_to_rag", False)
            query, kwargs = self._normalize_tool_query(tool_name, query, kwargs)

            logger.info("Invoking MCP server", extra={
                "server": self.name,
                "url": self.server_url,
                "tool": tool_name,
            })

            # Build tool arguments for the MCP call_tool request
            if tool_name == "fetch_content":
                arguments = {"url": query, **kwargs}
            else:
                arguments = {"query": query, **kwargs}

            # Per-call session via streamable-http transport
            async with streamablehttp_client(self.server_url) as (
                read_stream,
                write_stream,
                _,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments=arguments)

            # Parse CallToolResult content into a string
            text_parts = []
            for content_block in result.content:
                if hasattr(content_block, "text"):
                    text_parts.append(content_block.text)
                else:
                    text_parts.append(str(content_block))

            result_text = "\n".join(text_parts)

            if result.isError:
                logger.warning("MCP tool returned error", extra={
                    "server": self.name, "tool": tool_name,
                })
                return f"MCP tool error: {result_text}"

            logger.info("MCP server response", extra={
                "server": self.name,
                "tool": tool_name,
                "result_length": len(result_text),
            })

            # Save to RAG if requested
            if save_to_rag_flag:
                self._save_to_rag(query, result_text, tool_name)

            return result_text

        except Exception as e:
            error_msg = f"MCP server '{self.name}' invocation failed: {str(e)}"
            logger.error("MCP invocation failed", extra={
                "server": self.name, "error": str(e),
            })
            return error_msg

    async def _arun_legacy_http(self, query: str = "", *args: Any, **kwargs: Any) -> str:
        """Async backward-compatible HTTP invocation for legacy MCP adapters."""
        try:
            import httpx

            tool_name = kwargs.pop("tool", self.default_tool)
            save_to_rag_flag = kwargs.pop("save_to_rag", False)
            query, kwargs = self._normalize_tool_query(tool_name, query, kwargs)
            parameters = {"url": query, **kwargs} if tool_name == "fetch_content" else {"query": query, **kwargs}
            payload = {
                "tool": tool_name,
                "parameters": parameters,
                "save_to_rag": save_to_rag_flag,
            }
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(f"{self.server_url.rstrip('/')}/execute", json=payload)
                response.raise_for_status()
                body = response.json()

            if not body.get("success", False):
                error_text = body.get("error") or body.get("message") or "unknown error"
                return f"MCP tool error: {error_text}"

            result_text = body.get("result", "")
            if save_to_rag_flag:
                self._save_to_rag(query, str(result_text), tool_name)
            return str(result_text)
        except Exception as e:
            error_msg = f"MCP server '{self.name}' invocation failed: {str(e)}"
            logger.error("MCP invocation failed", extra={
                "server": self.name, "error": str(e),
            })
            return error_msg

    def _save_to_rag(self, query: str, result_text: str, tool_name: str) -> None:
        """Save MCP tool results to RAG data directory as markdown."""
        rag_path_str = self.rag_data_path or os.environ.get("RAG_WEB_DATA_PATH", "")
        if not rag_path_str:
            return
        try:
            rag_path = Path(rag_path_str)
            rag_path.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_query = re.sub(r"[^\w\s-]", "", query)[:50].strip().replace(" ", "_")
            query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
            filename = f"{timestamp}_{safe_query}_{query_hash}.md"

            md_content = f"# Web Search: {query}\n\n"
            md_content += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            md_content += f"**Tool:** {tool_name}\n\n"
            md_content += result_text

            filepath = rag_path / filename
            filepath.write_text(md_content, encoding="utf-8")
            logger.info("Saved to RAG", extra={"path": str(filepath)})
        except Exception as e:
            logger.warning("Failed to save to RAG", extra={"error": str(e)})


class ToolRegistry:
    """Central registry for LangChain tools and MCP servers."""

    def __init__(
        self, 
        config: Optional[Any] = None, 
        base_dir: Optional[Path] = None,
        fork_language: str = "en",
        extra_tools_dir: Optional[Path] = None,
    ) -> None:
        self.config = self._normalize_config(config)
        self.tools: Dict[str, BaseTool] = {}
        self.fork_language = fork_language
        self._builtin_dir = Path(__file__).parent
        self._base_dir = Path(base_dir) if base_dir else Path.cwd()
        self._extra_tools_dir = Path(extra_tools_dir) if extra_tools_dir else None

    def discover_and_load(self) -> List[BaseTool]:
        """Discover manifests, honor config enables, and load tools."""

        manifests = self._collect_manifests()
        for manifest in manifests:
            if not self._is_enabled(manifest):
                logger.info("Tool disabled via config", extra={"tool": manifest.name})
                continue

            try:
                tool = self._load_tool(manifest)
                self.tools[manifest.name] = tool
                logger.info(
                    "Tool loaded", extra={"tool": manifest.name, "type": manifest.type}
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning(
                    "Failed to load tool", extra={"tool": manifest.name, "error": str(exc)}
                )

        return list(self.tools.values())

    # ------------------------------------------------------------------
    # Manifest collection
    # ------------------------------------------------------------------
    def _collect_manifests(self) -> List[ToolManifest]:
        manifests: List[ToolManifest] = []
        discovered = {m.name: m for m in self._discover_builtin_manifests()}
        for manifest in self._discover_profile_manifests():
            discovered[manifest.name] = manifest

        # Apply config entries (override path/type if provided)
        for entry in self._tools_entries():
            name = entry.get("name")
            if not name:
                continue

            if path := entry.get("path"):
                manifest_path = self._resolve_path(path)
                if manifest_path and manifest_path.is_dir():
                    manifest_path = manifest_path / "tool.yaml"
            else:
                manifest_path = discovered.get(name, None)
                manifest_path = manifest_path.path if manifest_path else None

            if not manifest_path or not manifest_path.exists():
                logger.warning(
                    "Tool manifest not found", extra={"tool": name, "path": str(path)}
                )
                continue

            manifest = self._load_manifest(manifest_path)
            # Allow config overrides for type and name
            if entry.get("type"):
                manifest.type = entry["type"]
            # Override manifest name with config name to support multiple instances
            manifest.name = name
            manifests.append(manifest)

        # Optionally include remaining built-ins when auto-discovery is enabled
        if self._loading_mode() == "auto_discover":
            for name, manifest in discovered.items():
                if any(m.name == name for m in manifests):
                    continue
                manifests.append(manifest)

        return manifests

    def _discover_builtin_manifests(self) -> List[ToolManifest]:
        manifests: List[ToolManifest] = []
        for item in self._builtin_dir.iterdir():
            if not item.is_dir() or item.name.startswith("__"):
                continue
            manifest_path = item / "tool.yaml"
            if manifest_path.exists():
                manifests.append(self._load_manifest(manifest_path))
        return manifests

    def _discover_profile_manifests(self) -> List[ToolManifest]:
        if not self._extra_tools_dir or not self._extra_tools_dir.exists():
            return []

        manifests: List[ToolManifest] = []
        seen_names: set[str] = set()
        for item in self._extra_tools_dir.iterdir():
            if not item.is_dir() or item.name.startswith("__"):
                continue
            manifest_path = item / "tool.yaml"
            if not manifest_path.exists():
                continue
            manifest = self._load_manifest(manifest_path)
            if manifest.name in seen_names:
                raise ValueError(f"Duplicate profile tool name '{manifest.name}' in {self._extra_tools_dir}")
            seen_names.add(manifest.name)
            manifests.append(manifest)
        return manifests

    def _load_manifest(self, path: Path) -> ToolManifest:
        data = yaml.safe_load(path.read_text()) or {}
        name = data.get("name") or path.parent.name
        tool_type = data.get("type", "langchain_tool")
        entry_point = data.get("entry_point")
        description = data.get("description")
        return ToolManifest(
            name=name,
            type=tool_type,
            entry_point=entry_point,
            description=description,
            config=data,
            path=path,
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load_tool(self, manifest: ToolManifest) -> BaseTool:
        if manifest.type == "langchain_tool":
            if not manifest.entry_point:
                raise ValueError(f"entry_point required for tool {manifest.name}")
            tool_cls = self._import_entry_point(manifest.entry_point)
            tool = tool_cls(**self._get_tool_config(manifest))
            tool.name = manifest.name
            # Override class-level description with language-specific YAML description
            description = self._get_tool_description(manifest)
            if description:
                tool.description = description
            return tool

        if manifest.type == "mcp_server":
            tool_config = self._get_tool_config(manifest)
            server_url = tool_config.get("server_url")
            if not server_url:
                raise ValueError(f"server_url required for MCP server {manifest.name}")
            
            # Get language-specific description from manifest or config
            description = self._get_tool_description(manifest)
            
            default_tool = tool_config.get("default_tool", "search")
            rag_data_path = tool_config.get("rag_data_path", "")
            return MCPServerTool(
                name=manifest.name,
                description=description,
                server_url=server_url,
                default_tool=default_tool,
                rag_data_path=rag_data_path,
            )

        raise ValueError(f"Unknown tool type: {manifest.type}")

    def _import_entry_point(self, entry_point: str):
        module_path, _, attr = entry_point.partition(":")
        if not module_path or not attr:
            raise ValueError(f"Invalid entry_point: {entry_point}")
        module = importlib.import_module(module_path)
        tool_cls = getattr(module, attr, None)
        if tool_cls is None:
            raise ImportError(f"Cannot find {attr} in {module_path}")
        return tool_cls

    def _get_tool_config(self, manifest: ToolManifest) -> Dict[str, Any]:
        """Merge manifest config defaults with config/tools.yaml overrides."""
        manifest_config = manifest.config.get("config", {}) if manifest.config else {}
        entry_config: Dict[str, Any] = {}
        for entry in self._tools_entries():
            if entry.get("name") == manifest.name:
                entry_config = entry.get("config", {}) or {}
                break
        return {**manifest_config, **entry_config}

    def _get_tool_description(self, manifest: ToolManifest) -> str:
        """Get language-specific description for a tool.
        
        Priority:
        1. Language-specific description from config entry
        2. Tool-specific language description (for tools that expose multiple sub-tools)
        3. Language-specific description from manifest (descriptions.{language})
        4. Generic description from config entry
        5. Generic description from manifest
        6. Tool name as fallback
        """
        # Get fork language from loaded config
        fork_language = self._get_fork_language()
        
        # Check config entry for override
        for entry in self._tools_entries():
            if entry.get("name") == manifest.name:
                if entry.get("description"):
                    return entry["description"]
                break
        
        # Special handling for tools that expose multiple sub-tools via different entries
        tool_name_map = {
            "web_search_mcp": "search",
            "extract_webpage_mcp": "fetch_content",
        }
        
        # Check manifest for tool-specific language descriptions
        manifest_data = manifest.config
        if isinstance(manifest_data, dict):
            tool_specific_key = tool_name_map.get(manifest.name)
            if tool_specific_key:
                tool_descriptions = manifest_data.get("tool_descriptions", {})
                if isinstance(tool_descriptions, dict):
                    tool_desc = tool_descriptions.get(tool_specific_key, {})
                    if isinstance(tool_desc, dict) and fork_language in tool_desc:
                        return tool_desc[fork_language]
                    if isinstance(tool_desc, dict) and "en" in tool_desc:
                        return tool_desc["en"]
            
            # Check generic language-specific descriptions
            descriptions = manifest_data.get("descriptions", {})
            if isinstance(descriptions, dict):
                # Try fork language
                if fork_language in descriptions:
                    return descriptions[fork_language]
                # Try English as fallback
                if "en" in descriptions:
                    return descriptions["en"]
        
        # Fall back to generic description
        return manifest.description or manifest.name

    def _get_fork_language(self) -> str:
        """Get fork language from instance variable or default to 'en'."""
        return self.fork_language or "en"

    def _is_enabled(self, manifest: ToolManifest) -> bool:
        """Return enabled flag from config entry or manifest default (True)."""
        manifest_default = True
        if manifest.config and isinstance(manifest.config, dict):
            manifest_default = manifest.config.get("enabled", True)
        for entry in self._tools_entries():
            if entry.get("name") == manifest.name:
                # If user lists a tool in config, treat it as enabled unless explicitly false
                return entry.get("enabled", True)
        return manifest_default

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _normalize_config(self, raw_config: Optional[Any]) -> Dict[str, Any]:
        if raw_config is None:
            return {"tools": []}
        if ToolsConfig is not None and isinstance(raw_config, ToolsConfig):
            return raw_config.model_dump()
        if hasattr(raw_config, "model_dump"):
            return raw_config.model_dump()  # type: ignore[return-value]
        if isinstance(raw_config, dict):
            return raw_config
        return {"tools": []}

    def _tools_entries(self) -> Iterable[Dict[str, Any]]:
        return self.config.get("tools", [])

    def _loading_mode(self) -> str:
        mode = str(self.config.get("loading_mode", "explicit")).strip().lower()
        if mode not in {"explicit", "auto_discover"}:
            logger.warning("Unknown loading_mode, falling back to explicit", extra={"loading_mode": mode})
            return "explicit"
        return mode

    def _resolve_path(self, path_str: str) -> Optional[Path]:
        candidate = Path(path_str)
        if candidate.exists():
            return candidate
        if not candidate.is_absolute():
            alt = self._base_dir / path_str
            if alt.exists():
                return alt
        return None


__all__ = ["ToolRegistry", "MCPServerTool", "ToolManifest"]
