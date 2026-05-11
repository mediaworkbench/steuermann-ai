# Tool Development Guide

This guide explains how to create, register, and test tools for the Steuermann.

---

## Architecture Overview

The framework supports two types of tools:

| Type | Description | Use When |
| ------ | ------------- | ---------- |
| **LangChain Tool** | Native Python class extending `BaseTool` | Self-contained logic (math, file I/O, date/time) |
| **MCP Server Tool** | HTTP-based tool via Model Context Protocol | External services, sandboxed execution, language-agnostic |

**Discovery flow:**

1. `ToolRegistry` scans `tools/` subdirectories for `tool.yaml` manifests
2. Merges with `config/tools.yaml` overrides (profile can enable/disable)
3. Loads enabled tools into `BaseTool` instances
4. Graph node `node_load_tools` populates `GraphState.loaded_tools`
5. `node_prefilter_tools` scores queries against tool descriptions and selects candidates for model-driven calling

---

## Creating a LangChain Tool

### Step 1: Create the directory

```text
universal_agentic_framework/tools/my_tool/
├── __init__.py
├── tool.py
└── tool.yaml
```

### Step 2: Write `tool.yaml` manifest

```yaml
name: "my_tool"
version: "1.0.0"
category: "utilities"              # utilities, information_retrieval, communication
description: "Short description for fallback"

# These descriptions drive semantic routing — quality matters!
descriptions:
  en: "Full English description. Include trigger words users would say."
  de: "Vollständige deutsche Beschreibung mit Auslösewörtern."

type: "langchain_tool"
entry_point: "universal_agentic_framework.tools.my_tool.tool:MyTool"

dependencies: []                   # pip packages needed (informational)

config_schema:                     # Configurable parameters
  my_param:
    type: "string"
    default: "default_value"
    description: "What this param controls"

permissions:                       # Used by sandbox
  - "system:compute"

cost_per_call: 0.0                 # For cost tracking
```

### Step 3: Write `tool.py`

```python
"""My custom tool."""

from typing import Literal, Optional
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger()


class MyToolInput(BaseModel):
    """Input schema — defines what parameters the tool accepts."""
    operation: Literal["do_thing", "other_thing"] = Field(
        default="do_thing", description="Operation to perform"
    )
    input_text: Optional[str] = Field(
        default=None, description="Text input for the operation"
    )


class MyTool(BaseTool):
    """Short description of the tool."""

    name: str = "my_tool"
    description: str = (
        "Longer description used by semantic routing when tool.yaml "
        "language-specific description is not available."
    )
    args_schema: type[BaseModel] = MyToolInput

    # Config params injected by registry from tool.yaml / config/tools.yaml
    my_param: str = "default_value"

    def _run(self, operation: str = "do_thing", input_text: Optional[str] = None, **kwargs) -> str:
        """Execute the tool. Always return a string."""
        try:
            if operation == "do_thing":
                return self._do_thing(input_text)
            else:
                return f"Unknown operation: {operation}"
        except Exception as e:
            logger.error("Tool failed", error=str(e), operation=operation)
            return f"Error: {str(e)}"

    async def _arun(self, **kwargs) -> str:
        """Async execution — delegate to sync for simple tools."""
        return self._run(**kwargs)

    def _do_thing(self, input_text: Optional[str]) -> str:
        if not input_text:
            return "Error: input_text is required"
        logger.info("Doing thing", input_length=len(input_text))
        return f"Result: processed {len(input_text)} characters"
```

### Step 4: Write `__init__.py`

```python
"""My custom tool."""
from universal_agentic_framework.tools.my_tool.tool import MyTool
__all__ = ["MyTool"]
```

### Step 5: Register in `config/tools.yaml`

```yaml
tools:
  # ... existing tools ...
  - name: my_tool
    path: universal_agentic_framework/tools/my_tool
    enabled: true
    config:                        # Override tool.yaml defaults
      my_param: "custom_value"
```

---

## Creating an MCP Server Tool

MCP tools are HTTP services that expose a `/execute` endpoint. The framework wraps them with `MCPServerTool`.

### Manifest (`tool.yaml`)

```yaml
name: "my_mcp_tool"
type: "mcp_server"
# No entry_point needed — registry creates MCPServerTool wrapper

descriptions:
  en: "Description for semantic routing"

config_schema:
  server_url:
    type: "string"
    default: "http://localhost:9000"
  default_tool:
    type: "string"
    default: "main_operation"
```

### Server contract

Your MCP server must expose `POST /execute` accepting:

```json
{
  "tool": "main_operation",
  "parameters": { "query": "user text", ... }
}
```

And returning:

```json
{
  "result": "string output for the LLM"
}
```

### Registration

Multiple config entries can point to the same server with different `default_tool` values:

```yaml
tools:
  - name: my_mcp_search
    path: universal_agentic_framework/tools/my_mcp
    enabled: true
    config:
      server_url: $MY_MCP_URL
      default_tool: search

  - name: my_mcp_extract
    path: universal_agentic_framework/tools/my_mcp
    enabled: true
    config:
      server_url: $MY_MCP_URL
      default_tool: extract
```

---

## Tool Selection Architecture

Tools are selected through a **three-tier architecture** that combines semantic scoring with model-driven decisions.

### Layer 1: Semantic Pre-filter (always runs)

1. User query is embedded using the configured embedding model
2. Each tool's description is embedded (cached after first use)
3. Cosine similarity is computed between query and each tool
4. Intent detection boosts scores for matching tools (e.g., datetime patterns boost `datetime_tool` by `intent_boost`)
5. **Gates** filter candidates before they reach the model:
   - `min_top_score` (0.7): If no tool scores above this, skip Layer 2 entirely
   - `min_spread` (0.10): If all scores are within this range (flat distribution), clear all candidates
6. `similarity_threshold` (0.55) and `top_k` (5) further limit candidates
7. Candidates stored in state — **no tools are executed at this layer**

### Layer 2: Model-Driven Tool Calling

The LLM receives candidate tools and decides which to call. Mode is configured per model via `model_tool_calling` in `config/core.yaml`:

| Mode | How it works | Best for |
| ------ | ------------- | ---------- |
| `native` | Tools bound via `bind_tools()`, LLM produces `tool_calls` | Models with proven function calling (GPT-4, LFM2) |
| `structured` | Tool JSON schemas injected into system prompt, LLM outputs JSON | Any model — safe default |
| `react` | ReAct loop (Thought → Action → Observation), max iterations | Weaker models that can't follow JSON schemas |

Runtime safety rule: native mode requires a fresh successful probe; stale/missing/mismatch probe results force structured mode.

### Layer 3: Output Validation + Retry

After parsing the model's tool call:

- Validate tool name exists in candidate set
- Validate arguments against the tool's `args_schema`
- On parse failure: re-prompt with error feedback
- Max `max_retries` (2) attempts before falling back to no-tool response

### Intent boosting

For common patterns, intent detection **boosts** a tool's similarity score (does not force execution):

| Tool | Trigger patterns | Boost |
| ------ | ----------------- | ------- |
| `datetime_tool` | Date patterns (`12.05.2026`), time (`14:30`), keywords (`heute`, `time`, `date`) | +0.2 |
| `calculator_tool` | Math expressions (`2 + 3`), functions (`sqrt(16)`), keywords (`berechne`, `calculate`) | +0.2 |
| `file_ops_tool` | Keywords (`read file`, `list directory`, `datei lesen`) | +0.2 |
| `extract_webpage_mcp` | URL present (`https://...`) | +0.2 |

### Writing good descriptions

**Description quality directly impacts routing accuracy.** Tips:

- Include the user words that should trigger the tool (e.g., "calculate", "berechne", "how much")
- Be specific about capabilities
- Write descriptions in all supported languages (en, de, fr)
- Test with embedding similarity to verify scoring

---

## Security: Sandbox & Rate Limiting

### Tool Sandbox (`tools/sandbox.py`)

The `ToolSandbox` wraps tool execution with:

- **Permission checks**: Tools declare required permissions in `tool.yaml`; sandbox verifies policy allows them
- **Timeout enforcement**: ThreadPoolExecutor-based with configurable per-call timeout
- **Output size limits**: Prevents tools from returning excessively large outputs
- **Concurrent execution cap**: Limits parallel tool calls
- **Path restrictions**: Blocks access to sensitive filesystem paths

```python
from universal_agentic_framework.tools import ToolSandbox, SandboxPolicy, Permission

# Custom policy
policy = SandboxPolicy(
    allowed_permissions={Permission.SYSTEM_TIME.value, Permission.SYSTEM_COMPUTE.value},
    max_execution_time=10.0,
    max_output_size=524288,
    max_concurrent=3,
    denied_paths=["/etc/shadow", "~/.ssh"],
)

sandbox = ToolSandbox(policy=policy)
result = sandbox.execute(
    tool_name="calculator_tool",
    func=calc._run,
    kwargs={"operation": "evaluate", "expression": "2+2"},
    required_permissions=["system:compute"],
)
print(result.success, result.output)
```

### Permission categories

| Permission | Scope |
| ----------- | ------- |
| `system:time` | Date/time operations |
| `system:compute` | CPU-intensive math |
| `filesystem:read` | Read files |
| `filesystem:write` | Write files |
| `network:http` | External HTTP calls |
| `network:internal` | Internal service calls |

### Rate Limiter (`tools/rate_limiter.py`)

Sliding-window rate limiter with three levels:

```python
from universal_agentic_framework.tools import ToolRateLimiter, RateLimitConfig

limiter = ToolRateLimiter(RateLimitConfig(
    global_max_calls=100,         # All tools combined
    global_window_seconds=60,
    per_tool_max_calls=20,        # Per individual tool
    per_tool_window_seconds=60,
    per_user_max_calls=30,        # Per user
    per_user_window_seconds=60,
))

# Before executing a tool:
check = limiter.check("calculator_tool", user_id="user123")
if check.allowed:
    # execute tool
    limiter.record("calculator_tool", user_id="user123")
else:
    print(f"Rate limited: {check.reason}, retry after {check.retry_after_seconds}s")
```

---

## Testing Tools

### Test file structure

Create `tests/test_<tool_name>.py` following the pattern:

```python
import pytest
from universal_agentic_framework.tools.my_tool.tool import MyTool

class TestMyToolOperation:
    @pytest.fixture
    def tool(self):
        return MyTool(my_param="test_value")

    def test_basic_operation(self, tool):
        result = tool._run(operation="do_thing", input_text="hello")
        assert "Result" in result

    def test_error_handling(self, tool):
        result = tool._run(operation="do_thing", input_text=None)
        assert "Error" in result

    def test_unknown_operation(self, tool):
        result = tool._run(operation="nonexistent")
        assert "Unknown" in result or "Error" in result
```

### What to test

| Category | Test |
| ---------- | ------ |
| Happy path | Each operation with valid inputs |
| Edge cases | Empty inputs, None values, boundary values |
| Error handling | Invalid params, resource limits exceeded |
| Security | Path traversal, unsafe inputs, extension filtering |
| Registry | Tool discovered by `ToolRegistry.discover_and_load()` |
| Routing | Heuristic patterns correctly trigger the tool |

### Running tests

```bash
# Single tool tests
poetry run pytest tests/test_tool_ecosystem.py -v

# All tool-related tests
poetry run pytest tests/ -k "tool" -v

# Full test suite
poetry run pytest
```

---

## Standard Tools Reference

Standard tools are production-ready integrations provided by the template for all profiles. They follow five design principles:

1. **Configuration-driven** — all settings via `config/tools.yaml`
2. **Profile-optional** — profiles enable/disable any standard tool
3. **Production-ready** — uses established, maintained services
4. **Secure** — API keys via environment variables, never in code
5. **Testable** — comprehensive tests with mocking support

### Tool Categories

| Category | Tools | Status |
| ---------- | ------- | -------- |
| **Information Retrieval** | Web Search, Knowledge Base (RAG) | Implemented |
| **Utilities** | Date/Time, File Operations, Data Processing, Calculator | Implemented |
| **Communication** | Email, Notifications (Slack, Teams) | Future |
| **Domain-Specific** | Code Execution, API Integrations | Future |

### Reference Implementation: Web Search

The web search tool supports multiple providers via a fallback chain configured in `tool.yaml`:

```yaml
providers:
  - name: "tavily"       # $0.001/call, structured results, LLM-optimized
    requires_env: ["TAVILY_API_KEY"]
  - name: "brave"        # $0.0005/call, privacy-focused
    requires_env: ["BRAVE_API_KEY"]
  - name: "searxng"      # Free, self-hosted, requires extra Docker service
    requires_config: ["searxng_endpoint"]
```

Provider selection is automatic — the registry picks the first enabled provider whose env vars are present.

### Reference Implementation: DateTime

```python
from zoneinfo import ZoneInfo
from datetime import datetime

class DateTimeTool(BaseTool):
    name = "datetime_tool"
    default_timezone: str = "UTC"  # Injected from profile config

    def _run(self, operation="current_time", timezone=None):
        tz = timezone or self.default_timezone
        zone = ZoneInfo(tz)
        now = datetime.now(zone)
        return (
            f"Date: {now.strftime('%A, %B %d, %Y')}\n"
            f"Time: {now.strftime('%H:%M:%S')}\n"
            f"Timezone: {tz}"
        )
```

### Environment Variables

```bash
# Web Search API keys (only needed if provider is enabled)
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxxxxxxx
BRAVE_API_KEY=BSA-xxxxxxxxxxxxxxxxxxxxx
SEARXNG_ENDPOINT=http://searxng:8080  # If self-hosted
```

### Migration Path For Existing Profiles

1. Update the repository or release revision you deploy.
2. Add API keys to `.env` for the providers you enable.
3. Enable or tune tools in `config/tools.yaml` or the active profile overlay.
4. Rebuild services and validate routing behavior.

---

## Quick Reference

### Available tools

| Tool | Type | Category | Operations |
| ------ | ------ | ---------- | ----------- |
| `datetime_tool` | LangChain | Utilities | current_time, convert_timezone |
| `calculator_tool` | LangChain | Utilities | evaluate, convert, statistics, percentage |
| `file_ops_tool` | LangChain | Utilities | read, write, list, info, exists |
| `web_search_mcp` | MCP | Information | web_search ([DuckDuckGo MCP server](https://github.com/nickclyde/duckduckgo-mcp-server)) |
| `extract_webpage_mcp` | MCP | Information | extract_webpage_content |

### Config files

| File | Purpose |
| ------ | --------- |
| `config/tools.yaml` | Enable/disable tools, override config |
| `tools/<name>/tool.yaml` | Tool manifest (description, entry_point, schema) |
| `config/core.yaml` | Tool routing settings (threshold, embedding model) |
| `config/features.yaml` | Feature flags |

### Naming conventions

- LangChain tools: `<name>_tool` (e.g., `calculator_tool`)
- MCP tools: `<name>_mcp` (e.g., `web_search_mcp`)
- Test files: `test_<name>.py` or `test_tool_ecosystem.py` for Week 32 suite
