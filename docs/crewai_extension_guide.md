# CrewAI Extension Guide: Adding New Crews to the Framework

**Purpose**: Quick reference for adding new crew types and understanding CrewAI integration  
**Audience**: Developers implementing custom crews for domain-specific profiles  
**Status**: Advanced and experimental; use for deliberate extensions rather than baseline setup

---

## Current Framework Crews

The framework includes 4 specialized crews:

- **Research Crew**: Information gathering and synthesis
- **Analytics Crew**: Data analysis and insights
- **Code Generation Crew**: Code implementation and scaffolding
- **Planning Crew**: Project planning and task breakdown

All crews are configured in `config/agents.yaml` and integrated into the LangGraph orchestration pipeline.

---

## Quick Start: Three-Step Process

### Step 1: Add Configuration to `config/agents.yaml`

```yaml
crews:
  my_crew:
    enabled: true
    process: sequential # or hierarchical
    max_iterations: 10
    timeout_seconds: 300
    agents:
      agent_1:
        role: "Descriptive role"
        goal: "What they're trying to accomplish"
        backstory: |
          Multi-line backstory describing...
        tools:
          - tool_name_1
          - tool_name_2
      agent_2:
        role: "..."
        goal: "..."
        backstory: "..."
        tools: []
```

### Step 2: Create Crew Class in `universal_agentic_framework/crews/my_crew.py`

```python
from typing import Dict, Any
from universal_agentic_framework.config.loaders import load_agents_config, load_core_config
from universal_agentic_framework.llm.factory import LLMFactory
from universal_agentic_framework.tools.registry import ToolRegistry
from crew_ai import Agent, Task, Crew

class MyCrew:
    def __init__(self, language: str = "en", config_dir: str = "config",
                 model_override=None, tool_registry: ToolRegistry = None):
        self.language = language
        self.config = load_agents_config(config_dir)
        self.core_config = load_core_config(config_dir)
        self.llm = model_override or LLMFactory.get_llm(self.language, self.core_config)
        self.tool_registry = tool_registry or ToolRegistry()
        self._agents = None
        self._tasks = None
        self._crew = None

    def _get_tools_for_agent(self, agent_name: str):
        """Load tools from registry for agent"""
        config = self.config.get("crews", {}).get("my_crew", {})
        agent_config = config.get("agents", {}).get(agent_name, {})
        tool_names = agent_config.get("tools", [])

        tools = []
        for tool_name in tool_names:
            tool = self.tool_registry.get(tool_name)
            if tool:
                tools.append(tool)
        return tools

    def _build_agents(self):
        """Initialize crew agents from config"""
        config = self.config.get("crews", {}).get("my_crew", {})
        agents_config = config.get("agents", {})

        self._agents = {
            agent_name: Agent(
                role=agent_cfg["role"],
                goal=agent_cfg["goal"],
                backstory=agent_cfg["backstory"],
                llm=self.llm,
                tools=self._get_tools_for_agent(agent_name),
                verbose=True,
            )
            for agent_name, agent_cfg in agents_config.items()
        }
        return self._agents

    def _build_tasks(self):
        """Initialize crew tasks"""
        self._agents = self._agents or self._build_agents()

        self._tasks = [
            Task(
                description="Task 1 description...",
                agent=self._agents["agent_1"],
                expected_output="Expected output format...",
            ),
            Task(
                description="Task 2 description...",
                agent=self._agents["agent_2"],
                expected_output="Expected output format...",
            ),
        ]
        return self._tasks

    def _build_crew(self):
        """Initialize CrewAI crew"""
        tasks = self._tasks or self._build_tasks()
        config = self.config.get("crews", {}).get("my_crew", {})

        self._crew = Crew(
            agents=list(self._agents.values()),
            tasks=tasks,
            process=config.get("process", "sequential"),
            max_iterations=config.get("max_iterations", 10),
            memory=True,
            verbose=True,
        )
        return self._crew

    def kickoff(self, **kwargs) -> Dict[str, Any]:
        """Execute crew workflow"""
        try:
            crew = self._build_crew()
            result = crew.kickoff(inputs=kwargs)
            return {
                "success": True,
                "result": result,
                "steps": len(self._tasks),
                "error": None,
            }
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "steps": 0,
                "error": str(e),
            }

def create_my_crew(language: str = "en", **kwargs) -> Dict[str, Any]:
    """Convenience function for one-shot crew execution"""
    crew = MyCrew(language=language)
    return crew.kickoff(**kwargs)
```

### Step 3: Add Graph Node in `universal_agentic_framework/orchestration/crew_nodes.py`

```python
from universal_agentic_framework.crews.my_crew import MyCrew

def route_to_my_crew(state: GraphState) -> bool:
    """Determine if query should route to my crew"""
    if not state.get("messages"):
        return False

    last_message = state["messages"][-1].content.lower()

    # Keywords that trigger this crew
    keywords = ["keyword1", "keyword2", "keyword3"]

    return any(kw in last_message for kw in keywords)

def node_my_crew(state: GraphState) -> GraphState:
    """Execute my crew for specific query types"""
    try:
        last_message = state["messages"][-1].content

        crew = MyCrew(
            language=state.get("language", "en"),
            config_dir="config",
        )

        result = crew.kickoff(query=last_message)

        state["crew_results"]["my_crew"] = result

        if result["success"]:
            state["messages"].append(
                Message(role="assistant", content=result["result"])
            )
        else:
            state["messages"].append(
                Message(role="assistant", content=f"Crew execution failed: {result['error']}")
            )

        return state

    except Exception as e:
        state["crew_results"]["my_crew"] = {"error": str(e)}
        return state
```

### Step 4: Integrate into Graph

```python
# In universal_agentic_framework/orchestration/graph_builder.py

from universal_agentic_framework.orchestration.crew_nodes import (
    node_research_crew, route_to_research_crew,
    node_my_crew, route_to_my_crew,  # Add your new crew
)

def build_graph(...):
    # Existing code...

    # Add your crew node
    graph.add_node("my_crew", node_my_crew)

    # Change from single route to multi-level routing
    graph.add_conditional_edges(
        "route_tools",
        lambda state: (
            "research_crew" if route_to_research_crew(state) else
            "my_crew" if route_to_my_crew(state) else
            "memory_query_cache"
        ),
        {
            "research_crew": "research_crew",
            "my_crew": "my_crew",
            "memory_query_cache": "memory_query_cache",
        }
    )

    # Add edges from your crew to standard flow
    graph.add_edge("research_crew", "memory_query_cache")
    graph.add_edge("my_crew", "memory_query_cache")

    return graph
```

---

## Configuration Templates

See the existing crew definitions in `config/agents.yaml` for complete examples. The essential structure:

```yaml
crews:
  my_crew:
    enabled: true
    process: sequential # or hierarchical
    max_iterations: 10
    timeout_seconds: 300
    agents:
      agent_name:
        role: "Descriptive role"
        goal: "What they're trying to accomplish"
        backstory: "Context that shapes the agent's reasoning"
        tools:
          - tool_name
```

Use `process: hierarchical` when workers can run concurrently under a manager agent; use `sequential` when each step depends on the prior output.

---

## Tool Integration Patterns

### Pattern 1: Tool Functions from LangChain

```python
from langchain_core.tools import tool
from universal_agentic_framework.tools.registry import ToolRegistry

@tool("my_tool")
def my_tool(input_str: str) -> str:
    """Tool description for LLM"""
    return f"Result: {input_str}"

# Register in tools.yaml
tools:
  - name: my_tool
    type: langchain_function
    enabled: true
```

### Pattern 2: Tool Classes with Setup

```python
class MyCustomTool:
    def __init__(self, config):
        self.config = config
        self.setup()

    def setup(self):
        """Initialize tool resources"""
        pass

    def __call__(self, **kwargs):
        """Execute tool"""
        pass

# Instantiate in crew initialization
from universal_agentic_framework.tools.loaders import load_tool

tool_instance = load_tool("my_custom_tool", config_dir="config")
tools.append(tool_instance)
```

### Pattern 3: External API Tools via MCP

```yaml
tools:
  - name: api_tool
    type: mcp_server
    enabled: true
    config:
      server_url: http://localhost:8002
      timeout: 30
      server_name: "my_api_server"
```

---

## Testing Your New Crew

Add a test file at `tests/test_my_crew.py`. Essential assertions:

```python
from unittest.mock import MagicMock, patch

from universal_agentic_framework.crews.my_crew import MyCrew


def test_crew_config_exists():
    from universal_agentic_framework.config.loaders import load_agents_config

    config = load_agents_config("config")
    assert "my_crew" in config.get("crews", {})


def test_crew_initialization():
    crew = MyCrew(language="en")
    assert crew.language == "en"


@patch("universal_agentic_framework.llm.factory.LLMFactory.get_llm")
def test_crew_kickoff(mock_llm):
    mock_llm.return_value = MagicMock()
    crew = MyCrew(language="en", model_override=mock_llm.return_value)
    with patch.object(crew, "_build_crew") as mock_build:
        mock_build.return_value = MagicMock(kickoff=MagicMock(return_value="ok"))
        result = crew.kickoff(query="test")
    assert result["success"]
```

Also test routing (verify `True`/`False` for matching and non-matching inputs) and confirm your node appears in `graph.nodes`. See `tests/test_analytics_crew.py` for a complete crew test reference.

---

## Debugging & Troubleshooting

### Common Issues

**Issue**: Tools not found in crew

```python
# ❌ Wrong
tools = ["web_search_mcp"]  # Just string names

# ✅ Correct
tools = self._get_tools_for_agent("agent_name")  # Loads from registry
```

**Issue**: Agent initialization fails

```python
# ❌ Wrong - Missing required fields
Agent(role="Analyst", llm=llm)  # Missing goal, backstory

# ✅ Correct - All fields required
Agent(
    role="Data Analyst",
    goal="Analyze data patterns",
    backstory="Expert with 10+ years...",
    llm=llm,
    tools=tools
)
```

**Issue**: Route function returns wrong type

```python
# ❌ Wrong - Returns string
def route_func(state):
    return "my_crew"  # Should be bool for conditional_edges

# ✅ Correct - Returns bool
def route_func(state):
    return should_use_crew(state)  # Returns True/False
```

### Debug Commands

## Performance Optimization

Use `process: hierarchical` when tasks can run in parallel under a manager; use `sequential` when each step depends on the prior. Hierarchical reduces wall-clock time at the cost of extra manager tokens.

### Tool Call Optimization

```python
# Limit tools per agent
agent1.tools = ["web_search_mcp"]  # Focused tools = faster
agent2.tools = ["rag_retrieval", "python_repl"]  # More flexible

# Avoid circular tool dependencies
agent1 → agent2 ✅ (linear)
agent1 → agent2 → agent1 ❌ (avoid loops)
```

---

## Monitoring & Observability

Use `structlog.get_logger()` to emit events at key lifecycle points (`crew_kickoff`, task start/end). Instrument crew execution with `track_node_execution`:

```python
from universal_agentic_framework.monitoring.metrics import track_node_execution

with track_node_execution("my_crew", "my_crew"):
    result = crew.kickoff(...)
```

See [monitoring.md](monitoring.md) for the full Prometheus metrics reference.

---

## Checklist for New Crews

- [ ] Configuration added to agents.yaml with at least 2 agents
- [ ] Crew class created in universal_agentic_framework/crews/
- [ ] Routing function added to crew_nodes.py with keyword matching
- [ ] Crew node execution function added to crew_nodes.py
- [ ] Graph integration updated with conditional edge
- [ ] At least 5 tests created (config, routing, execution, graph, edge cases)
- [ ] All tests passing (run `pytest tests/test_my_crew.py`)
- [ ] Documentation added to docs/ folder
- [ ] Error handling in place (crew failures don't crash graph)
- [ ] Logging added at key points
- [ ] Tools configured in tools.yaml if using external services

---

## Additional Resources

- **Configuration Schema**: See [configuration.md](configuration.md) for complete `agents.yaml` schema
- **Tool Integration**: See [tool_development_guide.md](tool_development_guide.md) for tool system patterns
- **Testing**: See [technical_architecture.md](technical_architecture.md) for testing best practices
- **Monitoring**: See [monitoring.md](monitoring.md) for crew performance metrics
