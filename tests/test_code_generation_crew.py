"""Tests for Code Generation Crew implementation and integration."""

import pytest
from unittest.mock import MagicMock, patch

from universal_agentic_framework.config import load_agents_config


class TestCodeGenerationCrewConfigExists:
    """Test that code generation crew is properly configured."""
    
    def test_code_generation_crew_config_exists(self):
        """Verify code generation crew is defined in agents.yaml."""
        config = load_agents_config("config")
        assert "code_generation" in config.crews
        code_crew = config.crews["code_generation"]
        assert code_crew.enabled is True
        assert code_crew.process == "sequential"
        
    def test_code_generation_crew_has_required_agents(self):
        """Verify code generation crew has all three required agents."""
        config = load_agents_config("config")
        code_crew = config.crews["code_generation"]
        
        assert "architect" in code_crew.agents
        assert "developer" in code_crew.agents
        assert "qa_engineer" in code_crew.agents
        
    def test_code_generation_agents_have_required_fields(self):
        """Verify each agent has role, goal, backstory, and tools."""
        config = load_agents_config("config")
        code_crew = config.crews["code_generation"]
        
        for agent_name in ["architect", "developer", "qa_engineer"]:
            agent = code_crew.agents[agent_name]
            assert agent.role, f"{agent_name} missing role"
            assert agent.goal, f"{agent_name} missing goal"
            assert agent.backstory, f"{agent_name} missing backstory"
            assert agent.tools is not None, f"{agent_name} missing tools list"


class TestCodeGenerationCrewBasics:
    """Test basic code generation crew functionality."""
    
    def test_code_generation_crew_can_be_imported(self):
        """Verify CodeGenerationCrew class can be imported."""
        from universal_agentic_framework.crews import CodeGenerationCrew
        assert CodeGenerationCrew is not None
        
    def test_code_generation_crew_convenience_function_exists(self):
        """Verify create_code_generation_crew convenience function exists."""
        from universal_agentic_framework.crews.code_generation_crew import create_code_generation_crew
        assert callable(create_code_generation_crew)


class TestCodeGenerationCrewExecution:
    """Test code generation crew execution."""
    
    def test_code_generation_crew_node_can_be_imported(self):
        """Verify code generation crew node function can be imported."""
        from universal_agentic_framework.orchestration.crew_nodes import node_code_generation_crew
        assert callable(node_code_generation_crew)


class TestCodeGenerationCrewRouting:
    """Test code generation crew routing logic."""

    @pytest.fixture(autouse=True)
    def _enable_multi_agent_crews(self):
        with patch("universal_agentic_framework.orchestration.crew_nodes.load_features_config") as mock_load_features:
            mock_features = MagicMock()
            mock_features.multi_agent_crews = True
            mock_load_features.return_value = mock_features
            yield
    
    def test_routing_detects_code_keyword(self):
        """Verify routing function detects 'code' keyword."""
        from universal_agentic_framework.orchestration.crew_nodes import route_to_code_generation_crew
        from universal_agentic_framework.orchestration.graph_builder import GraphState
        
        state = GraphState(
            messages=[{"role": "user", "content": "Write code to calculate fibonacci numbers."}],
            language="en",
        )
        
        assert route_to_code_generation_crew(state) is True
        
    def test_routing_detects_implement_keyword(self):
        """Verify routing function detects 'implement' keyword."""
        from universal_agentic_framework.orchestration.crew_nodes import route_to_code_generation_crew
        from universal_agentic_framework.orchestration.graph_builder import GraphState
        
        state = GraphState(
            messages=[{"role": "user", "content": "Implement a binary search function in Python."}],
            language="en",
        )
        
        assert route_to_code_generation_crew(state) is True
        
    def test_routing_detects_function_keyword(self):
        """Verify routing function detects 'function' keyword."""
        from universal_agentic_framework.orchestration.crew_nodes import route_to_code_generation_crew
        from universal_agentic_framework.orchestration.graph_builder import GraphState
        
        state = GraphState(
            messages=[{"role": "user", "content": "Create a function to parse JSON data."}],
            language="en",
        )
        
        assert route_to_code_generation_crew(state) is True
        
    def test_routing_detects_class_keyword(self):
        """Verify routing function detects 'class' keyword."""
        from universal_agentic_framework.orchestration.crew_nodes import route_to_code_generation_crew
        from universal_agentic_framework.orchestration.graph_builder import GraphState
        
        state = GraphState(
            messages=[{"role": "user", "content": "Build a class to manage user authentication."}],
            language="en",
        )
        
        assert route_to_code_generation_crew(state) is True
        
    def test_routing_detects_api_keyword(self):
        """Verify routing function detects 'api' keyword."""
        from universal_agentic_framework.orchestration.crew_nodes import route_to_code_generation_crew
        from universal_agentic_framework.orchestration.graph_builder import GraphState
        
        state = GraphState(
            messages=[{"role": "user", "content": "Write a REST API for user management."}],
            language="en",
        )
        
        assert route_to_code_generation_crew(state) is True
        
    def test_routing_detects_german_keywords(self):
        """Verify routing function detects German code generation keywords."""
        from universal_agentic_framework.orchestration.crew_nodes import route_to_code_generation_crew
        from universal_agentic_framework.orchestration.graph_builder import GraphState
        
        state_implement = GraphState(
            messages=[{"role": "user", "content": "Implementiere eine Funktion zur Sortierung."}],
            language="de",
        )
        assert route_to_code_generation_crew(state_implement) is True
        
        state_code = GraphState(
            messages=[{"role": "user", "content": "Schreibe Code für eine API."}],
            language="de",
        )
        assert route_to_code_generation_crew(state_code) is True
        
    def test_routing_detects_pattern_matching(self):
        """Verify routing function detects code generation patterns."""
        from universal_agentic_framework.orchestration.crew_nodes import route_to_code_generation_crew
        from universal_agentic_framework.orchestration.graph_builder import GraphState
        
        state_write = GraphState(
            messages=[{"role": "user", "content": "Can you write a function that validates email addresses?"}],
            language="en",
        )
        assert route_to_code_generation_crew(state_write) is True
        
        state_create = GraphState(
            messages=[{"role": "user", "content": "I need to create a class for data validation."}],
            language="en",
        )
        assert route_to_code_generation_crew(state_create) is True
        
        state_build = GraphState(
            messages=[{"role": "user", "content": "Build an API endpoint for user registration."}],
            language="en",
        )
        assert route_to_code_generation_crew(state_build) is True
        
    def test_routing_rejects_non_code_queries(self):
        """Verify routing function rejects non-code-generation queries."""
        from universal_agentic_framework.orchestration.crew_nodes import route_to_code_generation_crew
        from universal_agentic_framework.orchestration.graph_builder import GraphState
        
        state_greeting = GraphState(
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            language="en",
        )
        assert route_to_code_generation_crew(state_greeting) is False
        
        state_weather = GraphState(
            messages=[{"role": "user", "content": "What is the weather today?"}],
            language="en",
        )
        assert route_to_code_generation_crew(state_weather) is False
        
    def test_routing_handles_empty_messages(self):
        """Verify routing function handles empty message list gracefully."""
        from universal_agentic_framework.orchestration.crew_nodes import route_to_code_generation_crew
        from universal_agentic_framework.orchestration.graph_builder import GraphState
        
        state = GraphState(messages=[], language="en")
        assert route_to_code_generation_crew(state) is False
        
    def test_routing_handles_missing_messages_key(self):
        """Verify routing function handles missing messages key gracefully."""
        from universal_agentic_framework.orchestration.crew_nodes import route_to_code_generation_crew
        
        state = {"language": "en"}
        assert route_to_code_generation_crew(state) is False
        
    def test_routing_detects_test_keyword(self):
        """Verify routing function detects 'test' keyword for unit test requests."""
        from universal_agentic_framework.orchestration.crew_nodes import route_to_code_generation_crew
        from universal_agentic_framework.orchestration.graph_builder import GraphState
        
        state = GraphState(
            messages=[{"role": "user", "content": "Write unit tests for the user service class."}],
            language="en",
        )
        
        assert route_to_code_generation_crew(state) is True
        
    def test_routing_detects_refactor_keyword(self):
        """Verify routing function detects 'refactor' keyword."""
        from universal_agentic_framework.orchestration.crew_nodes import route_to_code_generation_crew
        from universal_agentic_framework.orchestration.graph_builder import GraphState
        
        state = GraphState(
            messages=[{"role": "user", "content": "Refactor this code to improve performance."}],
            language="en",
        )
        
        assert route_to_code_generation_crew(state) is True
        
    def test_routing_detects_algorithm_keyword(self):
        """Verify routing function detects 'algorithm' keyword."""
        from universal_agentic_framework.orchestration.crew_nodes import route_to_code_generation_crew
        from universal_agentic_framework.orchestration.graph_builder import GraphState
        
        state = GraphState(
            messages=[{"role": "user", "content": "Implement a sorting algorithm."}],
            language="en",
        )
        
        assert route_to_code_generation_crew(state) is True


class TestCodeGenerationCrewFeatureFlag:
    """Test feature-flag contract for code generation routing."""

    @patch("universal_agentic_framework.orchestration.crew_nodes.load_features_config")
    def test_routing_disabled_when_feature_flag_off(self, mock_load_features):
        """Should not route when multi_agent_crews is disabled, even for code queries."""
        from universal_agentic_framework.orchestration.crew_nodes import route_to_code_generation_crew
        from universal_agentic_framework.orchestration.graph_builder import GraphState

        mock_features = MagicMock()
        mock_features.multi_agent_crews = False
        mock_load_features.return_value = mock_features

        state = GraphState(
            messages=[{"role": "user", "content": "Write code to calculate fibonacci numbers."}],
            language="en",
        )

        assert route_to_code_generation_crew(state) is False


class TestNodeCodeGenerationCrew:
    """Test code generation crew node behavior."""
    
    def test_node_handles_empty_messages(self):
        """Verify node handles empty message list gracefully."""
        from universal_agentic_framework.orchestration.crew_nodes import node_code_generation_crew
        
        state = {"messages": [], "language": "en"}
        result_state = node_code_generation_crew(state)
        
        # Should return state unchanged
        assert result_state is state


class TestGraphIntegration:
    """Test code generation crew integration with graph builder."""

    @pytest.fixture(autouse=True)
    def _enable_multi_agent_crews(self):
        with patch("universal_agentic_framework.orchestration.crew_nodes.load_features_config") as mock_load_features:
            mock_features = MagicMock()
            mock_features.multi_agent_crews = True
            mock_load_features.return_value = mock_features
            yield
    
    def test_router_function_includes_code_generation(self):
        """Verify multi-crew router includes code generation crew."""
        from universal_agentic_framework.orchestration.graph_builder import build_graph
        
        # build_graph should not raise errors
        graph = build_graph()
        assert graph is not None
        
    def test_code_generation_crew_results_field_in_state(self):
        """Verify crew_results field exists in GraphState for code generation."""
        from universal_agentic_framework.orchestration.graph_builder import GraphState
        
        state = GraphState(
            messages=[],
            language="en",
            crew_results={"code_generation": {"success": True}},
        )
        
        assert "crew_results" in state
        assert "code_generation" in state["crew_results"]
        
    def test_multi_crew_routing_priority_order(self):
        """Verify crew routing priority: research > analytics > code generation."""
        from universal_agentic_framework.orchestration.crew_nodes import (
            route_to_research_crew,
            route_to_analytics_crew,
            route_to_code_generation_crew,
        )
        from universal_agentic_framework.orchestration.graph_builder import GraphState
        
        # Pure code generation query (no research/analytics keywords)
        state_code = GraphState(
            messages=[{"role": "user", "content": "Write a function to reverse a string."}],
            language="en",
        )
        assert route_to_research_crew(state_code) is False
        assert route_to_analytics_crew(state_code) is False
        assert route_to_code_generation_crew(state_code) is True
        
        # Query with both code and research keywords
        state_mixed = GraphState(
            messages=[{"role": "user", "content": "Search for code examples to implement quicksort."}],
            language="en",
        )
        # Research should match first (priority)
        assert route_to_research_crew(state_mixed) is True
        # Code generation would also match
        assert route_to_code_generation_crew(state_mixed) is True


class TestCodeGenerationCrewProcessType:
    """Test code generation crew uses sequential process."""
    
    def test_code_generation_crew_uses_sequential_process(self):
        """Verify code generation crew is configured for sequential process."""
        config = load_agents_config("config")
        code_crew = config.crews["code_generation"]
        
        assert code_crew.process == "sequential"
        assert code_crew.max_iterations >= 10
        
    def test_code_generation_crew_timeout_is_configured(self):
        """Verify code generation crew has appropriate timeout."""
        config = load_agents_config("config")
        code_crew = config.crews["code_generation"]
        
        # Code generation tasks need reasonable timeout (between research and analytics)
        assert code_crew.timeout_seconds >= 300


class TestCodeGenerationCrewTools:
    """Test code generation crew tool configuration."""
    
    def test_architect_has_rag_tool(self):
        """Verify architect agent has RAG retrieval tool for patterns/best practices."""
        config = load_agents_config("config")
        code_crew = config.crews["code_generation"]
        architect = code_crew.agents["architect"]
        
        assert "rag_retrieval" in architect.tools
        
    def test_developer_has_python_repl(self):
        """Verify developer agent has python_repl for testing code."""
        config = load_agents_config("config")
        code_crew = config.crews["code_generation"]
        developer = code_crew.agents["developer"]
        
        assert "python_repl" in developer.tools
        
    def test_qa_engineer_has_both_tools(self):
        """Verify QA engineer has both python_repl and rag_retrieval."""
        config = load_agents_config("config")
        code_crew = config.crews["code_generation"]
        qa_engineer = code_crew.agents["qa_engineer"]
        
        assert "python_repl" in qa_engineer.tools
        assert "rag_retrieval" in qa_engineer.tools


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
