"""Integration tests for Research Crew orchestration.

Tests cover:
1. Research crew creation and configuration via config files
2. Agent collaboration and task flow
3. Tool integration (web search, RAG) configuration
4. Graph routing to research crew
5. State management across crew execution

Note: Full CrewAI execution tests are integration tests that require proper
configuration and LLM setup, so we focus on routing logic and node behavior.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from universal_agentic_framework.orchestration.crew_nodes import (
    route_to_research_crew,
    node_research_crew,
)
from universal_agentic_framework.orchestration.graph_builder import GraphState

class TestResearchCrewConfigExists:
    """Test that research crew configuration is defined."""

    def test_research_crew_config_defined_in_agents_yaml(self):
        """Research crew should be configured in config/agents.yaml."""
        from universal_agentic_framework.config import load_agents_config
        from pathlib import Path
        
        config = load_agents_config(config_dir=Path("config"))
        
        assert "research" in config.crews
        crew_config = config.crews["research"]
        
        # Check agents are configured
        assert "searcher" in crew_config.agents
        assert "analyst" in crew_config.agents
        assert "writer" in crew_config.agents

    def test_research_crew_agents_have_required_fields(self):
        """Research crew agents should have role, goal, and backstory."""
        from universal_agentic_framework.config import load_agents_config
        from pathlib import Path
        
        config = load_agents_config(config_dir=Path("config"))
        crew_config = config.crews["research"]
        
        for agent_name in ["searcher", "analyst", "writer"]:
            agent = crew_config.agents[agent_name]
            assert agent.role is not None
            assert agent.goal is not None
            assert agent.backstory is not None
            assert len(agent.role) > 0
            assert len(agent.goal) > 0
            assert len(agent.backstory) > 0

    def test_research_crew_agents_have_correct_tools(self):
        """Research crew agents should be configured with appropriate tools."""
        from universal_agentic_framework.config import load_agents_config
        from pathlib import Path
        
        config = load_agents_config(config_dir=Path("config"))
        crew_config = config.crews["research"]
        
        searcher = crew_config.agents["searcher"]
        analyst = crew_config.agents["analyst"]
        writer = crew_config.agents["writer"]
        
        # Searcher should have web search and RAG tools
        assert searcher.tools is not None
        assert "web_search_mcp" in searcher.tools
        assert "rag_retrieval" in searcher.tools
        
        # Analyst should have RAG retrieval
        assert analyst.tools is not None
        assert "rag_retrieval" in analyst.tools
        
        # Writer should have no tools
        assert writer.tools is None or len(writer.tools) == 0


class TestResearchCrewBasics:
    """Test Research Crew can be imported and instantiated."""

    def test_research_crew_can_be_imported(self):
        """Research crew module should be importable."""
        from universal_agentic_framework.crews import ResearchCrew
        
        assert ResearchCrew is not None

    def test_create_research_crew_convenience_function_exists(self):
        """Convenience function should exist for one-shot crew usage."""
        from universal_agentic_framework.crews.research_crew import create_research_crew
        
        assert callable(create_research_crew)


class TestResearchCrewExecution:
    """Test Research Crew routing setup in graph."""

    def test_research_crew_node_imported(self):
        """Research crew node should be properly imported."""
        from universal_agentic_framework.orchestration.crew_nodes import node_research_crew
        
        assert callable(node_research_crew)


class TestResearchCrewRouting:
    """Test routing logic for research queries."""

    @pytest.fixture(autouse=True)
    def _enable_multi_agent_crews(self):
        with patch("universal_agentic_framework.orchestration.crew_nodes.load_features_config") as mock_load_features:
            mock_features = MagicMock()
            mock_features.multi_agent_crews = True
            mock_load_features.return_value = mock_features
            yield

    def test_route_to_research_crew_detects_search_keyword(self):
        """Should route to research crew when 'search' keyword is present."""
        state = {
            "messages": [
                {"role": "user", "content": "search the web for climate change information"}
            ]
        }
        
        assert route_to_research_crew(state) is True

    def test_route_to_research_crew_detects_research_keyword(self):
        """Should route to research crew when 'research' keyword is present."""
        state = {
            "messages": [
                {"role": "user", "content": "please research the latest AI developments"}
            ]
        }
        
        assert route_to_research_crew(state) is True

    def test_route_to_research_crew_detects_find_keyword(self):
        """Should route to research crew when 'find' keyword is present."""
        state = {
            "messages": [
                {"role": "user", "content": "find information about renewable energy"}
            ]
        }
        
        assert route_to_research_crew(state) is True

    def test_route_to_research_crew_detects_german_keywords(self):
        """Should route to research crew for German research keywords."""
        state = {
            "messages": [
                {"role": "user", "content": "Suche nach aktuellen Nachrichten"}
            ]
        }
        
        assert route_to_research_crew(state) is True

    def test_route_to_research_crew_detects_question_pattern(self):
        """Should route to research crew for typical research question patterns."""
        state = {
            "messages": [
                {"role": "user", "content": "what is quantum computing?"}
            ]
        }
        
        assert route_to_research_crew(state) is True

    def test_route_to_research_crew_rejects_non_research_query(self):
        """Should NOT route to research crew for non-research queries."""
        state = {
            "messages": [
                {"role": "user", "content": "hello, how are you?"}
            ]
        }
        
        assert route_to_research_crew(state) is False

    def test_route_to_research_crew_empty_messages(self):
        """Should handle empty messages gracefully."""
        state = {"messages": []}
        
        assert route_to_research_crew(state) is False

    def test_route_to_research_crew_no_messages_key(self):
        """Should handle missing messages key gracefully."""
        state = {}
        
        assert route_to_research_crew(state) is False


class TestResearchCrewFeatureFlag:
    """Test feature-flag contract for research routing."""

    @patch("universal_agentic_framework.orchestration.crew_nodes.load_features_config")
    def test_route_to_research_crew_disabled_when_feature_flag_off(self, mock_load_features):
        """Should not route when multi_agent_crews is disabled, even for research queries."""
        mock_features = MagicMock()
        mock_features.multi_agent_crews = False
        mock_load_features.return_value = mock_features

        state = {
            "messages": [
                {"role": "user", "content": "search the web for climate change information"}
            ]
        }

        assert route_to_research_crew(state) is False


class TestNodeResearchCrew:
    """Test research_crew_node integration with graph."""

    def test_node_research_crew_handles_empty_messages(self):
        """Node should handle empty message lists gracefully."""
        with patch("universal_agentic_framework.orchestration.crew_nodes.load_core_config"):
            state = {"messages": []}
            
            result_state = node_research_crew(state)
            
            # Should return state as-is
            assert result_state is not None

class TestGraphIntegration:
    """Test research crew integration with LangGraph."""

    @pytest.fixture(autouse=True)
    def _enable_multi_agent_crews(self):
        with patch("universal_agentic_framework.orchestration.crew_nodes.load_features_config") as mock_load_features:
            mock_features = MagicMock()
            mock_features.multi_agent_crews = True
            mock_load_features.return_value = mock_features
            yield

    def test_router_function_can_be_imported_from_graph(self):
        """Router should be properly imported in graph builder."""
        from universal_agentic_framework.orchestration.graph_builder import build_graph
        
        assert callable(build_graph)

    def test_crew_node_in_graph_state(self):
        """Graph state should include crew_results field."""
        from universal_agentic_framework.orchestration.graph_builder import GraphState
        
        # Check that crew_results is in GraphState annotations
        assert "crew_results" in GraphState.__annotations__
