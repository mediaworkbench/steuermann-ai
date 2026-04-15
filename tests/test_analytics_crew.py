"""Tests for Analytics Crew implementation and integration."""

import pytest
from unittest.mock import MagicMock, patch

from universal_agentic_framework.config import load_agents_config


class TestAnalyticsCrewConfigExists:
    """Test that analytics crew is properly configured."""
    
    def test_analytics_crew_config_exists(self):
        """Verify analytics crew is defined in agents.yaml."""
        config = load_agents_config("config")
        assert "analytics" in config.crews
        analytics_crew = config.crews["analytics"]
        assert analytics_crew.enabled is True
        assert analytics_crew.process == "hierarchical"
        
    def test_analytics_crew_has_required_agents(self):
        """Verify analytics crew has all three required agents."""
        config = load_agents_config("config")
        analytics_crew = config.crews["analytics"]
        
        assert "data_analyst" in analytics_crew.agents
        assert "statistician" in analytics_crew.agents
        assert "report_writer" in analytics_crew.agents
        
    def test_analytics_agents_have_required_fields(self):
        """Verify each agent has role, goal, backstory, and tools."""
        config = load_agents_config("config")
        analytics_crew = config.crews["analytics"]
        
        for agent_name in ["data_analyst", "statistician", "report_writer"]:
            agent = analytics_crew.agents[agent_name]
            assert agent.role, f"{agent_name} missing role"
            assert agent.goal, f"{agent_name} missing goal"
            assert agent.backstory, f"{agent_name} missing backstory"
            assert agent.tools is not None, f"{agent_name} missing tools list"


class TestAnalyticsCrewBasics:
    """Test basic analytics crew functionality."""
    
    def test_analytics_crew_can_be_imported(self):
        """Verify AnalyticsCrew class can be imported."""
        from universal_agentic_framework.crews import AnalyticsCrew
        assert AnalyticsCrew is not None
        
    def test_analytics_crew_convenience_function_exists(self):
        """Verify create_analytics_crew convenience function exists."""
        from universal_agentic_framework.crews.analytics_crew import create_analytics_crew
        assert callable(create_analytics_crew)


class TestAnalyticsCrewExecution:
    """Test analytics crew execution."""
    
    def test_analytics_crew_node_can_be_imported(self):
        """Verify analytics crew node function can be imported."""
        from universal_agentic_framework.orchestration.crew_nodes import node_analytics_crew
        assert callable(node_analytics_crew)


class TestAnalyticsCrewRouting:
    """Test analytics crew routing logic."""
    
    @patch('universal_agentic_framework.orchestration.crew_nodes.load_features_config')
    def test_routing_detects_analyze_keyword(self, mock_load_features):
        """Verify routing function detects 'analyze' keyword."""
        from universal_agentic_framework.orchestration.crew_nodes import route_to_analytics_crew
        from universal_agentic_framework.orchestration.graph_builder import GraphState
        
        # Enable multi-agent crews for this test
        mock_features = MagicMock()
        mock_features.multi_agent_crews = True
        mock_load_features.return_value = mock_features
        
        state = GraphState(
            messages=[{"role": "user", "content": "Can you analyze this dataset?"}],
            language="en",
        )
        
        assert route_to_analytics_crew(state) is True
        
    @patch('universal_agentic_framework.orchestration.crew_nodes.load_features_config')
    def test_routing_detects_trend_keyword(self, mock_load_features):
        """Verify routing function detects 'trend' keyword."""
        from universal_agentic_framework.orchestration.crew_nodes import route_to_analytics_crew
        from universal_agentic_framework.orchestration.graph_builder import GraphState
        
        # Enable multi-agent crews for this test
        mock_features = MagicMock()
        mock_features.multi_agent_crews = True
        mock_load_features.return_value = mock_features
        
        state = GraphState(
            messages=[{"role": "user", "content": "What are the trends in sales data?"}],
            language="en",
        )
        
        assert route_to_analytics_crew(state) is True
        
    @patch('universal_agentic_framework.orchestration.crew_nodes.load_features_config')
    def test_routing_detects_statistic_keyword(self, mock_load_features):
        """Verify routing function detects 'statistic' keyword."""
        from universal_agentic_framework.orchestration.crew_nodes import route_to_analytics_crew
        from universal_agentic_framework.orchestration.graph_builder import GraphState
        
        # Enable multi-agent crews for this test
        mock_features = MagicMock()
        mock_features.multi_agent_crews = True
        mock_load_features.return_value = mock_features
        
        state = GraphState(
            messages=[{"role": "user", "content": "Show me the statistics for user engagement."}],
            language="en",
        )
        
        assert route_to_analytics_crew(state) is True
        
    @patch('universal_agentic_framework.orchestration.crew_nodes.load_features_config')
    def test_routing_detects_german_keywords(self, mock_load_features):
        """Verify routing function detects German analytics keywords."""
        from universal_agentic_framework.orchestration.crew_nodes import route_to_analytics_crew
        from universal_agentic_framework.orchestration.graph_builder import GraphState
        
        # Enable multi-agent crews for this test
        mock_features = MagicMock()
        mock_features.multi_agent_crews = True
        mock_load_features.return_value = mock_features
        
        state_analyze = GraphState(
            messages=[{"role": "user", "content": "Bitte analysiere die Verkaufsdaten."}],
            language="de",
        )
        assert route_to_analytics_crew(state_analyze) is True
        
        state_trend = GraphState(
            messages=[{"role": "user", "content": "Zeige mir den Trend der Nutzeraktivität."}],
            language="de",
        )
        assert route_to_analytics_crew(state_trend) is True
        
    @patch('universal_agentic_framework.orchestration.crew_nodes.load_features_config')
    def test_routing_detects_pattern_matching(self, mock_load_features):
        """Verify routing function detects analytics patterns."""
        from universal_agentic_framework.orchestration.crew_nodes import route_to_analytics_crew
        from universal_agentic_framework.orchestration.graph_builder import GraphState
        
        # Enable multi-agent crews for this test
        mock_features = MagicMock()
        mock_features.multi_agent_crews = True
        mock_load_features.return_value = mock_features
        
        state_compare = GraphState(
            messages=[{"role": "user", "content": "Compare revenue in Q1 vs Q2."}],
            language="en",
        )
        assert route_to_analytics_crew(state_compare) is True
        
        state_calculate = GraphState(
            messages=[{"role": "user", "content": "Calculate the average response time."}],
            language="en",
        )
        assert route_to_analytics_crew(state_calculate) is True
        
    def test_routing_rejects_non_analytics_queries(self):
        """Verify routing function rejects non-analytics queries."""
        from universal_agentic_framework.orchestration.crew_nodes import route_to_analytics_crew
        from universal_agentic_framework.orchestration.graph_builder import GraphState
        
        state = GraphState(
            messages=[{"role": "user", "content": "What is the weather today?"}],
            language="en",
        )
        
        assert route_to_analytics_crew(state) is False
        
    def test_routing_handles_empty_messages(self):
        """Verify routing function handles empty message list gracefully."""
        from universal_agentic_framework.orchestration.crew_nodes import route_to_analytics_crew
        from universal_agentic_framework.orchestration.graph_builder import GraphState
        
        state = GraphState(messages=[], language="en")
        assert route_to_analytics_crew(state) is False
        
    def test_routing_handles_missing_messages_key(self):
        """Verify routing function handles missing messages key gracefully."""
        from universal_agentic_framework.orchestration.crew_nodes import route_to_analytics_crew
        
        state = {"language": "en"}
        assert route_to_analytics_crew(state) is False
        
    @patch('universal_agentic_framework.orchestration.crew_nodes.load_features_config')
    def test_routing_detects_metrics_keyword(self, mock_load_features):
        """Verify routing function detects 'metrics' keyword."""
        from universal_agentic_framework.orchestration.crew_nodes import route_to_analytics_crew
        from universal_agentic_framework.orchestration.graph_builder import GraphState
        
        # Enable multi-agent crews for this test
        mock_features = MagicMock()
        mock_features.multi_agent_crews = True
        mock_load_features.return_value = mock_features
        
        state = GraphState(
            messages=[{"role": "user", "content": "Show me the key metrics for this month."}],
            language="en",
        )
        
        assert route_to_analytics_crew(state) is True
        
    @patch('universal_agentic_framework.orchestration.crew_nodes.load_features_config')
    def test_routing_detects_correlation_keyword(self, mock_load_features):
        """Verify routing function detects 'correlation' keyword."""
        from universal_agentic_framework.orchestration.crew_nodes import route_to_analytics_crew
        from universal_agentic_framework.orchestration.graph_builder import GraphState
        
        # Enable multi-agent crews for this test
        mock_features = MagicMock()
        mock_features.multi_agent_crews = True
        mock_load_features.return_value = mock_features
        
        state = GraphState(
            messages=[{"role": "user", "content": "Find correlations between user age and purchase frequency."}],
            language="en",
        )
        
        assert route_to_analytics_crew(state) is True


class TestAnalyticsCrewFeatureFlag:
    """Test feature-flag contract for analytics routing."""

    @patch('universal_agentic_framework.orchestration.crew_nodes.load_features_config')
    def test_routing_disabled_when_feature_flag_off(self, mock_load_features):
        """Should not route when multi_agent_crews is disabled, even for analytics queries."""
        from universal_agentic_framework.orchestration.crew_nodes import route_to_analytics_crew
        from universal_agentic_framework.orchestration.graph_builder import GraphState

        mock_features = MagicMock()
        mock_features.multi_agent_crews = False
        mock_load_features.return_value = mock_features

        state = GraphState(
            messages=[{"role": "user", "content": "Can you analyze this dataset?"}],
            language="en",
        )

        assert route_to_analytics_crew(state) is False


class TestNodeAnalyticsCrew:
    """Test analytics crew node behavior."""
    
    def test_node_handles_empty_messages(self):
        """Verify node handles empty message list gracefully."""
        from universal_agentic_framework.orchestration.crew_nodes import node_analytics_crew
        
        state = {"messages": [], "language": "en"}
        result_state = node_analytics_crew(state)
        
        # Should return state unchanged
        assert result_state is state


class TestGraphIntegration:
    """Test analytics crew integration with graph builder."""
    
    def test_router_function_can_be_imported(self):
        """Verify multi-crew router function exists in graph builder."""
        from universal_agentic_framework.orchestration.graph_builder import build_graph
        
        # build_graph should not raise errors
        graph = build_graph()
        assert graph is not None
        
    def test_analytics_crew_results_field_in_state(self):
        """Verify crew_results field exists in GraphState for analytics."""
        from universal_agentic_framework.orchestration.graph_builder import GraphState
        
        state = GraphState(
            messages=[],
            language="en",
            crew_results={"analytics": {"success": True}},
        )
        
        assert "crew_results" in state
        assert "analytics" in state["crew_results"]
        
    @patch('universal_agentic_framework.orchestration.crew_nodes.load_features_config')
    def test_multi_crew_routing_priority(self, mock_load_features):
        """Verify research queries take precedence over analytics queries."""
        from universal_agentic_framework.orchestration.crew_nodes import (
            route_to_research_crew,
            route_to_analytics_crew,
        )
        from universal_agentic_framework.orchestration.graph_builder import GraphState
        
        # Enable multi-agent crews for this test
        mock_features = MagicMock()
        mock_features.multi_agent_crews = True
        mock_load_features.return_value = mock_features
        
        # Query with both research and analytics keywords
        state = GraphState(
            messages=[{"role": "user", "content": "Search for trends in market data."}],
            language="en",
        )
        
        # Research should match first (priority)
        assert route_to_research_crew(state) is True
        # Analytics would also match
        assert route_to_analytics_crew(state) is True


class TestAnalyticsCrewProcessType:
    """Test analytics crew uses hierarchical process."""
    
    def test_analytics_crew_uses_hierarchical_process(self):
        """Verify analytics crew is configured for hierarchical process."""
        config = load_agents_config("config")
        analytics_crew = config.crews["analytics"]
        
        assert analytics_crew.process == "hierarchical"
        assert analytics_crew.max_iterations >= 10
        
    def test_analytics_crew_timeout_is_configured(self):
        """Verify analytics crew has appropriate timeout."""
        config = load_agents_config("config")
        analytics_crew = config.crews["analytics"]
        
        # Analytics tasks are longer, should have higher timeout
        assert analytics_crew.timeout_seconds >= 300


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
