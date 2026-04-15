"""Tests for Planning Crew implementation and integration."""

import pytest
from unittest.mock import MagicMock, patch

from universal_agentic_framework.config import load_agents_config


class TestPlanningCrewConfigExists:
    """Test that planning crew is properly configured."""
    
    def test_planning_crew_config_exists(self):
        """Verify planning crew is defined in agents.yaml."""
        config = load_agents_config("config")
        assert "planning" in config.crews
        planning_crew = config.crews["planning"]
        assert planning_crew.enabled is True
        assert planning_crew.process == "hierarchical"
        
    def test_planning_crew_has_required_agents(self):
        """Verify planning crew has all three required agents."""
        config = load_agents_config("config")
        planning_crew = config.crews["planning"]
        
        assert "analyst" in planning_crew.agents
        assert "planner" in planning_crew.agents
        assert "reviewer" in planning_crew.agents
        
    def test_planning_agents_have_required_fields(self):
        """Verify each agent has role, goal, backstory, and tools."""
        config = load_agents_config("config")
        planning_crew = config.crews["planning"]
        
        for agent_name in ["analyst", "planner", "reviewer"]:
            agent = planning_crew.agents[agent_name]
            assert agent.role, f"{agent_name} missing role"
            assert agent.goal, f"{agent_name} missing goal"
            assert agent.backstory, f"{agent_name} missing backstory"
            assert agent.tools is not None, f"{agent_name} missing tools list"


class TestPlanningCrewBasics:
    """Test basic planning crew functionality."""
    
    def test_planning_crew_can_be_imported(self):
        """Verify PlanningCrew class can be imported."""
        from universal_agentic_framework.crews import PlanningCrew
        assert PlanningCrew is not None
        
    def test_planning_crew_convenience_function_exists(self):
        """Verify create_planning_crew convenience function exists."""
        from universal_agentic_framework.crews.planning_crew import create_planning_crew
        assert callable(create_planning_crew)


class TestPlanningCrewExecution:
    """Test planning crew execution."""
    
    def test_planning_crew_node_can_be_imported(self):
        """Verify planning crew node function can be imported."""
        from universal_agentic_framework.orchestration.crew_nodes import node_planning_crew
        assert callable(node_planning_crew)


class TestPlanningCrewRouting:
    """Test planning crew routing logic."""

    @pytest.fixture(autouse=True)
    def _enable_multi_agent_crews(self):
        with patch("universal_agentic_framework.orchestration.crew_nodes.load_features_config") as mock_load_features:
            mock_features = MagicMock()
            mock_features.multi_agent_crews = True
            mock_load_features.return_value = mock_features
            yield
    
    def test_routing_detects_plan_keyword(self):
        """Verify routing function detects 'plan' keyword."""
        from universal_agentic_framework.orchestration.crew_nodes import route_to_planning_crew
        from universal_agentic_framework.orchestration.graph_builder import GraphState
        
        state = GraphState(
            messages=[{"role": "user", "content": "Can you plan a migration project?"}],
            language="en",
        )
        
        assert route_to_planning_crew(state) is True
        
    def test_routing_detects_roadmap_keyword(self):
        """Verify routing function detects 'roadmap' keyword."""
        from universal_agentic_framework.orchestration.crew_nodes import route_to_planning_crew
        from universal_agentic_framework.orchestration.graph_builder import GraphState
        
        state = GraphState(
            messages=[{"role": "user", "content": "Create a roadmap for Q3 features."}],
            language="en",
        )
        
        assert route_to_planning_crew(state) is True
        
    def test_routing_detects_sprint_keyword(self):
        """Verify routing function detects 'sprint' keyword."""
        from universal_agentic_framework.orchestration.crew_nodes import route_to_planning_crew
        from universal_agentic_framework.orchestration.graph_builder import GraphState
        
        state = GraphState(
            messages=[{"role": "user", "content": "Help with sprint planning for the next release."}],
            language="en",
        )
        
        assert route_to_planning_crew(state) is True
        
    def test_routing_detects_german_keywords(self):
        """Verify routing function detects German planning keywords."""
        from universal_agentic_framework.orchestration.crew_nodes import route_to_planning_crew
        from universal_agentic_framework.orchestration.graph_builder import GraphState
        
        state_plan = GraphState(
            messages=[{"role": "user", "content": "Bitte erstelle einen Projektplan."}],
            language="de",
        )
        assert route_to_planning_crew(state_plan) is True
        
        state_milestone = GraphState(
            messages=[{"role": "user", "content": "Definiere Meilensteine fuer das Projekt."}],
            language="de",
        )
        assert route_to_planning_crew(state_milestone) is True
        
    def test_routing_detects_pattern_matching(self):
        """Verify routing function detects planning patterns."""
        from universal_agentic_framework.orchestration.crew_nodes import route_to_planning_crew
        from universal_agentic_framework.orchestration.graph_builder import GraphState
        
        state_breakdown = GraphState(
            messages=[{"role": "user", "content": "Break down this initiative into tasks."}],
            language="en",
        )
        assert route_to_planning_crew(state_breakdown) is True
        
        state_estimate = GraphState(
            messages=[{"role": "user", "content": "Estimate the effort for this project."}],
            language="en",
        )
        assert route_to_planning_crew(state_estimate) is True
        
    def test_routing_rejects_non_planning_queries(self):
        """Verify routing function rejects non-planning queries."""
        from universal_agentic_framework.orchestration.crew_nodes import route_to_planning_crew
        from universal_agentic_framework.orchestration.graph_builder import GraphState
        
        state = GraphState(
            messages=[{"role": "user", "content": "What is the weather today?"}],
            language="en",
        )
        
        assert route_to_planning_crew(state) is False
        
    def test_routing_handles_empty_messages(self):
        """Verify routing function handles empty message list gracefully."""
        from universal_agentic_framework.orchestration.crew_nodes import route_to_planning_crew
        from universal_agentic_framework.orchestration.graph_builder import GraphState
        
        state = GraphState(messages=[], language="en")
        assert route_to_planning_crew(state) is False
        
    def test_routing_handles_missing_messages_key(self):
        """Verify routing function handles missing messages key gracefully."""
        from universal_agentic_framework.orchestration.crew_nodes import route_to_planning_crew
        
        state = {"language": "en"}
        assert route_to_planning_crew(state) is False
        
    def test_routing_detects_dependency_keyword(self):
        """Verify routing function detects 'dependency' keyword."""
        from universal_agentic_framework.orchestration.crew_nodes import route_to_planning_crew
        from universal_agentic_framework.orchestration.graph_builder import GraphState
        
        state = GraphState(
            messages=[{"role": "user", "content": "Identify dependencies between tasks."}],
            language="en",
        )
        
        assert route_to_planning_crew(state) is True
        
    def test_routing_detects_milestone_keyword(self):
        """Verify routing function detects 'milestone' keyword."""
        from universal_agentic_framework.orchestration.crew_nodes import route_to_planning_crew
        from universal_agentic_framework.orchestration.graph_builder import GraphState
        
        state = GraphState(
            messages=[{"role": "user", "content": "Define key milestones for the rollout."}],
            language="en",
        )
        
        assert route_to_planning_crew(state) is True


class TestPlanningCrewFeatureFlag:
    """Test feature-flag contract for planning routing."""

    @patch("universal_agentic_framework.orchestration.crew_nodes.load_features_config")
    def test_routing_disabled_when_feature_flag_off(self, mock_load_features):
        """Should not route when multi_agent_crews is disabled, even for planning queries."""
        from universal_agentic_framework.orchestration.crew_nodes import route_to_planning_crew
        from universal_agentic_framework.orchestration.graph_builder import GraphState

        mock_features = MagicMock()
        mock_features.multi_agent_crews = False
        mock_load_features.return_value = mock_features

        state = GraphState(
            messages=[{"role": "user", "content": "Can you plan a migration project?"}],
            language="en",
        )

        assert route_to_planning_crew(state) is False


class TestNodePlanningCrew:
    """Test planning crew node behavior."""
    
    def test_node_handles_empty_messages(self):
        """Verify node handles empty message list gracefully."""
        from universal_agentic_framework.orchestration.crew_nodes import node_planning_crew
        
        state = {"messages": [], "language": "en"}
        result_state = node_planning_crew(state)
        
        # Should return state unchanged
        assert result_state is state


class TestGraphIntegration:
    """Test planning crew integration with graph builder."""

    @pytest.fixture(autouse=True)
    def _enable_multi_agent_crews(self):
        with patch("universal_agentic_framework.orchestration.crew_nodes.load_features_config") as mock_load_features:
            mock_features = MagicMock()
            mock_features.multi_agent_crews = True
            mock_load_features.return_value = mock_features
            yield
    
    def test_router_function_includes_planning(self):
        """Verify multi-crew router function includes planning crew."""
        from universal_agentic_framework.orchestration.graph_builder import build_graph
        
        # build_graph should not raise errors
        graph = build_graph()
        assert graph is not None
        
    def test_planning_crew_results_field_in_state(self):
        """Verify crew_results field exists in GraphState for planning."""
        from universal_agentic_framework.orchestration.graph_builder import GraphState
        
        state = GraphState(
            messages=[],
            language="en",
            crew_results={"planning": {"success": True}},
        )
        
        assert "crew_results" in state
        assert "planning" in state["crew_results"]
        
    def test_multi_crew_routing_priority_order(self):
        """Verify crew routing priority: research > analytics > code generation > planning."""
        from universal_agentic_framework.orchestration.crew_nodes import (
            route_to_research_crew,
            route_to_analytics_crew,
            route_to_code_generation_crew,
            route_to_planning_crew,
        )
        from universal_agentic_framework.orchestration.graph_builder import GraphState
        
        # Pure planning query (no research/analytics/code keywords)
        state_planning = GraphState(
            messages=[{"role": "user", "content": "Need a project roadmap with milestones."}],
            language="en",
        )
        assert route_to_research_crew(state_planning) is False
        assert route_to_analytics_crew(state_planning) is False
        assert route_to_code_generation_crew(state_planning) is False
        assert route_to_planning_crew(state_planning) is True
        
        # Query with both planning and research keywords
        state_mixed = GraphState(
            messages=[{"role": "user", "content": "Research best practices and create a project plan."}],
            language="en",
        )
        # Research should match first (priority)
        assert route_to_research_crew(state_mixed) is True
        # Planning would also match
        assert route_to_planning_crew(state_mixed) is True


class TestPlanningCrewProcessType:
    """Test planning crew uses hierarchical process."""
    
    def test_planning_crew_uses_hierarchical_process(self):
        """Verify planning crew is configured for hierarchical process."""
        config = load_agents_config("config")
        planning_crew = config.crews["planning"]
        
        assert planning_crew.process == "hierarchical"
        assert planning_crew.max_iterations >= 10
        
    def test_planning_crew_timeout_is_configured(self):
        """Verify planning crew has appropriate timeout."""
        config = load_agents_config("config")
        planning_crew = config.crews["planning"]
        
        # Planning tasks are longer, should have higher timeout
        assert planning_crew.timeout_seconds >= 300


class TestPlanningCrewTools:
    """Test planning crew tool assignments."""
    
    def test_analyst_has_rag_tool(self):
        """Verify analyst has rag_retrieval tool."""
        config = load_agents_config("config")
        planning_crew = config.crews["planning"]
        analyst = planning_crew.agents["analyst"]
        
        assert "rag_retrieval" in analyst.tools
        
    def test_planner_has_rag_tool(self):
        """Verify planner has rag_retrieval tool."""
        config = load_agents_config("config")
        planning_crew = config.crews["planning"]
        planner = planning_crew.agents["planner"]
        
        assert "rag_retrieval" in planner.tools
        
    def test_reviewer_has_rag_tool(self):
        """Verify reviewer has rag_retrieval tool."""
        config = load_agents_config("config")
        planning_crew = config.crews["planning"]
        reviewer = planning_crew.agents["reviewer"]
        
        assert "rag_retrieval" in reviewer.tools


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
