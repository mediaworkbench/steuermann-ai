"""Planning Crew: Multi-agent workflow for project planning and task decomposition.

Hierarchical workflow with three agents coordinated by a manager:
1. Analyst: Analyzes requirements, identifies stakeholders and objectives
2. Planner: Breaks down projects into tasks, identifies dependencies
3. Reviewer: Reviews plans for completeness, identifies risks and gaps

Inherits from BaseCrew for retry, timeout, and validation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from crewai import Agent, Crew, Task
from crewai.process import Process

from universal_agentic_framework.crews.base import BaseCrew
from universal_agentic_framework.monitoring.logging import get_logger

logger = get_logger(__name__)


class PlanningCrew(BaseCrew):
    """Multi-agent crew for project planning, task decomposition, and risk assessment.

    Uses three agents under a hierarchical manager:
    - Analyst: Analyzes requirements and clarifies objectives
    - Planner: Creates structured execution plans with task breakdown
    - Reviewer: Validates plans and identifies risks

    Configured via ``config/agents.yaml`` → ``crews.planning``.
    """

    crew_name = "planning"

    # ------------------------------------------------------------------
    # Build methods
    # ------------------------------------------------------------------

    def _build_agents(self) -> None:
        crew_config = self.agents_config.crews.get(self.crew_name)
        if not crew_config:
            raise ValueError(f"'{self.crew_name}' crew not configured in config/agents.yaml")

        llm = self._get_llm()

        for agent_key, defaults in [
            ("analyst", ("Business Analyst", "Analyze requirements and clarify objectives", "You are a business analyst")),
            ("planner", ("Project Planner", "Create structured execution plans", "You are a project planner")),
            ("reviewer", ("Planning Reviewer", "Review plans and identify risks", "You are a planning reviewer")),
        ]:
            agent_def = crew_config.agents.get(agent_key)
            if not agent_def:
                raise ValueError(f"'{agent_key}' agent not configured for {self.crew_name} crew")

            self._agents[agent_key] = Agent(
                role=agent_def.role or defaults[0],
                goal=agent_def.goal or defaults[1],
                backstory=agent_def.backstory or defaults[2],
                tools=self._get_tools_for_agent(agent_key),
                llm=llm,
                verbose=True,
                allow_delegation=False,
            )

        logger.info("Planning crew agents initialized", agents=list(self._agents.keys()))

    def _build_tasks(self) -> None:
        self._tasks["analysis"] = Task(
            description=(
                "Analyze the following project or initiative: {project_description}\n\n"
                "1. Understand the business objective and desired outcomes\n"
                "2. Identify key stakeholders and their concerns\n"
                "3. Clarify scope, boundaries, and constraints\n"
                "4. Extract functional and non-functional requirements\n"
                "5. Identify success criteria and KPIs\n"
                "6. Note any assumptions or open questions"
            ),
            agent=self._agents["analyst"],
            expected_output=(
                "Requirements analysis document with clear objectives, stakeholder map, "
                "detailed requirements, constraints, success criteria, and initial risk assessment."
            ),
        )

        self._tasks["planning"] = Task(
            description=(
                "Based on the requirements analysis, create a detailed execution plan.\n\n"
                "1. Break down the project into phases, epics, and actionable tasks\n"
                "2. Identify dependencies between tasks\n"
                "3. Estimate effort for each task\n"
                "4. Define acceptance criteria for each major deliverable\n"
                "5. Identify critical path and potential bottlenecks\n"
                "6. Suggest resource allocation and team structure\n"
                "7. Define milestones and checkpoints"
            ),
            agent=self._agents["planner"],
            expected_output=(
                "Detailed execution plan with task breakdown, dependencies, estimates, "
                "critical path, milestones, and recommended execution sequence."
            ),
        )

        self._tasks["review"] = Task(
            description=(
                "Review the execution plan for completeness, feasibility, and risks.\n\n"
                "1. Validate completeness (are all requirements covered?)\n"
                "2. Assess realism of estimates\n"
                "3. Identify missing tasks or overlooked dependencies\n"
                "4. Conduct risk assessment (technical, resource, timeline)\n"
                "5. Recommend risk mitigation strategies\n"
                "6. Provide overall feasibility assessment"
            ),
            agent=self._agents["reviewer"],
            expected_output=(
                "Comprehensive review report with completeness assessment, risk register, "
                "mitigation strategies, improvement suggestions, and feasibility rating."
            ),
        )
        logger.info("Built planning crew tasks", count=len(self._tasks))

    def _build_crew(self) -> None:
        crew_config = self.agents_config.crews.get(self.crew_name)
        if not crew_config:
            raise ValueError(f"'{self.crew_name}' crew not configured")

        manager_llm = self._get_manager_llm()

        self._crew = Crew(
            agents=list(self._agents.values()),
            tasks=list(self._tasks.values()),
            process=Process.hierarchical,
            manager_llm=manager_llm,
            max_iter=crew_config.max_iterations or 15,
            memory=True,
            verbose=True,
        )
        logger.info("Planning crew initialized (hierarchical)")

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _kickoff(self, **kwargs) -> Dict[str, Any]:
        """Execute the crew. Expects ``project_description`` kwarg, optional ``context``."""
        project_description = kwargs.get("project_description", "")
        context = kwargs.get("context")
        if not self._crew:
            return {"success": False, "error": "Crew not properly initialized"}

        logger.info("Planning crew starting", project=project_description[:100])
        inputs: Dict[str, Any] = {"project_description": project_description}
        if context:
            inputs.update(context)

        result = self._crew.kickoff(inputs=inputs)
        logger.info("Planning crew completed")

        return {
            "success": True,
            "result": str(result),
            "steps": self._extract_task_results(),
        }


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def create_planning_crew(
    project_description: str,
    language: str = "en",
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create and execute planning crew in one call."""
    crew = PlanningCrew(language=language)
    result = crew.kickoff_with_retry(project_description=project_description, context=context)
    return result.model_dump()
