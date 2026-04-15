"""Analytics Crew: Multi-agent workflow for data analysis and reporting.

Hierarchical workflow with three agents coordinated by a manager:
1. Data Analyst: Explores data, identifies patterns and trends
2. Statistician: Provides statistical validation and hypothesis testing
3. Report Writer: Creates business reports with actionable insights

Inherits from BaseCrew for retry, timeout, and validation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from crewai import Agent, Crew, Task
from crewai.process import Process

from universal_agentic_framework.crews.base import BaseCrew
from universal_agentic_framework.monitoring.logging import get_logger

logger = get_logger(__name__)


class AnalyticsCrew(BaseCrew):
    """Multi-agent crew for data analysis, statistical validation, and reporting.

    Uses three agents under a hierarchical manager:
    - Data Analyst: Performs exploratory analysis and pattern recognition
    - Statistician: Validates findings with rigorous statistical methods
    - Report Writer: Transforms analysis into actionable business reports

    Configured via ``config/agents.yaml`` → ``crews.analytics``.
    """

    crew_name = "analytics"

    # ------------------------------------------------------------------
    # Build methods
    # ------------------------------------------------------------------

    def _build_agents(self) -> None:
        crew_config = self.agents_config.crews.get(self.crew_name)
        if not crew_config:
            raise ValueError(f"'{self.crew_name}' crew not configured in config/agents.yaml")

        llm = self._get_llm()

        for agent_key, defaults in [
            ("data_analyst", ("Data Analyst", "Analyze data and identify patterns", "You are a data analyst")),
            ("statistician", ("Statistician", "Provide statistical validation", "You are a statistician")),
            ("report_writer", ("Report Writer", "Create business reports", "You are a report writer")),
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

        logger.info("Analytics crew agents initialized", agents=list(self._agents.keys()))

    def _build_tasks(self) -> None:
        self._tasks["analysis"] = Task(
            description=(
                "Perform comprehensive data analysis on the provided dataset or query: {query}\n\n"
                "Your responsibilities:\n"
                "1. Understand the analytical objective and data characteristics\n"
                "2. Perform exploratory data analysis (EDA)\n"
                "3. Identify key patterns, trends, correlations, and anomalies\n"
                "4. Document preliminary findings with statistical summaries\n"
                "5. Prepare insights for statistical validation"
            ),
            agent=self._agents["data_analyst"],
            expected_output=(
                "Comprehensive data analysis report with statistical summaries, "
                "identified patterns, trends, and preliminary insights ready for validation."
            ),
        )

        self._tasks["validation"] = Task(
            description=(
                "Provide rigorous statistical validation of the analytical findings.\n\n"
                "1. Review the data analyst's findings and hypotheses\n"
                "2. Select and perform appropriate statistical tests\n"
                "3. Calculate confidence intervals and p-values\n"
                "4. Validate or refute preliminary hypotheses\n"
                "5. Assess statistical significance of patterns"
            ),
            agent=self._agents["statistician"],
            expected_output=(
                "Statistical validation report with test results, significance assessments, "
                "and confirmation of analytical findings with appropriate confidence levels."
            ),
        )

        self._tasks["reporting"] = Task(
            description=(
                "Transform the analytical and statistical findings into a clear, "
                "actionable business report.\n\n"
                "1. Synthesize findings from data analysis and statistical validation\n"
                "2. Create executive summary with key takeaways\n"
                "3. Translate technical findings into business language\n"
                "4. Provide actionable recommendations"
            ),
            agent=self._agents["report_writer"],
            expected_output=(
                "Professional business report with executive summary, key insights, "
                "actionable recommendations, and clear communication of findings."
            ),
        )
        logger.info("Built analytics crew tasks", count=len(self._tasks))

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
        logger.info("Analytics crew initialized (hierarchical)")

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _kickoff(self, **kwargs) -> Dict[str, Any]:
        """Execute the crew. Expects ``query`` kwarg, optional ``context``."""
        query = kwargs.get("query", "")
        context = kwargs.get("context")
        if not self._crew:
            return {"success": False, "error": "Crew not properly initialized"}

        logger.info("Analytics crew starting", query=query[:100])
        inputs: Dict[str, Any] = {"query": query}
        if context:
            inputs.update(context)

        result = self._crew.kickoff(inputs=inputs)
        logger.info("Analytics crew completed")

        return {
            "success": True,
            "result": str(result),
            "steps": self._extract_task_results(),
        }


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def create_analytics_crew(
    query: str,
    language: str = "en",
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create and execute analytics crew in one call."""
    crew = AnalyticsCrew(language=language)
    result = crew.kickoff_with_retry(query=query, context=context)
    return result.model_dump()
