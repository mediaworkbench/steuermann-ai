"""Research Crew: Multi-agent workflow for web research and analysis.

This crew implements a sequential workflow with three agents:
1. Searcher: Finds and retrieves web research
2. Analyst: Synthesizes findings into insights
3. Writer: Creates structured reports

Inherits from BaseCrew for retry, timeout, and validation.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, Optional

from crewai import Agent, Crew, Task
from crewai.process import Process
from langchain_core.language_models import BaseChatModel

from universal_agentic_framework.crews.base import BaseCrew, CrewResult
from universal_agentic_framework.monitoring.logging import get_logger

logger = get_logger(__name__)


class ResearchCrew(BaseCrew):
    """Multi-agent crew for research, analysis, and report writing.

    Agents (sequential):
    - Searcher: Performs web research using semantic search and RAG
    - Analyst: Synthesizes findings and extracts key insights
    - Writer: Produces structured, well-organized reports

    Configured via ``config/agents.yaml`` → ``crews.research``.
    """

    crew_name = "research"

    # ------------------------------------------------------------------
    # Build methods (called by BaseCrew.__init__)
    # ------------------------------------------------------------------

    def _build_agents(self) -> None:
        crew_config = self.agents_config.crews.get(self.crew_name)
        if not crew_config:
            raise ValueError(f"'{self.crew_name}' crew not configured in config/agents.yaml")

        llm = self._get_llm()

        for agent_key, defaults in [
            ("searcher", ("Searcher", "Find relevant information", "You are a research assistant")),
            ("analyst", ("Analyst", "Analyze information", "You are an analyst")),
            ("writer", ("Writer", "Write reports", "You are a technical writer")),
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

        logger.info("Research crew agents initialized", agents=list(self._agents.keys()))

    def _build_tasks(self) -> None:
        self._tasks["research"] = Task(
            description=(
                "Today's date is {current_date}. "
                "Search the web comprehensively for the most recent information about {topic}. "
                "Prioritise results from {current_year} and avoid presenting outdated content as current. "
                "Use semantic search and RAG retrieval to find relevant, authoritative sources. "
                "Return a detailed research summary with key findings from multiple sources."
            ),
            expected_output=(
                "A comprehensive research summary including:\n"
                "- Key findings from web sources\n"
                "- Relevant quotes and citations\n"
                "- Source information (titles, URLs)\n"
                "- Data points and statistics\n"
                "- Current trends or recent developments"
            ),
            agent=self._agents["searcher"],
        )

        self._tasks["analysis"] = Task(
            description=(
                "Analyze the research findings from the first task. "
                "Synthesize the information into key insights, identify patterns, "
                "extract important facts, and organize them logically. "
                "Focus on what's most relevant and important about {topic}."
            ),
            expected_output=(
                "A structured analysis including:\n"
                "- Key insights and takeaways\n"
                "- Important patterns and trends\n"
                "- Relationships between findings\n"
                "- Critical facts and statistics\n"
                "- Gaps in information (if any)\n"
                "- Recommendations for further research (if applicable)"
            ),
            agent=self._agents["analyst"],
        )

        self._tasks["writing"] = Task(
            description=(
                "Create a well-structured, professional research report based on "
                "the analysis provided. The report should be clear, comprehensive, "
                "and suitable for sharing with stakeholders. Include proper formatting, "
                "citations, and section organization."
            ),
            expected_output=(
                "A professional research report with:\n"
                "- Executive summary\n"
                "- Introduction and background\n"
                "- Key findings (organized by theme)\n"
                "- Analysis and insights\n"
                "- Conclusions\n"
                "- References/sources (with URLs where applicable)\n"
                "- Appendices (if needed)"
            ),
            agent=self._agents["writer"],
        )

    def _build_crew(self) -> None:
        crew_config = self.agents_config.crews.get(self.crew_name)
        if not crew_config:
            raise ValueError(f"'{self.crew_name}' crew not configured")

        process_map = {
            "sequential": Process.sequential,
            "hierarchical": Process.hierarchical,
        }
        process = process_map.get(crew_config.process or "sequential", Process.sequential)

        self._crew = Crew(
            agents=list(self._agents.values()),
            tasks=list(self._tasks.values()),
            process=process,
            max_iter=crew_config.max_iterations or 10,
            verbose=True,
        )
        logger.info(
            "Research crew initialized",
            process=crew_config.process or "sequential",
            max_iterations=crew_config.max_iterations or 10,
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _kickoff(self, **kwargs) -> Dict[str, Any]:
        """Execute the crew. Expects ``topic`` kwarg."""
        topic = kwargs.get("topic", "")
        if not self._crew:
            return {"success": False, "error": "Crew not properly initialized"}

        today = datetime.date.today()
        current_date = today.strftime("%B %d, %Y")
        current_year = str(today.year)
        logger.info("Research crew starting", topic=topic, current_date=current_date)
        result = self._crew.kickoff(inputs={"topic": topic, "current_date": current_date, "current_year": current_year})
        logger.info("Research crew completed", topic=topic)

        return {
            "success": True,
            "result": str(result),
            "steps": self._extract_task_results(),
        }


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def create_research_crew(
    topic: str,
    language: str = "en",
    config_dir: Optional[str] = None,
    model_override: Optional[BaseChatModel] = None,
) -> Dict[str, Any]:
    """Create and run a research crew in one call (returns plain dict)."""
    crew = ResearchCrew(language=language, config_dir=config_dir, model_override=model_override)
    result = crew.kickoff_with_retry(topic=topic)
    return result.model_dump()
