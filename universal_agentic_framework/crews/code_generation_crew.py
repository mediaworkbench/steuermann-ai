"""Code Generation Crew: Multi-agent workflow for code design, implementation, and testing.

Sequential workflow with three agents:
1. Architect: Analyzes requirements and designs technical solution
2. Developer: Implements code based on specifications
3. QA Engineer: Reviews code quality and writes tests

Inherits from BaseCrew for retry, timeout, and validation.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from crewai import Agent, Crew, Task
from crewai.process import Process
from langchain_core.language_models import BaseChatModel

from universal_agentic_framework.crews.base import BaseCrew
from universal_agentic_framework.monitoring.logging import get_logger

logger = get_logger(__name__)


class CodeGenerationCrew(BaseCrew):
    """Multi-agent crew for code generation, implementation, and QA.

    Agents (sequential):
    - Architect: Designs technical solutions and creates specifications
    - Developer: Implements code following specifications and best practices
    - QA Engineer: Reviews code quality and writes comprehensive tests

    Configured via ``config/agents.yaml`` → ``crews.code_generation``.
    """

    crew_name = "code_generation"

    # ------------------------------------------------------------------
    # Build methods
    # ------------------------------------------------------------------

    def _build_agents(self) -> None:
        crew_config = self.agents_config.crews.get(self.crew_name)
        if not crew_config:
            raise ValueError(f"'{self.crew_name}' crew not configured in config/agents.yaml")

        llm = self._get_llm()

        for agent_key, defaults in [
            ("architect", ("Software Architect", "Design technical solutions", "You are a software architect")),
            ("developer", ("Software Developer", "Implement code", "You are a software developer")),
            ("qa_engineer", ("QA Engineer", "Validate code quality", "You are a QA engineer")),
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

        logger.info("Code generation crew agents initialized", agents=list(self._agents.keys()))

    def _build_tasks(self) -> None:
        self._tasks["design"] = Task(
            description=(
                "Analyze the following requirement: {requirement}\n\n"
                "Design a comprehensive technical solution including:\n"
                "- Overall architecture and design patterns\n"
                "- Detailed API specifications (if applicable)\n"
                "- Data structures and models\n"
                "- Function/class signatures and responsibilities\n"
                "- Error handling strategy\n"
                "- Edge cases to consider\n"
                "- Implementation notes and best practices"
            ),
            expected_output=(
                "A comprehensive technical specification document including:\n"
                "- Architecture diagram (text description)\n"
                "- Detailed module/class breakdown\n"
                "- Function signatures with parameters and return types\n"
                "- Data structure definitions\n"
                "- Error handling approach\n"
                "- Implementation guidelines"
            ),
            agent=self._agents["architect"],
        )

        self._tasks["implementation"] = Task(
            description=(
                "Based on the technical specification from the architect, implement high-quality code.\n\n"
                "Follow these guidelines:\n"
                "- Write clean, readable, well-commented code\n"
                "- Follow SOLID principles and design patterns specified\n"
                "- Implement comprehensive error handling\n"
                "- Add docstrings for classes and functions\n"
                "- Consider edge cases and input validation\n"
                "- Use appropriate data structures and algorithms"
            ),
            expected_output=(
                "Complete, production-ready code including:\n"
                "- All required imports\n"
                "- Well-structured classes/functions\n"
                "- Comprehensive docstrings\n"
                "- Input validation and error handling\n"
                "- Type hints\n"
                "- Example usage or main function"
            ),
            agent=self._agents["developer"],
        )

        self._tasks["qa_testing"] = Task(
            description=(
                "Review the implemented code and ensure production readiness.\n\n"
                "1. Code Review: Check quality, bugs, edge cases, error handling\n"
                "2. Test Generation: Write comprehensive unit tests\n"
                "3. Validation: Verify implementation matches specifications"
            ),
            expected_output=(
                "A comprehensive QA report including:\n"
                "- Code review findings\n"
                "- Complete test suite code\n"
                "- Security and performance notes\n"
                "- Final recommendation (ready / needs revision)"
            ),
            agent=self._agents["qa_engineer"],
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
            max_iter=crew_config.max_iterations or 12,
            verbose=True,
        )
        logger.info(
            "Code generation crew initialized",
            process=crew_config.process or "sequential",
            max_iterations=crew_config.max_iterations or 12,
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _kickoff(self, **kwargs) -> Dict[str, Any]:
        """Execute the crew. Expects ``requirement`` kwarg, optional ``context``."""
        requirement = kwargs.get("requirement", "")
        context = kwargs.get("context")
        if not self._crew:
            return {"success": False, "error": "Crew not properly initialized"}

        inputs: Dict[str, Any] = {"requirement": requirement}
        if context:
            inputs["context"] = json.dumps(context, indent=2)

        logger.info("Code generation crew starting", requirement=requirement[:100])
        result = self._crew.kickoff(inputs=inputs)
        logger.info("Code generation crew completed")

        return {
            "success": True,
            "result": str(result),
            "steps": self._extract_task_results(),
        }


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def create_code_generation_crew(
    requirement: str,
    language: str = "en",
    config_dir: Optional[str] = None,
    model_override: Optional[BaseChatModel] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create and run a code generation crew in one call."""
    crew = CodeGenerationCrew(language=language, config_dir=config_dir, model_override=model_override)
    result = crew.kickoff_with_retry(requirement=requirement, context=context)
    return result.model_dump()
