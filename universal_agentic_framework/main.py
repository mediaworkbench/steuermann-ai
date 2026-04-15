"""Entry point placeholder.

In future, this can start LangGraph Platform/Server or provide a simple CLI.
"""
from universal_agentic_framework.graph import run_graph


def main() -> None:
    result = run_graph("hello")
    print(result["response"])


if __name__ == "__main__":
    main()
