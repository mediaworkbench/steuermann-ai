from universal_agentic_framework.graph import run_graph, run_llm_graph
from universal_agentic_framework.llm.budget import TokenBudgetExceeded


def test_run_graph_echo():
    result = run_graph("ping")
    assert result["response"] == "Echo: ping"


class _FakeChatModel:
    def invoke(self, prompt: str):
        class _Out:
            content = f"LLM: {prompt}"

        return _Out()


def test_run_llm_graph_with_override():
    fake = _FakeChatModel()
    res = run_llm_graph("hello", language="en", model_override=fake)
    assert res["response"] == "LLM: hello"
    assert res["tokens_used"] > 0
    assert res["token_budget"] >= res["tokens_used"]


def test_run_llm_graph_budget_exceeded():
    fake = _FakeChatModel()
    # Create a very large input to blow past default budget (10000/approx)
    huge = "x" * (10000 * 5)  # 5x characters to exceed char/4 estimator
    try:
        run_llm_graph(huge, language="en", model_override=fake)
    except TokenBudgetExceeded as exc:
        # Ensure the exception message mentions exhausted/budget
        assert "exceeds budget" in str(exc) or "Budget exhausted" in str(exc)
    else:
        assert False, "Expected TokenBudgetExceeded"
