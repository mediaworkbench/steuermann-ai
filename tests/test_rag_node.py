

def test_filter_and_deduplicate_orders_by_score_and_keeps_best_duplicate():
    from universal_agentic_framework.orchestration.helpers.rag_retrieval import filter_and_deduplicate

    results = [
        {"id": "a", "score": 0.74, "payload": {"text": "variant hit", "file_path": "a.md"}},
        {"id": "b", "score": 0.80, "payload": {"text": "b", "file_path": "b.md"}},
        {"id": "a", "score": 0.95, "payload": {"text": "keyword hit", "file_path": "a.md"}},
    ]
    docs = filter_and_deduplicate(results, min_relevance_score=0.72, top_k=5)

    assert [(d["file_name"], d["score"]) for d in docs] == [("a.md", 0.95), ("b.md", 0.80)]
