"""Response-generation helpers extracted from graph_builder.node_generate_response.

The node stays in graph_builder as the orchestrator (so the heavy monkeypatch surface in
the test suite — load_core_config, get_model, track_* — keeps working); these modules hold
the pure, parameterized pieces it composes.
"""
