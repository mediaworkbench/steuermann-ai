"""Smoke tests for the steuermann CLI surface."""

import universal_agentic_framework.cli.steuermann as steuermann


def test_parser_has_expected_top_level_commands() -> None:
    parser = steuermann.create_parser()
    args = parser.parse_args(["profile", "active"])
    assert args.command == "profile"
    assert args.profile_command == "active"


def test_ingest_group_has_reindex_subcommand() -> None:
    parser = steuermann.create_parser()
    args = parser.parse_args(
        [
            "ingest",
            "reindex",
            "--source",
            "data/rag-data",
            "--yes",
        ]
    )
    assert args.command == "ingest"
    assert args.ingest_command == "reindex"


def test_docs_check_command_runs_non_strict() -> None:
    code = steuermann.main(["docs", "check", "--format", "json"])
    assert code == 0
