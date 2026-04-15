"""Legacy tests for the removed in-repo web_search server.

The framework migrated to the external `mcp/duckduckgo` service, so the
`universal_agentic_framework.tools.web_search.src` module no longer exists.
"""

import pytest


pytestmark = pytest.mark.skip(
    reason="Legacy in-repo web_search module removed; covered by MCP integration tests."
)
