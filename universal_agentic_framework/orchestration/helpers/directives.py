"""Directive parsing helpers for exact-reply and structured output detection."""

import re
from typing import Optional


def extract_exact_reply_directive(user_msg: str) -> Optional[str]:
    """Extract strict literal response directives from user text.

    Supports patterns like:
    - Reply with exactly: FOO
    - Respond with exactly: "FOO"
    - Antworte genau mit: FOO
    """
    if not user_msg:
        return None

    patterns = [
        r"(?is)\b(?:reply|respond)\s+with\s+exactly\s*:?\s*(?:\"([^\"]+)\"|'([^']+)'|([^\n\r]+))\s*$",
        r"(?is)\bantworte\s+genau\s+mit\s*:?\s*(?:\"([^\"]+)\"|'([^']+)'|([^\n\r]+))\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, user_msg)
        if not match:
            continue
        for group in match.groups():
            if group and group.strip():
                return group.strip()
    return None
