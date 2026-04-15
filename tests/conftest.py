import os


# Ensure config placeholder substitution succeeds in tests that call load_core_config() directly.
os.environ.setdefault("LLM_ENDPOINT", "http://localhost:11434")
