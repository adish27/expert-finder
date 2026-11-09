"""Test-wide setup.

Tracing is switched off here rather than in a fixture, and before anything imports
expert_finder: `load_dotenv` does not override variables that are already set, so setting this
first is what stops .env turning tracing back on. Without it every `pytest` run files the
scripted-model agent runs in the real LangSmith project, next to genuine ones.

Set LANGSMITH_TRACING=true in the environment to trace a test run deliberately.
"""

import os

if "LANGSMITH_TRACING" not in os.environ:
    os.environ["LANGSMITH_TRACING"] = "false"
