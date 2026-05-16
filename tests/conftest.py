"""Shared pytest hooks so unit tests run without a local ``.env`` file."""

from __future__ import annotations

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:  # noqa: ARG001
    """Guarantee a minimal OPENAI_API_KEY before any ``src.api`` imports in tests."""
    os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder")
