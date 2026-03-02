"""Namespace tests for ``thenvoi.preprocessing``."""

from __future__ import annotations

import pytest

import thenvoi.preprocessing as preprocessing_namespace
from thenvoi.preprocessing.default import DefaultPreprocessor


def test_namespace_exports_default_preprocessor() -> None:
    assert preprocessing_namespace.DefaultPreprocessor is DefaultPreprocessor


def test_namespace_dir_includes_public_exports() -> None:
    assert "DefaultPreprocessor" in dir(preprocessing_namespace)


def test_namespace_rejects_unknown_attributes() -> None:
    with pytest.raises(AttributeError, match="has no attribute"):
        getattr(preprocessing_namespace, "UnknownSymbol")
