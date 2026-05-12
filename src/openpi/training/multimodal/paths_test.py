"""Unit tests for the data-root resolution logic."""

from __future__ import annotations

import os
import pathlib

from openpi.training.multimodal import paths


def test_explicit_takes_precedence_over_env(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.DATA_ROOT_ENV, str(tmp_path / "env_root"))
    explicit = tmp_path / "explicit_root"
    explicit.mkdir()
    result = paths.resolve_data_root(explicit)
    assert result == explicit.resolve()


def test_env_var_used_when_no_explicit(monkeypatch, tmp_path):
    env_root = tmp_path / "env_root"
    env_root.mkdir()
    monkeypatch.setenv(paths.DATA_ROOT_ENV, str(env_root))
    result = paths.resolve_data_root()
    assert result == env_root.resolve()


def test_default_falls_back_to_sibling_or_in_tree(monkeypatch):
    monkeypatch.delenv(paths.DATA_ROOT_ENV, raising=False)
    result = paths.resolve_data_root()
    # We can't assume which of sibling vs in-tree exists on a given host, but
    # the resolved path must be absolute and end with `Data`.
    assert isinstance(result, pathlib.Path)
    assert result.is_absolute()
    assert result.name == "Data"


def test_expanduser_handled(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(paths.DATA_ROOT_ENV, raising=False)
    result = paths.resolve_data_root("~/somewhere")
    assert str(result).startswith(str(tmp_path))
    assert result.name == "somewhere"
    # Sanity: no leading tilde survives.
    assert "~" not in str(result)
    # Avoid an unused-import warning when running ruff in tests.
    _ = os.environ
