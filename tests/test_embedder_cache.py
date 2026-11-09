# pyright: reportPrivateUsage=false, reportUnusedFunction=false
"""Tests for the idle-unloading embedding model cache (no real models loaded)."""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest
from memory_vec import embedder as emb
from memory_vec.embedder import _IdleModelCache


class _Loader:
    """Counts calls and hands out a fresh sentinel object per load."""

    def __init__(self, delay: float = 0.0) -> None:
        self.calls = 0
        self._delay = delay

    def __call__(self) -> object:
        self.calls += 1
        if self._delay:
            time.sleep(self._delay)
        return object()


@pytest.fixture(autouse=True)
def _no_reaper(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable the background reaper unless a test opts back in."""
    monkeypatch.setenv(emb._IDLE_UNLOAD_ENV, "0")


# ---------------------------------------------------------------------------
# _idle_unload_seconds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, emb._DEFAULT_IDLE_UNLOAD_SECONDS),
        ("", emb._DEFAULT_IDLE_UNLOAD_SECONDS),
        ("120", 120.0),
        ("0", 0.0),
        ("-5", 0.0),
        ("nope", emb._DEFAULT_IDLE_UNLOAD_SECONDS),
    ],
)
def test_idle_unload_seconds_parsing(monkeypatch: pytest.MonkeyPatch, raw: str | None, expected: float) -> None:
    if raw is None:
        monkeypatch.delenv(emb._IDLE_UNLOAD_ENV, raising=False)
    else:
        monkeypatch.setenv(emb._IDLE_UNLOAD_ENV, raw)
    assert emb._idle_unload_seconds() == expected


# ---------------------------------------------------------------------------
# _IdleModelCache
# ---------------------------------------------------------------------------


def test_get_loads_once_and_reuses() -> None:
    cache = _IdleModelCache("t")
    loader = _Loader()
    first = cache.get("m", loader)
    assert cache.get("m", loader) is first
    assert loader.calls == 1
    assert "m" in cache and len(cache) == 1


def test_concurrent_first_get_loads_once() -> None:
    cache = _IdleModelCache("t")
    loader = _Loader(delay=0.05)
    results: list[Any] = []
    threads = [threading.Thread(target=lambda: results.append(cache.get("m", loader))) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert loader.calls == 1
    assert all(r is results[0] for r in results)


def test_evict_idle_drops_only_stale_entries() -> None:
    cache = _IdleModelCache("t")
    cache.get("old", _Loader())
    cache.get("new", _Loader())
    now = time.monotonic()
    cache._last_used["old"] = now - 100
    cache._last_used["new"] = now - 1

    assert cache.evict_idle(60, now=now) == 1
    assert "old" not in cache
    assert "new" in cache


def test_evicted_entry_reloads_on_next_get() -> None:
    cache = _IdleModelCache("t")
    loader = _Loader()
    first = cache.get("m", loader)
    assert cache.clear() == 1
    second = cache.get("m", loader)
    assert second is not first
    assert loader.calls == 2


def test_get_refreshes_last_used() -> None:
    cache = _IdleModelCache("t")
    loader = _Loader()
    cache.get("m", loader)
    cache._last_used["m"] = time.monotonic() - 100
    cache.get("m", loader)  # touch
    assert cache.evict_idle(60) == 0
    assert loader.calls == 1


def test_no_reaper_when_unload_disabled() -> None:
    cache = _IdleModelCache("t")
    cache.get("m", _Loader())
    assert cache._reaper is None


def test_reaper_unloads_after_idle_and_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(emb._IDLE_UNLOAD_ENV, "0.2")
    cache = _IdleModelCache("t")
    cache.get("m", _Loader())
    reaper = cache._reaper
    assert reaper is not None and reaper.is_alive()

    deadline = time.monotonic() + 5
    while "m" in cache and time.monotonic() < deadline:
        time.sleep(0.02)
    assert "m" not in cache

    reaper.join(timeout=5)
    assert not reaper.is_alive()
    assert cache._reaper is None

    # A later load starts a fresh reaper.
    cache.get("m", _Loader())
    assert cache._reaper is not None and cache._reaper is not reaper
    cache.clear()


def test_evict_failure_in_reaper_does_not_kill_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(emb._IDLE_UNLOAD_ENV, "0.2")
    cache = _IdleModelCache("t")
    real_evict = cache.evict_idle
    calls = {"n": 0}

    def flaky(idle_seconds: float, now: float | None = None) -> int:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return real_evict(idle_seconds, now)

    monkeypatch.setattr(cache, "evict_idle", flaky)
    cache.get("m", _Loader())
    deadline = time.monotonic() + 5
    while "m" in cache and time.monotonic() < deadline:
        time.sleep(0.02)
    assert "m" not in cache
    assert calls["n"] >= 2


# ---------------------------------------------------------------------------
# Embedders: dimension must not force a model load for known models
# ---------------------------------------------------------------------------


def test_onnx_dimension_known_model_does_not_load(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(emb, "_onnx_available", True)

    def _fail_load(self: emb.OnnxEmbedder) -> tuple[Any, Any]:
        raise AssertionError("model should not be loaded")

    monkeypatch.setattr(emb.OnnxEmbedder, "_load", _fail_load)
    assert emb.OnnxEmbedder().dimension == 384


def test_onnx_dimension_unknown_model_loads_and_records(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(emb, "_onnx_available", True)
    monkeypatch.setattr(emb.OnnxEmbedder, "_cache", _IdleModelCache("onnx-test"))
    monkeypatch.setattr(emb, "_KNOWN_DIMENSIONS", dict(emb._KNOWN_DIMENSIONS))
    loads = {"n": 0}

    def _fake_load(self: emb.OnnxEmbedder) -> tuple[Any, Any]:
        loads["n"] += 1
        emb._KNOWN_DIMENSIONS[self._model_name] = 768
        return object(), object()

    monkeypatch.setattr(emb.OnnxEmbedder, "_load", _fake_load)
    e = emb.OnnxEmbedder(model_name="custom-model")
    assert e.dimension == 768
    # After the model is unloaded the dimension is still known — no reload.
    emb.OnnxEmbedder._cache.clear()
    assert e.dimension == 768
    assert loads["n"] == 1


def test_unload_models_clears_both_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    onnx_cache = _IdleModelCache("onnx-test")
    st_cache = _IdleModelCache("st-test")
    monkeypatch.setattr(emb.OnnxEmbedder, "_cache", onnx_cache)
    monkeypatch.setattr(emb.SentenceTransformerEmbedder, "_model_cache", st_cache)
    onnx_cache.get("a", _Loader())
    st_cache.get("b", _Loader())

    assert emb.unload_models() == 2
    assert len(onnx_cache) == 0 and len(st_cache) == 0
