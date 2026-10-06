#
# Copyright (c) 2026, p99lab
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for the ONNX runner and the model loader, with onnxruntime and the Hub faked."""

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from pipecat_p99lab import _onnx
from pipecat_p99lab._onnx import STEP_SAMPLES, Turn1MiniStream, resolve_model_path


class FakeSession:
    """A stand-in graph: state is a running sum of the audio, probs are ramps of it."""

    def __init__(self, *args, **kwargs):
        self.runs = 0

    def get_inputs(self):
        names = ("audio", "conv1", "conv2", "k", "v", "k_valid", "h6")
        inputs = [SimpleNamespace(name="audio", shape=[1, STEP_SAMPLES])]
        return inputs + [SimpleNamespace(name=f"state_{n}", shape=[1, 2]) for n in names]

    def run(self, outputs, feed):
        self.runs += 1
        assert feed["audio"].shape == (1, STEP_SAMPLES) and feed["audio"].dtype == np.float32
        frame_sums = feed["audio"].reshape(8, -1).sum(axis=1)
        probs = (feed["state_h6"][0, 0] + np.cumsum(frame_sums)).astype(np.float32)
        new_states = [np.full((1, 2), probs[-1], np.float32) for _ in range(7)]
        return [probs] + new_states


@pytest.fixture
def stream(monkeypatch) -> Turn1MiniStream:
    import onnxruntime

    monkeypatch.setattr(onnxruntime, "InferenceSession", FakeSession)
    return Turn1MiniStream("unused.onnx")


def test_push_returns_eight_values_per_finished_step(stream):
    assert len(stream.push(np.ones(STEP_SAMPLES - 1, np.float32))) == 0
    probs = stream.push(np.ones(1, np.float32))
    assert len(probs) == 8
    assert stream.last_probability == pytest.approx(STEP_SAMPLES)
    assert len(stream.push(np.ones(3 * STEP_SAMPLES, np.float32))) == 24


def test_chunking_does_not_change_the_output(stream):
    audio = np.random.default_rng(0).standard_normal(5 * STEP_SAMPLES + 123).astype(np.float32)
    whole = stream.push(audio)
    stream.reset()
    pieces = np.concatenate([stream.push(audio[i : i + 1000]) for i in range(0, len(audio), 1000)])
    np.testing.assert_array_equal(whole, pieces)


def test_state_is_fed_back_and_reset_clears_it(stream):
    first = stream.push(np.ones(STEP_SAMPLES, np.float32))
    second = stream.push(np.ones(STEP_SAMPLES, np.float32))
    assert second[-1] == pytest.approx(2 * first[-1])
    stream.reset()
    assert stream.last_probability == 0.0
    np.testing.assert_array_equal(stream.push(np.ones(STEP_SAMPLES, np.float32)), first)


def test_peek_answers_between_steps_without_touching_the_stream(stream):
    stream.push(np.ones(STEP_SAMPLES + 1000, np.float32))  # 3 full frames + 40 samples pending
    assert stream.peek() == pytest.approx(STEP_SAMPLES + 3 * 320)
    assert stream.peek() == pytest.approx(STEP_SAMPLES + 3 * 320)  # repeatable
    rest = stream.push(np.ones(STEP_SAMPLES - 1000, np.float32))
    assert rest[-1] == pytest.approx(2 * STEP_SAMPLES)  # same as if peek had never run


def test_peek_with_less_than_one_frame_pending_returns_the_last_score(stream):
    stream.push(np.ones(STEP_SAMPLES + 100, np.float32))
    runs = stream._session.runs
    assert stream.peek() == stream.last_probability
    assert stream._session.runs == runs


def _fake_hub(monkeypatch, cached: bool):
    calls = []

    def hf_hub_download(**kwargs):
        calls.append(kwargs)
        if kwargs.get("local_files_only") and not cached:
            raise FileNotFoundError("not cached")
        return "/cache/" + kwargs["filename"]

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=hf_hub_download)
    )
    return calls


def test_loader_uses_the_cache_without_network_when_the_file_is_there(monkeypatch):
    calls = _fake_hub(monkeypatch, cached=True)
    assert resolve_model_path() == "/cache/onnx/turn-1-mini.step.int8dyn.onnx"
    assert len(calls) == 1 and calls[0]["local_files_only"] is True
    assert calls[0]["repo_id"] == "p99lab/turn-1-mini"
    assert calls[0]["revision"] == _onnx.DEFAULT_REVISION


def test_loader_downloads_once_when_the_file_is_missing(monkeypatch):
    calls = _fake_hub(monkeypatch, cached=False)
    assert resolve_model_path(variant="float32") == "/cache/onnx/turn-1-mini.step.onnx"
    assert len(calls) == 2 and "local_files_only" not in calls[1]


def test_loader_rejects_unknown_variants():
    with pytest.raises(ValueError):
        resolve_model_path(variant="int4")


def test_model_path_skips_the_hub(monkeypatch):
    import onnxruntime

    from pipecat_p99lab import LocalTurn1MiniAnalyzer

    calls = _fake_hub(monkeypatch, cached=False)
    monkeypatch.setattr(onnxruntime, "InferenceSession", FakeSession)
    LocalTurn1MiniAnalyzer(model_path="/models/turn-1-mini.step.onnx")
    assert calls == []
