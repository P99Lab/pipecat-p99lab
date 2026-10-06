#
# Copyright (c) 2026, p99lab
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Shared test helpers: a scripted stand-in for the model and synthetic audio."""

import numpy as np
import pytest

from pipecat_p99lab import LocalTurn1MiniAnalyzer
from pipecat_p99lab._onnx import FRAME_SAMPLES, FRAMES_PER_STEP, STEP_SAMPLES

SAMPLE_RATE = 16000


class FakeStream:
    """Stands in for Turn1MiniStream.

    The score is ``rise_per_sec`` times the length of the silence at the end
    of the audio pushed so far (capped at 1.0), so tests can predict it.
    """

    def __init__(self, rise_per_sec: float = 1.0):
        self.rise_per_sec = rise_per_sec
        self.resets = 0
        self.samples_since_reset = 0
        self.total_samples = 0
        self.reset()
        self.resets = 0

    def reset(self):
        self.resets += 1
        self.samples_since_reset = 0
        self._pending = np.zeros(0, np.float32)
        self._silent_frames = 0
        self._last = 0.0

    @property
    def last_probability(self) -> float:
        return self._last

    def _frame_scores(self, audio: np.ndarray, silent_frames: int) -> tuple[list[float], int]:
        scores = []
        for frame in audio.reshape(-1, FRAME_SAMPLES):
            silent_frames = silent_frames + 1 if np.abs(frame).max() < 1e-3 else 0
            scores.append(min(1.0, self.rise_per_sec * silent_frames * 0.02))
        return scores, silent_frames

    def push(self, chunk: np.ndarray) -> np.ndarray:
        chunk = np.asarray(chunk, np.float32)
        self.samples_since_reset += len(chunk)
        self.total_samples += len(chunk)
        self._pending = np.concatenate([self._pending, chunk])
        out: list[float] = []
        while len(self._pending) >= STEP_SAMPLES:
            scores, self._silent_frames = self._frame_scores(
                self._pending[:STEP_SAMPLES], self._silent_frames
            )
            self._pending = self._pending[STEP_SAMPLES:]
            self._last = scores[-1]
            out.extend(scores)
        return np.asarray(out, np.float32)

    def peek(self) -> float:
        frames = len(self._pending) // FRAME_SAMPLES
        if frames == 0:
            return self._last
        scores, _ = self._frame_scores(self._pending[: frames * FRAME_SAMPLES], self._silent_frames)
        return scores[-1]


def tone(seconds: float, sample_rate: int = SAMPLE_RATE) -> bytes:
    """A loud 220 Hz tone as 16-bit PCM bytes: 'speech' for the fake model."""
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return (0.5 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16).tobytes()


def silence(seconds: float, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Digital silence as 16-bit PCM bytes."""
    return np.zeros(int(seconds * sample_rate), np.int16).tobytes()


def frames(pcm: bytes, sample_rate: int = SAMPLE_RATE, frame_ms: int = 20):
    """Split PCM bytes into transport-sized frames."""
    size = sample_rate * frame_ms // 1000 * 2
    for start in range(0, len(pcm), size):
        yield pcm[start : start + size]


def synthetic_speech(
    seconds: float = 3.0, sample_rate: int = SAMPLE_RATE, seed: int = 0
) -> np.ndarray:
    """Speech-like audio made from scratch: a voiced harmonic source with a
    falling pitch, three formant-like peaks and about four syllables per second.

    Returns:
        float32 audio in [-1, 1].
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * sample_rate)
    t = np.arange(n) / sample_rate
    f0 = 120 + 25 * np.sin(2 * np.pi * 0.7 * t) - 12 * t
    phase = 2 * np.pi * np.cumsum(f0) / sample_rate
    x = np.zeros(n)
    for h in range(1, 28):
        f = h * f0
        gain = sum(
            np.exp(-0.5 * ((f - c) / w) ** 2) for c, w in ((600, 150), (1400, 250), (2600, 350))
        )
        x += gain * np.sin(h * phase) / h**0.3
    envelope = np.clip(np.sin(2 * np.pi * 4 * t), 0, None) ** 0.6
    x = x * envelope + 0.01 * rng.standard_normal(n)
    return (0.3 * x / np.abs(x).max()).astype(np.float32)


@pytest.fixture
def fake_stream(monkeypatch) -> FakeStream:
    """Make LocalTurn1MiniAnalyzer use a FakeStream instead of loading the model."""
    stream = FakeStream()
    monkeypatch.setattr(LocalTurn1MiniAnalyzer, "_create_stream", lambda self, path, cpus: stream)
    return stream


@pytest.fixture
def make_analyzer(fake_stream):
    """Build an analyzer on the fake model, ready for 16 kHz audio by default."""

    def _make(sample_rate: int = SAMPLE_RATE, **kwargs) -> LocalTurn1MiniAnalyzer:
        analyzer = LocalTurn1MiniAnalyzer(model_path="unused.onnx", **kwargs)
        analyzer.set_sample_rate(sample_rate)
        return analyzer

    return _make


assert STEP_SAMPLES == FRAME_SAMPLES * FRAMES_PER_STEP
