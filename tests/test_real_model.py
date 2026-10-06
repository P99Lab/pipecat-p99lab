#
# Copyright (c) 2026, p99lab
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Integration tests with the real turn-1-mini ONNX graph from the Hugging Face Hub.

Needs the network the first time (afterwards the local cache is enough).
Skip it offline with:  pytest -m "not network"
"""

import numpy as np
import pytest
from conftest import SAMPLE_RATE, synthetic_speech
from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState

from pipecat_p99lab import LocalTurn1MiniAnalyzer, Turn1MiniParams
from pipecat_p99lab._onnx import Turn1MiniStream, resolve_model_path

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def model_path() -> str:
    try:
        return resolve_model_path()
    except RuntimeError as e:  # no network and nothing cached
        pytest.skip(f"turn-1-mini could not be loaded from the Hub: {e}")


def quiet(seconds: float) -> np.ndarray:
    """Room-like silence: very low noise rather than digital zeros."""
    rng = np.random.default_rng(1)
    return (1e-4 * rng.standard_normal(int(seconds * SAMPLE_RATE))).astype(np.float32)


def test_score_rises_when_speech_is_followed_by_silence(model_path):
    stream = Turn1MiniStream(model_path)
    probs = stream.push(np.concatenate([synthetic_speech(3.0), quiet(1.0)]))
    assert len(probs) == 200 and np.all((probs >= 0) & (probs <= 1))  # 50 scores per second

    during_speech = probs[50:150]
    after_300ms = probs[150 + 15]
    after_800ms = probs[150 + 40]
    assert after_300ms > during_speech.max()
    assert after_800ms > after_300ms
    assert after_800ms > 2 * during_speech.mean()


def test_peek_matches_the_score_the_stream_gives_later(model_path):
    stream = Turn1MiniStream(model_path)
    audio = np.concatenate([synthetic_speech(3.0), quiet(1.0)])
    cut = 3 * SAMPLE_RATE + 2560 + 1000  # 3 whole frames into a step
    stream.push(audio[:cut])
    now = stream.peek()
    later = stream.push(audio[cut : cut + 2560])
    assert now == pytest.approx(float(later[2]), abs=0.02)


def test_reset_gives_the_same_scores_as_a_new_stream(model_path):
    stream = Turn1MiniStream(model_path)
    audio = synthetic_speech(1.0)
    first = stream.push(audio)
    stream.push(quiet(0.5))
    stream.reset()
    np.testing.assert_array_equal(stream.push(audio), first)


def test_analyzer_declares_end_of_turn_from_the_model_not_the_timeout(model_path):
    analyzer = LocalTurn1MiniAnalyzer(
        model_path=model_path, params=Turn1MiniParams(threshold=0.4, stop_secs=3.0)
    )
    analyzer.set_sample_rate(SAMPLE_RATE)

    def pcm_frames(audio):
        pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        for start in range(0, len(pcm), 320):
            yield pcm[start : start + 320].tobytes()

    for frame in pcm_frames(quiet(0.5)):
        assert analyzer.append_audio(frame, False) == EndOfTurnState.INCOMPLETE
    for frame in pcm_frames(synthetic_speech(3.0)):
        assert analyzer.append_audio(frame, True) == EndOfTurnState.INCOMPLETE

    silence_ms = 0
    state = EndOfTurnState.INCOMPLETE
    for frame in pcm_frames(quiet(3.0)):
        silence_ms += 20
        state = analyzer.append_audio(frame, False)
        if state == EndOfTurnState.COMPLETE:
            break
    assert state == EndOfTurnState.COMPLETE
    assert silence_ms < 1500  # well before the 3 s stop_secs fallback


@pytest.mark.parametrize("rate", [8000, 16000, 24000, 48000])
def test_real_model_ends_the_turn_at_every_common_input_rate(model_path, rate):
    import soxr

    analyzer = LocalTurn1MiniAnalyzer(model_path=model_path, params=Turn1MiniParams(threshold=0.4))
    analyzer.set_sample_rate(rate)

    def pcm_frames(audio):
        if rate != SAMPLE_RATE:
            audio = soxr.resample(audio, SAMPLE_RATE, rate, quality="HQ")
        pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        size = rate // 50
        for start in range(0, len(pcm) - size + 1, size):
            yield pcm[start : start + size].tobytes()

    for frame in pcm_frames(synthetic_speech(3.0)):
        assert analyzer.append_audio(frame, True) == EndOfTurnState.INCOMPLETE
    silence_ms, state = 0, EndOfTurnState.INCOMPLETE
    for frame in pcm_frames(quiet(3.0)):
        silence_ms += 20
        state = analyzer.append_audio(frame, False)
        if state == EndOfTurnState.COMPLETE:
            break
    assert state == EndOfTurnState.COMPLETE
    assert silence_ms < 1500


def test_int8_variant_loads_and_scores_close_to_the_default(model_path):
    try:
        int8_path = resolve_model_path(variant="int8")
    except RuntimeError as e:
        pytest.skip(str(e))
    audio = np.concatenate([synthetic_speech(3.0), quiet(1.0)])
    a = Turn1MiniStream(model_path).push(audio)
    b = Turn1MiniStream(int8_path).push(audio)
    assert np.abs(a - b).max() < 0.15


def test_bad_samples_do_not_poison_the_stream():
    """NaN, infinity and out-of-range samples must not corrupt later steps."""
    import numpy as np

    from pipecat_p99lab._onnx import STEP_SAMPLES, Turn1MiniStream, resolve_model_path

    rng = np.random.default_rng(0)
    good = (0.05 * rng.standard_normal(STEP_SAMPLES)).astype(np.float32)
    clean = Turn1MiniStream(resolve_model_path())
    dirty = Turn1MiniStream(resolve_model_path())
    for bad in (
        np.full(STEP_SAMPLES, np.nan, np.float32),
        np.full(STEP_SAMPLES, np.inf, np.float32),
        (50 * rng.standard_normal(STEP_SAMPLES)).astype(np.float32),
    ):
        assert np.isfinite(dirty.push(bad)).all()
    for _ in range(100):  # 16 s: longer than the model's memory (six layers of 1.2 s each)
        a, b = clean.push(good), dirty.push(good)
    assert np.isfinite(b).all()
    assert np.abs(a - b).max() < 1e-3  # the stream has fully recovered
