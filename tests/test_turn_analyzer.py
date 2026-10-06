#
# Copyright (c) 2026, p99lab
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for LocalTurn1MiniAnalyzer. No network, no model file."""

import pytest
from conftest import SAMPLE_RATE, frames, silence, tone
from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer, BaseTurnParams, EndOfTurnState
from pipecat.metrics.metrics import TurnMetricsData

from pipecat_p99lab import LocalTurn1MiniAnalyzer, Turn1MiniParams


def feed(analyzer, pcm: bytes, is_speech: bool, sample_rate: int = SAMPLE_RATE):
    """Feed PCM in 20 ms frames. Returns (state, seconds fed when it completed)."""
    for i, frame in enumerate(frames(pcm, sample_rate)):
        if analyzer.append_audio(frame, is_speech) == EndOfTurnState.COMPLETE:
            return EndOfTurnState.COMPLETE, (i + 1) * 0.02
    return EndOfTurnState.INCOMPLETE, None


def test_is_a_pipecat_turn_analyzer(make_analyzer):
    analyzer = make_analyzer()
    assert isinstance(analyzer, BaseTurnAnalyzer)
    assert isinstance(analyzer.params, BaseTurnParams)
    assert analyzer.params == Turn1MiniParams(threshold=0.5, stop_secs=3, pre_speech_ms=500)
    assert analyzer.sample_rate == SAMPLE_RATE
    assert analyzer.speech_triggered is False
    assert analyzer.last_probability is None


def test_silence_before_speech_never_completes_and_does_not_run_the_model(
    make_analyzer, fake_stream
):
    analyzer = make_analyzer()
    state, _ = feed(analyzer, silence(5.0), is_speech=False)
    assert state == EndOfTurnState.INCOMPLETE
    assert analyzer.speech_triggered is False
    assert fake_stream.total_samples == 0


def test_completes_when_score_reaches_threshold_in_silence(make_analyzer):
    # The fake score rises 1.0 per second of silence, so 0.5 is reached 0.5 s
    # in; the analyzer sees it at the next 160 ms step boundary.
    analyzer = make_analyzer(params=Turn1MiniParams(threshold=0.5, pre_speech_ms=0))
    assert feed(analyzer, tone(1.6), is_speech=True)[0] == EndOfTurnState.INCOMPLETE
    assert analyzer.speech_triggered is True
    state, at = feed(analyzer, silence(2.0), is_speech=False)
    assert state == EndOfTurnState.COMPLETE
    assert at == pytest.approx(0.64, abs=0.021)
    assert analyzer.speech_triggered is False


async def test_analyze_after_complete_returns_metrics_once(make_analyzer):
    analyzer = make_analyzer(params=Turn1MiniParams(pre_speech_ms=0))
    feed(analyzer, tone(1.6), is_speech=True)
    assert feed(analyzer, silence(2.0), is_speech=False)[0] == EndOfTurnState.COMPLETE

    state, metrics = await analyzer.analyze_end_of_turn()
    assert state == EndOfTurnState.COMPLETE
    assert isinstance(metrics, TurnMetricsData)
    assert metrics.is_complete is True
    assert metrics.probability >= 0.5
    assert metrics.e2e_processing_time_ms >= 0

    assert await analyzer.analyze_end_of_turn() == (EndOfTurnState.INCOMPLETE, None)


async def test_unread_complete_does_not_end_the_next_turn(make_analyzer):
    # Pipecat 0.0.x does not call analyze_end_of_turn after append_audio
    # returns COMPLETE. The next turn's first VAD stop must get a fresh score.
    analyzer = make_analyzer(params=Turn1MiniParams(pre_speech_ms=0))
    feed(analyzer, tone(1.6), is_speech=True)
    assert feed(analyzer, silence(2.0), is_speech=False)[0] == EndOfTurnState.COMPLETE

    feed(analyzer, tone(1.6), is_speech=True)
    feed(analyzer, silence(0.2), is_speech=False)
    state, metrics = await analyzer.analyze_end_of_turn()
    assert state == EndOfTurnState.INCOMPLETE
    assert metrics.probability == pytest.approx(0.2)
    assert analyzer.speech_triggered is True


async def test_analyze_at_vad_stop_reads_the_score_between_step_boundaries(make_analyzer):
    # 1.0 s of tone, then 0.2 s of silence still flagged as speech (the VAD's
    # stop delay): 1.2 s = 7.5 steps, so half a step is pending when the VAD
    # reports the stop.
    analyzer = make_analyzer(params=Turn1MiniParams(threshold=0.15, pre_speech_ms=0))
    feed(analyzer, tone(1.0) + silence(0.2), is_speech=True)
    assert analyzer.last_probability == pytest.approx(0.12)  # last full step: 0.12 s of silence

    state, metrics = await analyzer.analyze_end_of_turn()
    assert state == EndOfTurnState.COMPLETE
    assert metrics.probability == pytest.approx(0.2)  # includes the pending 80 ms
    assert metrics.is_complete is True
    assert analyzer.speech_triggered is False


async def test_analyze_at_vad_stop_incomplete_keeps_the_turn_open(make_analyzer, fake_stream):
    analyzer = make_analyzer(params=Turn1MiniParams(threshold=0.5, pre_speech_ms=0))
    feed(analyzer, tone(1.0) + silence(0.2), is_speech=True)
    samples = fake_stream.samples_since_reset

    state, metrics = await analyzer.analyze_end_of_turn()
    assert state == EndOfTurnState.INCOMPLETE
    assert metrics.is_complete is False
    assert analyzer.speech_triggered is True
    assert fake_stream.samples_since_reset == samples  # the stream was not disturbed

    state, at = feed(analyzer, silence(2.0), is_speech=False)
    assert state == EndOfTurnState.COMPLETE
    # Step boundaries fall 0.08, 0.24 and 0.40 s later: scores 0.28, 0.44, 0.60.
    assert at == pytest.approx(0.40, abs=0.001)


async def test_analyze_without_speech_is_incomplete(make_analyzer):
    analyzer = make_analyzer()
    assert await analyzer.analyze_end_of_turn() == (EndOfTurnState.INCOMPLETE, None)


async def test_stop_secs_ends_the_turn_when_the_score_stays_low(make_analyzer, fake_stream):
    fake_stream.rise_per_sec = 0.0
    analyzer = make_analyzer(params=Turn1MiniParams(stop_secs=1.0))
    feed(analyzer, tone(1.0), is_speech=True)
    state, at = feed(analyzer, silence(3.0), is_speech=False)
    assert state == EndOfTurnState.COMPLETE
    assert at == pytest.approx(1.0, abs=0.021)
    # A timeout carries no model decision.
    assert await analyzer.analyze_end_of_turn() == (EndOfTurnState.COMPLETE, None)


async def test_fallback_ends_a_weakly_scored_ending_early(make_analyzer, fake_stream):
    # The score creeps up 0.1 per second: it never reaches 0.5, but after
    # 1.5 s of silence its peak is above the fallback's 0.1.
    fake_stream.rise_per_sec = 0.1
    analyzer = make_analyzer(params=Turn1MiniParams(fallback_secs=1.5, pre_speech_ms=0))
    feed(analyzer, tone(1.6), is_speech=True)
    state, at = feed(analyzer, silence(3.5), is_speech=False)
    assert state == EndOfTurnState.COMPLETE
    assert at == pytest.approx(1.5, abs=0.021)
    state, metrics = await analyzer.analyze_end_of_turn()
    assert state == EndOfTurnState.COMPLETE
    assert 0.1 <= metrics.probability < 0.5


def test_fallback_leaves_a_pause_scored_near_zero_to_stop_secs(make_analyzer, fake_stream):
    fake_stream.rise_per_sec = 0.01
    analyzer = make_analyzer(params=Turn1MiniParams(fallback_secs=1.5, pre_speech_ms=0))
    feed(analyzer, tone(1.6), is_speech=True)
    state, at = feed(analyzer, silence(3.5), is_speech=False)
    assert state == EndOfTurnState.COMPLETE
    assert at == pytest.approx(3.0, abs=0.021)


def test_fallback_is_off_by_default_and_peak_restarts_with_speech(make_analyzer, fake_stream):
    fake_stream.rise_per_sec = 0.1
    analyzer = make_analyzer(params=Turn1MiniParams(pre_speech_ms=0))
    feed(analyzer, tone(1.6), is_speech=True)
    assert feed(analyzer, silence(2.5), is_speech=False)[0] == EndOfTurnState.INCOMPLETE

    analyzer = make_analyzer(params=Turn1MiniParams(fallback_secs=1.5, pre_speech_ms=0))
    feed(analyzer, tone(1.6), is_speech=True)
    assert feed(analyzer, silence(1.4), is_speech=False)[0] == EndOfTurnState.INCOMPLETE
    feed(analyzer, tone(0.5), is_speech=True)
    # A new silence starts from scratch: 1.4 s is again too short.
    assert feed(analyzer, silence(1.4), is_speech=False)[0] == EndOfTurnState.INCOMPLETE


def test_speech_resuming_restarts_the_silence_clock_but_not_the_stream(make_analyzer, fake_stream):
    fake_stream.rise_per_sec = 0.0
    analyzer = make_analyzer(params=Turn1MiniParams(stop_secs=1.0, pre_speech_ms=0))
    feed(analyzer, tone(1.0), is_speech=True)
    assert feed(analyzer, silence(0.8), is_speech=False)[0] == EndOfTurnState.INCOMPLETE
    feed(analyzer, tone(0.5), is_speech=True)
    assert feed(analyzer, silence(0.8), is_speech=False)[0] == EndOfTurnState.INCOMPLETE
    assert fake_stream.resets == 1  # one stream for the whole turn
    assert fake_stream.samples_since_reset == int(3.1 * SAMPLE_RATE)
    assert feed(analyzer, silence(0.3), is_speech=False)[0] == EndOfTurnState.COMPLETE


def test_new_turn_starts_a_new_stream_seeded_with_pre_speech_audio(make_analyzer, fake_stream):
    analyzer = make_analyzer(params=Turn1MiniParams(pre_speech_ms=500))
    analyzer.update_vad_start_secs(0.2)
    feed(analyzer, silence(4.0), is_speech=False)
    feed(analyzer, tone(1.0), is_speech=True)
    # 0.5 s pre-speech + 0.2 s VAD start delay + 1.0 s of speech
    assert fake_stream.samples_since_reset == int(1.7 * SAMPLE_RATE)
    assert feed(analyzer, silence(2.0), is_speech=False)[0] == EndOfTurnState.COMPLETE

    resets = fake_stream.resets
    feed(analyzer, silence(0.1), is_speech=False)
    feed(analyzer, tone(0.4), is_speech=True)
    assert fake_stream.resets > resets
    assert fake_stream.samples_since_reset == int(0.5 * SAMPLE_RATE)  # nothing from the last turn


async def test_clear_resets_everything(make_analyzer, fake_stream):
    analyzer = make_analyzer(params=Turn1MiniParams(pre_speech_ms=0))
    feed(analyzer, tone(1.0), is_speech=True)
    feed(analyzer, silence(0.3), is_speech=False)
    assert analyzer.speech_triggered is True

    analyzer.clear()
    assert analyzer.speech_triggered is False
    assert fake_stream.samples_since_reset == 0
    assert await analyzer.analyze_end_of_turn() == (EndOfTurnState.INCOMPLETE, None)
    # Silence after a clear must not finish a turn that no longer exists.
    assert feed(analyzer, silence(4.0), is_speech=False)[0] == EndOfTurnState.INCOMPLETE


async def test_clear_drops_an_unread_complete(make_analyzer):
    analyzer = make_analyzer(params=Turn1MiniParams(pre_speech_ms=0))
    feed(analyzer, tone(1.0), is_speech=True)
    assert feed(analyzer, silence(2.0), is_speech=False)[0] == EndOfTurnState.COMPLETE
    analyzer.clear()
    assert await analyzer.analyze_end_of_turn() == (EndOfTurnState.INCOMPLETE, None)


@pytest.mark.parametrize("rate", [8000, 16000, 24000, 48000])
def test_every_common_sample_rate_reaches_the_model_at_16k(make_analyzer, fake_stream, rate):
    from loguru import logger

    complaints = []
    sink = logger.add(lambda message: complaints.append(str(message)), level="WARNING")
    analyzer = make_analyzer(sample_rate=rate, params=Turn1MiniParams(pre_speech_ms=0))
    assert analyzer.sample_rate == rate
    feed(analyzer, tone(2.0, rate), is_speech=True, sample_rate=rate)
    # The streaming resampler holds back a few milliseconds.
    assert fake_stream.samples_since_reset == pytest.approx(2.0 * SAMPLE_RATE, abs=800)
    state, at = feed(analyzer, silence(2.0, rate), is_speech=False, sample_rate=rate)
    assert state == EndOfTurnState.COMPLETE
    assert at == pytest.approx(0.6, abs=0.2)
    logger.remove(sink)
    assert complaints == []  # resampling needs no configuration and makes no noise


def test_presets():
    assert Turn1MiniParams() == Turn1MiniParams.balanced()
    assert Turn1MiniParams.fast().threshold == 0.2
    assert Turn1MiniParams.balanced().threshold == 0.5
    assert Turn1MiniParams.patient().threshold == 0.7
    assert Turn1MiniParams.fast().threshold < Turn1MiniParams.patient().threshold
    for preset in (Turn1MiniParams.fast, Turn1MiniParams.balanced, Turn1MiniParams.patient):
        assert preset().stop_secs == 3  # the safety timeout stays on
        assert preset(stop_secs=1.5).stop_secs == 1.5
        assert "Measured" in preset.__doc__
    assert Turn1MiniParams.fast(threshold=0.1).threshold == 0.1


def test_stop_secs_counts_time_at_the_input_rate(make_analyzer, fake_stream):
    fake_stream.rise_per_sec = 0.0
    analyzer = make_analyzer(sample_rate=48000, params=Turn1MiniParams(stop_secs=1.0))
    feed(analyzer, tone(0.5, 48000), is_speech=True, sample_rate=48000)
    state, at = feed(analyzer, silence(3.0, 48000), is_speech=False, sample_rate=48000)
    assert state == EndOfTurnState.COMPLETE
    assert at == pytest.approx(1.0, abs=0.021)


def test_fixed_sample_rate_wins_over_the_pipeline_rate(fake_stream):
    analyzer = LocalTurn1MiniAnalyzer(model_path="unused.onnx", sample_rate=8000)
    analyzer.set_sample_rate(48000)
    assert analyzer.sample_rate == 8000


def test_empty_buffer_is_ignored(make_analyzer):
    analyzer = make_analyzer()
    assert analyzer.append_audio(b"", True) == EndOfTurnState.INCOMPLETE
    assert analyzer.speech_triggered is False
