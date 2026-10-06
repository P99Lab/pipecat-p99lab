#
# Copyright (c) 2026, p99lab
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The analyzer inside Pipecat's own TurnAnalyzerUserTurnStopStrategy (fake model, no network)."""

import inspect

import pytest
from pipecat.processors.frame_processor import FrameProcessorSetup as _Setup

# These tests drive the user-turn strategy API of Pipecat 1.x. On 0.0.x the analyzer is given to the transport.
pytestmark = pytest.mark.skipif(
    "pipeline_worker" not in inspect.signature(_Setup).parameters,
    reason="needs the user-turn strategy API of Pipecat 1.x",
)

from conftest import SAMPLE_RATE, frames, silence, tone
from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    InputAudioRawFrame,
    MetricsFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import TurnMetricsData
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.utils.asyncio.task_manager import TaskManager

from pipecat_p99lab import Turn1MiniParams


async def make_strategy(analyzer, sample_rate=SAMPLE_RATE, **kwargs):
    strategy = TurnAnalyzerUserTurnStopStrategy(turn_analyzer=analyzer, **kwargs)
    await strategy.setup(
        FrameProcessorSetup(
            clock=SystemClock(),
            task_manager=TaskManager(),
            pipeline_worker=None,
            audio_in_sample_rate=sample_rate,
        )
    )
    events = {"stopped": 0, "metrics": []}

    @strategy.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(strategy, params):
        events["stopped"] += 1

    @strategy.event_handler("on_push_frame")
    async def on_push_frame(strategy, frame, direction):
        if isinstance(frame, MetricsFrame):
            events["metrics"].extend(frame.data)

    return strategy, events


async def send_audio(strategy, pcm, sample_rate=SAMPLE_RATE):
    for chunk in frames(pcm, sample_rate):
        await strategy.process_frame(
            InputAudioRawFrame(audio=chunk, sample_rate=sample_rate, num_channels=1)
        )


async def test_strategy_sets_the_sample_rate_and_ends_the_turn(make_analyzer):
    analyzer = make_analyzer(params=Turn1MiniParams(threshold=0.5))
    strategy, events = await make_strategy(analyzer, sample_rate=24000, wait_for_transcript=False)
    assert analyzer.sample_rate == 24000

    await send_audio(strategy, silence(1.0, 24000), 24000)
    await strategy.process_frame(VADUserStartedSpeakingFrame(start_secs=0.2))
    await send_audio(strategy, tone(1.0, 24000) + silence(0.2, 24000), 24000)
    await strategy.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2))
    # 0.2 s of silence scores 0.2 on the fake model: not complete yet.
    assert events["stopped"] == 0
    assert [m.is_complete for m in events["metrics"]] == [False]

    await send_audio(strategy, silence(1.0, 24000), 24000)
    assert events["stopped"] == 1
    assert isinstance(events["metrics"][-1], TurnMetricsData)
    assert events["metrics"][-1].is_complete is True
    assert events["metrics"][-1].processor == "LocalTurn1MiniAnalyzer"
    await strategy.cleanup()


async def test_strategy_ends_the_turn_at_vad_stop_when_the_score_is_already_high(make_analyzer):
    analyzer = make_analyzer(params=Turn1MiniParams(threshold=0.15))
    strategy, events = await make_strategy(analyzer, wait_for_transcript=False)
    await strategy.process_frame(VADUserStartedSpeakingFrame(start_secs=0.2))
    await send_audio(strategy, tone(1.0) + silence(0.2))
    await strategy.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2))
    assert events["stopped"] == 1
    await strategy.cleanup()


async def test_strategy_waits_for_a_transcript_by_default(make_analyzer):
    analyzer = make_analyzer(params=Turn1MiniParams(threshold=0.15))
    strategy, events = await make_strategy(analyzer)
    await strategy.process_frame(VADUserStartedSpeakingFrame(start_secs=0.2))
    await send_audio(strategy, tone(1.0) + silence(0.2))
    await strategy.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2))
    assert events["stopped"] == 0
    await strategy.process_frame(
        TranscriptionFrame(text="Hello!", user_id="user", timestamp="", finalized=True)
    )
    assert events["stopped"] == 1
    await strategy.cleanup()


async def test_user_speaking_again_keeps_the_turn_open(make_analyzer):
    analyzer = make_analyzer(params=Turn1MiniParams(threshold=0.5))
    strategy, events = await make_strategy(analyzer, wait_for_transcript=False)
    await strategy.process_frame(VADUserStartedSpeakingFrame(start_secs=0.2))
    await send_audio(strategy, tone(1.0) + silence(0.2))
    await strategy.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2))
    await send_audio(strategy, silence(0.1))
    await strategy.process_frame(VADUserStartedSpeakingFrame(start_secs=0.2))
    await send_audio(strategy, tone(1.0))
    assert events["stopped"] == 0
    assert analyzer.speech_triggered is True
    await strategy.cleanup()
