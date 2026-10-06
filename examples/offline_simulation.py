#
# Copyright (c) 2026, p99lab
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Offline check: play an audio file through the analyzer the way a transport would.

No microphone, no API keys. The file is cut into 20 ms frames, Pipecat's
Silero VAD marks speech, and the frames go through Pipecat's
``TurnAnalyzerUserTurnStopStrategy`` with ``LocalTurn1MiniAnalyzer`` inside.
Silence is appended to the file so the end of the turn can be observed.

    python examples/offline_simulation.py speech.wav
    python examples/offline_simulation.py speech.wav --threshold 0.2 --sample-rate 8000

Times are positions in the audio, not wall-clock time.
"""

import argparse
import asyncio
import sys

import numpy as np
import soundfile as sf
import soxr
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADState
from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    InputAudioRawFrame,
    MetricsFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import TurnMetricsData
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.utils.asyncio.task_manager import TaskManager

from pipecat_p99lab import LocalTurn1MiniAnalyzer, Turn1MiniParams


async def simulate(args) -> int:
    audio, file_rate = sf.read(args.audio, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    rate = args.sample_rate
    if rate != file_rate:
        audio = soxr.resample(audio, file_rate, rate, quality="HQ")
    speech_secs = len(audio) / rate
    rng = np.random.default_rng(0)

    def room(secs: float) -> np.ndarray:
        """Room-like silence: very low noise rather than digital zeros."""
        return (1e-4 * rng.standard_normal(int(secs * rate))).astype(np.float32)

    audio = np.concatenate([room(args.lead_secs), audio, room(args.silence_secs)])
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    file_end = args.lead_secs + speech_secs

    analyzer = LocalTurn1MiniAnalyzer(
        variant=args.variant,
        params=Turn1MiniParams(threshold=args.threshold, stop_secs=args.stop_secs),
    )
    strategy = TurnAnalyzerUserTurnStopStrategy(turn_analyzer=analyzer, wait_for_transcript=False)
    await strategy.setup(
        FrameProcessorSetup(
            clock=SystemClock(),
            task_manager=TaskManager(),
            pipeline_worker=None,
            audio_in_sample_rate=rate,
        )
    )
    vad = SileroVADAnalyzer(sample_rate=rate)
    vad.set_sample_rate(rate)

    now = 0.0
    vad_stop = 0.0
    turn_ends: list[float] = []

    @strategy.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(strategy, params):
        turn_ends.append(now)
        print(
            f"{now:7.2f} s  END OF TURN  ({(now - vad_stop) * 1000:.0f} ms after the VAD "
            f"reported the stop, which itself comes {vad.params.stop_secs * 1000:.0f} ms "
            f"after the speech ended)"
        )

    @strategy.event_handler("on_push_frame")
    async def on_push_frame(strategy, frame, direction):
        if isinstance(frame, MetricsFrame):
            for data in frame.data:
                if isinstance(data, TurnMetricsData):
                    verdict = "complete" if data.is_complete else "incomplete"
                    print(
                        f"{now:7.2f} s  model: P(end of turn) = {data.probability:.3f} -> {verdict}"
                    )

    print(
        f"{args.audio}: {speech_secs:.2f} s of audio at {rate} Hz, then {args.silence_secs:.1f} s "
        f"of silence; threshold {args.threshold}, stop_secs {args.stop_secs}"
    )

    frame_samples = rate // 50  # 20 ms, as transports deliver it
    speaking = False
    for start in range(0, len(pcm) - frame_samples + 1, frame_samples):
        chunk = pcm[start : start + frame_samples].tobytes()
        now = (start + frame_samples) / rate
        state = await vad.analyze_audio(chunk)
        if state == VADState.SPEAKING and not speaking:
            speaking = True
            print(f"{now:7.2f} s  VAD: user started speaking")
            await strategy.process_frame(
                VADUserStartedSpeakingFrame(start_secs=vad.params.start_secs)
            )
        elif state == VADState.QUIET and speaking:
            speaking = False
            vad_stop = now
            print(f"{now:7.2f} s  VAD: user stopped speaking")
            await strategy.process_frame(
                VADUserStoppedSpeakingFrame(stop_secs=vad.params.stop_secs)
            )
        await strategy.process_frame(
            InputAudioRawFrame(audio=chunk, sample_rate=rate, num_channels=1)
        )
        tracing = args.trace and now > file_end - 0.5 and analyzer.speech_triggered
        if tracing and round(now * 50) % 8 == 0 and analyzer.last_probability is not None:
            print(f"{now:7.2f} s    score {analyzer.last_probability:.3f}")

    await strategy.cleanup()
    await vad.cleanup()
    if not turn_ends:
        print("No end of turn was declared.")
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "audio", help="A speech recording (wav or flac; resampled to --sample-rate)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="fast 0.2, balanced 0.5 (default), patient 0.7"
    )
    parser.add_argument("--stop-secs", type=float, default=3.0)
    parser.add_argument("--variant", default="float32", choices=["float32", "int8"])
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        choices=[8000, 16000],
        help="Simulated transport rate (the rates Silero VAD accepts)",
    )
    parser.add_argument("--lead-secs", type=float, default=1.0, help="Silence before the audio")
    parser.add_argument("--silence-secs", type=float, default=4.0, help="Silence after the audio")
    parser.add_argument("--trace", action="store_true", help="Print the score every 160 ms")
    args = parser.parse_args()
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    sys.exit(asyncio.run(simulate(args)))


if __name__ == "__main__":
    main()
