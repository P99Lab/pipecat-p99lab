#
# Copyright (c) 2026, p99lab
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Local turn analyzer for Pipecat using the turn-1-mini model by p99lab.

turn-1-mini is a small causal streaming end-of-turn model that works on audio
only. This module wraps its ONNX graph in a Pipecat ``BaseTurnAnalyzer`` so it
can be used with ``TurnAnalyzerUserTurnStopStrategy``.
"""

import time
from collections import deque

import numpy as np
import soxr
from loguru import logger
from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer, BaseTurnParams, EndOfTurnState
from pipecat.metrics.metrics import MetricsData, TurnMetricsData

from pipecat_p99lab._onnx import (
    DEFAULT_REPO_ID,
    DEFAULT_REVISION,
    MODEL_SAMPLE_RATE,
    Turn1MiniStream,
    resolve_model_path,
)

# Default parameters
THRESHOLD = 0.5
STOP_SECS = 3
PRE_SPEECH_MS = 500


class Turn1MiniParams(BaseTurnParams):
    """Configuration parameters for turn-1-mini turn analysis.

    Start from one of the presets, :meth:`fast`, :meth:`balanced` (the
    default) or :meth:`patient`, or set ``threshold`` yourself.

    Parameters:
        threshold: P(end of turn) at or above which the turn is complete
            (0.0 to 1.0). The model's scores are not calibrated: tune this on
            your own audio. Lower values respond sooner and cut in more often.
        stop_secs: Safety timeout: silence duration in seconds after which the
            turn ends regardless of the model's score.
        pre_speech_ms: Milliseconds of audio before the detected start of
            speech that are given to the model as context.
    """

    threshold: float = THRESHOLD
    stop_secs: float = STOP_SECS
    pre_speech_ms: float = PRE_SPEECH_MS

    # The figures in the presets below were measured by p99lab on its own
    # development audio (English phone-call turns: 608 turns, 379 mid-turn
    # pauses of 0.2 to 5 s), run through Pipecat's Silero VAD and this analyzer
    # with stop_secs=3. "Wrong endings" is the share of mid-turn pauses in
    # which the turn was ended; the delay runs from the end of speech to the
    # end-of-turn decision and includes the VAD's 0.2 s stop delay. Your audio
    # will differ: treat them as a starting point.

    @classmethod
    def fast(cls, **overrides) -> "Turn1MiniParams":
        """Answer as soon as possible (threshold 0.2).

        Measured: 16.4% wrong endings; delay 200 ms at the median and 200 ms at
        the 90th percentile.

        Args:
            **overrides: Any other parameter to set, e.g. ``stop_secs``.

        Returns:
            The preset parameters.
        """
        return cls(**{"threshold": 0.2, **overrides})

    @classmethod
    def balanced(cls, **overrides) -> "Turn1MiniParams":
        """The default (threshold 0.5).

        Measured: 11.3% wrong endings; delay 200 ms at the median and 750 ms at
        the 90th percentile.

        Args:
            **overrides: Any other parameter to set, e.g. ``stop_secs``.

        Returns:
            The preset parameters.
        """
        return cls(**{"threshold": 0.5, **overrides})

    @classmethod
    def patient(cls, **overrides) -> "Turn1MiniParams":
        """Interrupt least; wait out more pauses (threshold 0.7).

        Measured: 7.9% wrong endings; delay 200 ms at the median, and about one
        turn in ten ends only at the ``stop_secs`` timeout (3 s).

        Args:
            **overrides: Any other parameter to set, e.g. ``stop_secs``.

        Returns:
            The preset parameters.
        """
        return cls(**{"threshold": 0.7, **overrides})


class LocalTurn1MiniAnalyzer(BaseTurnAnalyzer):
    """Local turn analyzer using the turn-1-mini ONNX model.

    The model runs as a live stream on the user's audio. A new stream starts
    from a zero state at each turn (seeded with a short stretch of audio from
    before the speech started), so the score at any moment equals the score of
    the turn's audio so far. While the VAD reports silence, the latest score is
    compared with ``params.threshold``: once at the moment the VAD reports the
    stop, then after every 160 ms step. If the score never reaches the
    threshold, the turn ends after ``params.stop_secs`` of silence.

    Inference runs on the CPU inside ``append_audio`` (about 1 to 2 ms per
    160 ms of audio on one core). After the model file is in the local Hugging
    Face cache, nothing touches the network.
    """

    def __init__(
        self,
        *,
        model_path: str | None = None,
        variant: str = "float32",
        repo_id: str = DEFAULT_REPO_ID,
        revision: str | None = DEFAULT_REVISION,
        cpu_count: int = 1,
        sample_rate: int | None = None,
        params: Turn1MiniParams | None = None,
    ):
        """Initialize the turn-1-mini analyzer.

        Args:
            model_path: Path to a ``turn-1-mini.step*.onnx`` file. If this is
                not set, the graph is taken from the local Hugging Face cache
                and downloaded on first use.
            variant: Which published graph to load when ``model_path`` is not
                set: ``"float32"`` (21.8 MB, default, the benchmarked weights)
                or ``"int8"`` (7.0 MB, 8-bit weights, faster).
            repo_id: Hugging Face repository to load the graph from.
            revision: Branch, tag or commit of the repository. Defaults to the
                commit this package was tested against.
            cpu_count: The number of CPUs to use for inference. Defaults to 1.
            sample_rate: Optional sample rate of the incoming audio. If not
                set, the pipeline's input sample rate is used.
            params: Configuration parameters for turn analysis behavior.
        """
        super().__init__(sample_rate=sample_rate)
        self._params = params or Turn1MiniParams()

        if not model_path:
            model_path = resolve_model_path(variant=variant, repo_id=repo_id, revision=revision)

        logger.debug(f"Loading turn-1-mini model from {model_path}...")
        self._model = self._create_stream(model_path, cpu_count)
        logger.debug("Loaded turn-1-mini")

        self._resampler: soxr.ResampleStream | None = None
        self._resampler_rate = 0

        # Audio heard before the speech started, already at the model's rate.
        self._pre_speech: deque[np.ndarray] = deque()
        self._pre_speech_samples = 0
        self._vad_start_secs = 0.0

        self._speech_triggered = False
        self._silence_ms = 0.0
        self._speech_stopped_time: float | None = None
        self._last_probability: float | None = None
        # Result of a COMPLETE decided in append_audio, handed out by the next
        # analyze_end_of_turn call.
        self._pending_complete = False
        self._pending_metrics: TurnMetricsData | None = None

    def _create_stream(self, model_path: str, cpu_count: int) -> Turn1MiniStream:
        """Open the model. Kept separate so tests can substitute a fake."""
        return Turn1MiniStream(model_path, cpu_count=cpu_count)

    @property
    def speech_triggered(self) -> bool:
        """Check if speech has been detected and triggered analysis.

        Returns:
            True if speech has been detected and turn analysis is active.
        """
        return self._speech_triggered

    @property
    def params(self) -> Turn1MiniParams:
        """Get the current turn analyzer parameters.

        Returns:
            Current turn analyzer configuration parameters.
        """
        return self._params

    @property
    def last_probability(self) -> float | None:
        """Get the most recent P(end of turn) the model produced.

        Returns:
            The last score, or None if the model has not scored any audio in
            the current turn yet.
        """
        return self._last_probability

    def update_vad_start_secs(self, vad_start_secs: float):
        """Store the VAD start delay so the pre-speech context covers it.

        Args:
            vad_start_secs: Seconds of voice activity before the VAD reports
                that the user started speaking.
        """
        self._vad_start_secs = vad_start_secs

    def append_audio(self, buffer: bytes, is_speech: bool) -> EndOfTurnState:
        """Append audio data for turn analysis.

        Args:
            buffer: Raw 16-bit mono PCM audio at the analyzer's sample rate.
            is_speech: Whether the VAD currently reports the user speaking.

        Returns:
            COMPLETE if the turn ended with this audio (the model's score
            reached the threshold during silence, or the silence lasted
            ``stop_secs``), otherwise INCOMPLETE.
        """
        audio_int16 = np.frombuffer(buffer, dtype=np.int16)
        if len(audio_int16) == 0:
            return EndOfTurnState.INCOMPLETE
        # A COMPLETE returned by the previous call is only valid until more
        # audio arrives. Pipecat 0.0.x never asks for it, and it must not end
        # the next turn at its first pause.
        self._pending_complete = False
        self._pending_metrics = None
        audio = self._to_model_rate(audio_int16.astype(np.float32) / 32768.0)

        if is_speech:
            if not self._speech_triggered:
                self._start_turn()
            self._silence_ms = 0.0
            self._speech_stopped_time = None
            self._score(audio)
            return EndOfTurnState.INCOMPLETE

        if not self._speech_triggered:
            self._remember_pre_speech(audio)
            return EndOfTurnState.INCOMPLETE

        if self._speech_stopped_time is None:
            self._speech_stopped_time = time.perf_counter()
        input_rate = self._sample_rate or MODEL_SAMPLE_RATE
        self._silence_ms += len(audio_int16) / (input_rate / 1000)

        probability = self._score(audio)
        if probability is not None and probability >= self._params.threshold:
            logger.debug(f"End of Turn complete, probability: {probability:.4f}")
            self._complete(probability)
            return EndOfTurnState.COMPLETE

        if self._silence_ms >= self._params.stop_secs * 1000:
            logger.debug(
                f"End of Turn complete due to stop_secs. Silence in ms: {self._silence_ms}"
            )
            self._complete(None)
            return EndOfTurnState.COMPLETE

        return EndOfTurnState.INCOMPLETE

    async def analyze_end_of_turn(self) -> tuple[EndOfTurnState, MetricsData | None]:
        """Analyze the current audio state to determine if the turn has ended.

        Called when the VAD reports that the user stopped speaking: reads the
        model's score at this moment, without waiting for the next 160 ms step
        boundary. Also called right after ``append_audio`` returned COMPLETE,
        in which case that decision and its metrics are returned.

        Returns:
            Tuple containing the end-of-turn state and optional metrics data
            from the model.
        """
        if self._pending_complete:
            metrics = self._pending_metrics
            self._pending_complete = False
            self._pending_metrics = None
            return EndOfTurnState.COMPLETE, metrics

        if not self._speech_triggered:
            return EndOfTurnState.INCOMPLETE, None

        if self._speech_stopped_time is None:
            self._speech_stopped_time = time.perf_counter()

        probability = self._model.peek()
        self._last_probability = probability
        is_complete = probability >= self._params.threshold
        metrics = self._metrics(is_complete, probability)
        logger.debug(
            f"End of Turn result: {'COMPLETE' if is_complete else 'INCOMPLETE'}, "
            f"probability: {probability:.4f}"
        )
        if is_complete:
            self._reset_turn()
            return EndOfTurnState.COMPLETE, metrics
        return EndOfTurnState.INCOMPLETE, metrics

    def clear(self):
        """Reset the turn analyzer to its initial state."""
        self._reset_turn()
        self._pending_complete = False
        self._pending_metrics = None

    def _to_model_rate(self, audio: np.ndarray) -> np.ndarray:
        """Resample a chunk of the incoming stream to 16 kHz if needed."""
        rate = self._sample_rate or MODEL_SAMPLE_RATE
        if rate == MODEL_SAMPLE_RATE:
            return audio
        if self._resampler is None or self._resampler_rate != rate:
            self._resampler = soxr.ResampleStream(
                rate, MODEL_SAMPLE_RATE, 1, dtype="float32", quality="HQ"
            )
            self._resampler_rate = rate
        return self._resampler.resample_chunk(audio)

    def _remember_pre_speech(self, audio: np.ndarray):
        """Keep the most recent audio heard while nobody is speaking."""
        self._pre_speech.append(audio)
        self._pre_speech_samples += len(audio)
        keep = int((self._params.pre_speech_ms / 1000 + self._vad_start_secs) * MODEL_SAMPLE_RATE)
        while self._pre_speech and self._pre_speech_samples - len(self._pre_speech[0]) >= keep:
            self._pre_speech_samples -= len(self._pre_speech.popleft())

    def _start_turn(self):
        """Start a new model stream, seeded with the pre-speech audio."""
        logger.trace("Speech detected, turn analysis started")
        self._speech_triggered = True
        self._model.reset()
        self._last_probability = None
        if self._pre_speech:
            self._score(np.concatenate(self._pre_speech))
        self._pre_speech.clear()
        self._pre_speech_samples = 0

    def _score(self, audio: np.ndarray) -> float | None:
        """Push audio into the model. Returns the newest score, if a step finished."""
        probs = self._model.push(audio)
        if len(probs) == 0:
            return None
        self._last_probability = float(probs[-1])
        return self._last_probability

    def _metrics(self, is_complete: bool, probability: float) -> TurnMetricsData:
        stopped = self._speech_stopped_time
        elapsed_ms = (time.perf_counter() - stopped) * 1000 if stopped is not None else 0.0
        return TurnMetricsData(
            processor="LocalTurn1MiniAnalyzer",
            is_complete=is_complete,
            probability=probability,
            e2e_processing_time_ms=elapsed_ms,
        )

    def _complete(self, probability: float | None):
        """Record a COMPLETE decided in append_audio and reset for the next turn."""
        self._pending_complete = True
        # A turn ended by the silence timeout carries no model decision.
        self._pending_metrics = (
            self._metrics(True, probability) if probability is not None else None
        )
        self._reset_turn()

    def _reset_turn(self):
        """Drop all per-turn state. The next speech starts a new model stream."""
        self._speech_triggered = False
        self._silence_ms = 0.0
        self._speech_stopped_time = None
        self._pre_speech.clear()
        self._pre_speech_samples = 0
        self._model.reset()
