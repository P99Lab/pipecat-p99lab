#
# Copyright (c) 2026, p99lab
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""ONNX streaming runner for the turn-1-mini end-of-turn model.

The published graph computes one 160 ms step: it takes 2,560 samples of
16 kHz mono audio plus the state of the previous step and returns eight
values of P(end of turn), one per 20 ms frame, plus the new state. The 4 kHz
low-pass and the log-mel features are inside the graph, so this module needs
only ``numpy`` and ``onnxruntime``.

Interface reference: https://huggingface.co/p99lab/turn-1-mini/blob/main/onnx/README.md
"""

import numpy as np
from loguru import logger

MODEL_SAMPLE_RATE = 16000
STEP_SAMPLES = 2560  # 160 ms at 16 kHz
FRAME_SAMPLES = 320  # 20 ms at 16 kHz
FRAMES_PER_STEP = STEP_SAMPLES // FRAME_SAMPLES

DEFAULT_REPO_ID = "p99lab/turn-1-mini"
# The Hub commit this package was tested against. Pass ``revision="main"`` to
# follow the repository instead.
DEFAULT_REVISION = "22bcc75c9857d664ede4811cf3dfaa46be7c4376"

_STATE_NAMES = ("audio", "conv1", "conv2", "k", "v", "k_valid", "h6")
_VARIANT_FILES = {
    "float32": "onnx/turn-1-mini.step.onnx",
    "int8": "onnx/turn-1-mini.step.int8dyn.onnx",
}
_VARIANT_SIZES = {"float32": "22 MB", "int8": "7 MB"}


def resolve_model_path(
    *,
    variant: str = "float32",
    repo_id: str = DEFAULT_REPO_ID,
    revision: str | None = DEFAULT_REVISION,
) -> str:
    """Return a local path to a turn-1-mini ONNX graph, downloading it once.

    The Hugging Face cache is checked first without touching the network. The
    graph is downloaded only when it is not in the cache yet.

    Args:
        variant: ``"float32"`` (21.8 MB, the benchmarked weights) or
            ``"int8"`` (7.0 MB, 8-bit weights, faster).
        repo_id: Hugging Face repository that holds the graphs.
        revision: Branch, tag or commit of the repository.

    Returns:
        Path of the ONNX file on the local disk.

    Raises:
        ValueError: If ``variant`` is not a known variant.
        RuntimeError: If the graph is not cached and cannot be downloaded.
    """
    if variant not in _VARIANT_FILES:
        raise ValueError(f"Unknown variant {variant!r}; use one of {sorted(_VARIANT_FILES)}")

    from huggingface_hub import hf_hub_download

    filename, size = _VARIANT_FILES[variant], _VARIANT_SIZES[variant]
    try:
        return hf_hub_download(
            repo_id=repo_id, filename=filename, revision=revision, local_files_only=True
        )
    except Exception:
        pass

    logger.info(
        f"Downloading turn-1-mini ({size}) from https://huggingface.co/{repo_id}. "
        "This happens once; later runs use the local cache and work offline."
    )
    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
    except Exception as e:
        raise RuntimeError(
            f"turn-1-mini is not in the local Hugging Face cache and could not be downloaded "
            f"from {repo_id} ({e}). Connect to the network once, or pass model_path= with a "
            f"local copy of {filename.split('/')[-1]}."
        ) from e
    logger.info(f"turn-1-mini saved to {path}")
    return path


class Turn1MiniStream:
    """One audio stream scored by the turn-1-mini ONNX graph.

    Feed 16 kHz mono float32 audio of any length with :meth:`push`; each
    finished 160 ms step yields eight probabilities. The state tensors start
    as zeros and are fed back unchanged after every step, as the graph
    requires. Use one instance per audio stream.
    """

    def __init__(self, model_path: str, *, cpu_count: int = 1):
        """Open the ONNX graph.

        Args:
            model_path: Path to one of the ``turn-1-mini.step*.onnx`` files.
            cpu_count: Number of CPU threads for inference. One is enough for
                a model this small.
        """
        try:
            import onnxruntime as ort
        except ModuleNotFoundError as e:
            raise ImportError(
                "LocalTurn1MiniAnalyzer needs onnxruntime to run turn-1-mini. "
                "Install it with: pip install onnxruntime"
            ) from e

        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = cpu_count
        self._session = ort.InferenceSession(
            str(model_path), sess_options=so, providers=["CPUExecutionProvider"]
        )
        self._state_shapes = {
            i.name: tuple(i.shape)
            for i in self._session.get_inputs()
            if i.name.startswith("state_")
        }
        self._outputs = ["probs"] + [f"new_state_{name}" for name in _STATE_NAMES]
        self._state: dict[str, np.ndarray] = {}
        self._pending = np.zeros(0, np.float32)
        self._last = 0.0
        self.reset()

    @property
    def last_probability(self) -> float:
        """P(end of turn) of the most recent finished frame (0.0 after a reset)."""
        return self._last

    def reset(self):
        """Start a new stream: zero state, no pending audio."""
        self._state = {
            name: np.zeros(shape, np.float32) for name, shape in self._state_shapes.items()
        }
        self._pending = np.zeros(0, np.float32)
        self._last = 0.0

    def _run(self, audio: np.ndarray) -> list[np.ndarray]:
        feed = {"audio": audio.reshape(1, STEP_SAMPLES).astype(np.float32, copy=False)}
        return self._session.run(self._outputs, {**feed, **self._state})

    def push(self, chunk: np.ndarray) -> np.ndarray:
        """Append audio and run every 160 ms step that is now complete.

        Args:
            chunk: 16 kHz mono float32 audio in [-1, 1], any length.

        Returns:
            One P(end of turn) per finished 20 ms frame, oldest first. Empty
            until 160 ms of audio have accumulated.
        """
        # Bad samples (NaN, infinity, values far outside [-1, 1]) would otherwise
        # enter the cached state and corrupt every later step of the stream.
        samples = np.nan_to_num(np.asarray(chunk, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(samples, -1.0, 1.0, out=samples)
        self._pending = np.concatenate([self._pending, samples])
        out = []
        while len(self._pending) >= STEP_SAMPLES:
            result = self._run(self._pending[:STEP_SAMPLES])
            self._pending = self._pending[STEP_SAMPLES:]
            if not all(np.isfinite(r).all() for r in result):
                # Should not happen after the clean-up above; if it does, start
                # over from a zero state rather than emit garbage.
                pending = self._pending
                self.reset()
                self._pending = pending
                out.append(np.zeros(FRAMES_PER_STEP, np.float32))
                continue
            self._state = {f"state_{name}": r for name, r in zip(_STATE_NAMES, result[1:])}
            self._last = float(result[0][-1])
            out.append(result[0])
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    def peek(self) -> float:
        """P(end of turn) right now, without waiting for the next step boundary.

        Audio that has not filled a step yet is padded with zeros and run on a
        copy of the state; the stream itself is left untouched, so the real
        step still runs when the rest of the chunk arrives. The value returned
        is the one for the last frame made entirely of real audio.

        Returns:
            The current P(end of turn).
        """
        frames = len(self._pending) // FRAME_SAMPLES
        if frames == 0:
            return self._last
        padded = np.zeros(STEP_SAMPLES, np.float32)
        padded[: len(self._pending)] = self._pending
        # session.run does not modify its inputs, so the stream state is kept.
        return float(self._run(padded)[0][frames - 1])
