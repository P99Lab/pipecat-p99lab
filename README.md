# pipecat-p99lab

A [Pipecat](https://github.com/pipecat-ai/pipecat) community integration for
[turn-1-mini](https://huggingface.co/p99lab/turn-1-mini), an end-of-turn detection model by p99lab.

turn-1-mini is a 5.3M-parameter causal streaming model that listens to the user's audio (no transcript needed) and
outputs the probability that they have finished their turn. It is a 7 MB ONNX file, runs on one CPU core in about
1 ms per 160 ms of audio, and needs neither PyTorch nor a GPU.

This package provides `LocalTurn1MiniAnalyzer`, a Pipecat turn analyzer you can use wherever Pipecat accepts one. It
is an additional option next to the turn analyzers that ship with Pipecat, with the same role and the same wiring.

On LiveKit's public eot-bench (English), turn-1-mini scores 22.6% false cut-offs at a 300 ms budget and 849 ms latency
at 5% false cut-offs, ahead of the downloadable models on that board including Smart Turn v3.2 (35.2%, 1,051 ms), on
point estimates, from p99lab's own runs. Details, intervals and limits are on the
[model card](https://huggingface.co/p99lab/turn-1-mini).

**Maintained by p99lab**, the company that makes turn-1-mini. Contact: research@p99lab.com

## Pipecat compatibility

Tested with **Pipecat v1.12.0** (`pipecat-ai==1.12.0`), Python 3.12, turn-1-mini revision `22bcc75`.

The integration uses Pipecat's `BaseTurnAnalyzer` interface and `TurnAnalyzerUserTurnStopStrategy`.

## Installation

```bash
pip install pipecat-p99lab
```

or from source:

```bash
git clone https://github.com/p99lab/pipecat-p99lab
pip install ./pipecat-p99lab
```

The first time an analyzer is created, the 7 MB model is downloaded from Hugging Face into the local Hugging Face
cache. After that nothing touches the network: the cache is checked first, and no audio ever leaves the machine.
To avoid the download altogether, pass `model_path=` with a local copy of the ONNX file.

## Usage with a Pipecat pipeline

Give the analyzer to `TurnAnalyzerUserTurnStopStrategy`, together with a VAD, in the user aggregator's parameters:

```python
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from pipecat_p99lab import LocalTurn1MiniAnalyzer, Turn1MiniParams

context = LLMContext()
user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
    context,
    user_params=LLMUserAggregatorParams(
        user_turn_strategies=UserTurnStrategies(
            stop=[
                TurnAnalyzerUserTurnStopStrategy(
                    turn_analyzer=LocalTurn1MiniAnalyzer(
                        params=Turn1MiniParams(threshold=0.5, stop_secs=3.0)
                    )
                )
            ]
        ),
        vad_analyzer=SileroVADAnalyzer(),
    ),
)

pipeline = Pipeline(
    [
        transport.input(),
        stt,
        user_aggregator,
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
    ]
)
```

A VAD is required: it tells the analyzer when the user is speaking. Everything else in the pipeline stays as it is.
Each decision is published as Pipecat `TurnMetricsData` (probability and whether the turn was judged complete), so
`MetricsLogObserver` and your own observers can see it.

## Running the example

[`examples/foundational/turn-management-turn-1-mini.py`](examples/foundational/turn-management-turn-1-mini.py) is
Pipecat's local turn-detection example with the analyzer swapped in. Its STT, LLM and TTS services (Deepgram, OpenAI,
Cartesia) are placeholders: replace them with the services you use.

```bash
git clone https://github.com/p99lab/pipecat-p99lab
cd pipecat-p99lab
pip install ".[example]"
cp env.example .env        # then put your own keys in .env
python examples/foundational/turn-management-turn-1-mini.py
```

Open http://localhost:7860 and talk to the bot. Turn decisions are logged with their probability.

To try the analyzer without any keys or a microphone, play a recording through it:

```bash
python examples/offline_simulation.py your_speech.wav --trace
```

This cuts the file into 20 ms frames, runs Pipecat's Silero VAD and `TurnAnalyzerUserTurnStopStrategy` on them, and
prints when the end of the turn is declared.

## Configuration

`Turn1MiniParams`:

| Parameter | Default | Meaning |
|---|---:|---|
| `threshold` | `0.5` | P(end of turn) at or above which the turn is complete. Lower answers sooner and cuts in more often. **Calibrate it on your own audio** |
| `stop_secs` | `3.0` | Silence after which the turn ends whatever the model says |
| `pre_speech_ms` | `500` | Audio from before the detected start of speech that the model gets as context |

`LocalTurn1MiniAnalyzer(...)`:

| Argument | Default | Meaning |
|---|---|---|
| `params` | `Turn1MiniParams()` | See above |
| `variant` | `"int8"` | `"int8"`: 7.0 MB graph with 8-bit weights, the fastest. `"float32"`: 21.8 MB graph |
| `model_path` | `None` | Path to a local `turn-1-mini.step*.onnx` file. Skips the Hugging Face cache and download |
| `revision` | tested commit | Revision of `p99lab/turn-1-mini` to load. Pass `"main"` to follow the repository |
| `repo_id` | `"p99lab/turn-1-mini"` | Hugging Face repository |
| `cpu_count` | `1` | CPU threads for inference. One is enough |
| `sample_rate` | `None` | Fix the input sample rate. By default the pipeline's input rate is used |

Audio at any sample rate other than 16 kHz is resampled to 16 kHz with a streaming resampler. Do not put a
low-pass filter in front: the model's own input filter is inside the ONNX graph.

## How it decides

- While the VAD reports speech, audio is streamed through the model and nothing is decided.
- When the VAD reports that the user stopped, the analyzer reads the model's score at that moment (without waiting
  for the model's next 160 ms step) and ends the turn if it is at or above `threshold`.
- If not, it keeps streaming the silence and checks the newest score after every 160 ms step.
- If the user starts speaking again, the turn simply continues.
- If the score never reaches the threshold, the turn ends after `stop_secs` of silence.

The VAD's own `stop_secs` (0.2 s by default in Pipecat) is therefore the shortest silence after which a turn can end.

**Live stream or re-scoring?** turn-1-mini was benchmarked by scoring the audio of the turn so far from a zero
state, at any moment of a pause. This integration gets the same scores with a live stream: at the start of each turn
the model state is reset to zero and the stream is seeded with the `pre_speech_ms` of audio before the speech, so the
score at any moment is the score of the turn's audio prefix up to that moment, at the constant cost of streaming
instead of re-running the whole prefix at every check. State never carries over from one turn to the next.

## Limits

- **English only.** turn-1-mini has been measured on English only; other languages are untested.
- **Scores are not calibrated.** `0.5` is a cautious default, not a tuned one. The model's eot-bench operating points
  used thresholds between 0.06 and 0.21 (0.06 at the 300 ms budget). Calibrate `threshold` on your own audio and
  users, and keep `stop_secs` as a fallback.
- The default `"int8"` graph has not itself been run on eot-bench; the published numbers are for the float32 weights.
  Use `variant="float32"` to stay closest to them.
- Audio only: the model does not read words, so it can end a turn inside a spoken number, a spelled-out word or an
  email address.
- Not evaluated with overlapping speech, far-field audio or agent echo.
- Inference runs inside the audio path (about 1 ms per 160 ms step for the default graph on one CPU core).

See the [model card](https://huggingface.co/p99lab/turn-1-mini) for the full evaluation and limits.

## Tests

```bash
pip install ".[test]"
pytest -m "not network"     # unit tests: no network, no keys, model faked
pytest                      # also loads the real 7 MB model from Hugging Face (or the local cache)
```

## Licence

This integration is released under the **BSD 2-Clause License**, the same licence as Pipecat (see [LICENSE](LICENSE)).
The turn-1-mini model it downloads is released separately by p99lab under Apache-2.0.

## Maintainer

p99lab, research@p99lab.com. Issues and questions are welcome by email or on this repository's issue tracker.
See [CHANGELOG.md](CHANGELOG.md) for version history.
