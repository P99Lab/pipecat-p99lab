# pipecat-p99lab

End-of-turn detection for [Pipecat](https://github.com/pipecat-ai/pipecat) with
[turn-1-mini](https://huggingface.co/p99lab/turn-1-mini): a small model by p99lab that listens to the user's audio
and tells your bot when they have finished speaking. It runs locally on one CPU core, with no API key.

```bash
pip install git+https://github.com/P99Lab/pipecat-p99lab
```

```python
from pipecat_p99lab import LocalTurn1MiniAnalyzer
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

turn_strategies = UserTurnStrategies(
    stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalTurn1MiniAnalyzer())]
)
# pass it as LLMUserAggregatorParams(user_turn_strategies=turn_strategies, vad_analyzer=SileroVADAnalyzer())
```

That is all: the model (22 MB) downloads once on first use and then works offline. To trade speed against
interruptions, pick a preset, e.g. `LocalTurn1MiniAnalyzer(params=Turn1MiniParams.patient())`:

| Preset | Threshold | Ends the turn in a mid-turn pause | Delay after the user stops: median / 90th percentile |
|---|---:|---:|---:|
| `Turn1MiniParams.fast()` | 0.2 | 16.4% | 200 ms / 200 ms |
| `Turn1MiniParams.balanced()` (default) | 0.5 | 11.3% | 200 ms / 750 ms |
| `Turn1MiniParams.patient()` | 0.7 | 7.9% | 200 ms / 3 s (one turn in ten waits for the timeout) |

Measured by p99lab on its own development audio (English phone-call turns: 608 turns, 379 mid-turn pauses), in a
Pipecat pipeline with Silero VAD; the delay includes the VAD's 200 ms. English only. Your audio will differ:
see [Choosing a threshold](#choosing-a-threshold) and [Limits](#limits).

---

## What it is

turn-1-mini is a 5.3M-parameter causal streaming model, audio only (no transcript needed), that outputs the
probability that the user has finished their turn. This package wraps it as `LocalTurn1MiniAnalyzer`, a Pipecat turn
analyzer you can use wherever Pipecat accepts one. It is an additional option next to the turn analyzers that ship
with Pipecat, with the same role and the same wiring.

On LiveKit's public eot-bench (English), turn-1-mini scores 22.6% false cut-offs at a 300 ms budget and 849 ms latency
at 5% false cut-offs, ahead of the downloadable models on that board including Smart Turn v3.2 (35.2%, 1,051 ms), on
point estimates, from p99lab's own runs. Details, intervals and limits are on the
[model card](https://huggingface.co/p99lab/turn-1-mini).

**Maintained by p99lab**, the company that makes turn-1-mini. Contact: research@p99lab.com

## Pipecat compatibility

Tested with **Pipecat v1.12.0** and **v0.0.108**, Python 3.12, turn-1-mini revision `22bcc75`. Versions in between
are expected to work and have not been tested.

On Pipecat 0.0.x, where turn analyzers are given to the transport, pass it there instead:

```python
params = FastAPIWebsocketParams(          # or any other TransportParams
    vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
    turn_analyzer=LocalTurn1MiniAnalyzer(),
    ...
)
```

Use a short VAD `stop_secs` (about 0.2) with a turn analyzer: the VAD only proposes the moment, the model decides.
8 kHz telephone audio (for example Twilio) is supported and is resampled internally.

The integration uses Pipecat's `BaseTurnAnalyzer` interface and `TurnAnalyzerUserTurnStopStrategy`.

## Installation

```bash
pip install git+https://github.com/P99Lab/pipecat-p99lab
```

or from source:

```bash
git clone https://github.com/p99lab/pipecat-p99lab
pip install ./pipecat-p99lab
```

No PyTorch and no GPU are needed. The first time an analyzer is created, the model (22 MB) is downloaded from
Hugging Face into the local Hugging Face cache, with a log line saying so. After that nothing touches the network:
the cache is checked first, and no audio ever leaves the machine. To avoid the download altogether, pass
`model_path=` with a local copy of the ONNX file.

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
                    turn_analyzer=LocalTurn1MiniAnalyzer(params=Turn1MiniParams.balanced())
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

`Turn1MiniParams` (start from a preset, or set the fields yourself; presets accept overrides, e.g.
`Turn1MiniParams.fast(stop_secs=2.0)`):

| Parameter | Default | Meaning |
|---|---:|---|
| `threshold` | `0.5` | P(end of turn) at or above which the turn is complete. Lower answers sooner and cuts in more often |
| `stop_secs` | `3.0` | Safety timeout: silence after which the turn ends whatever the model says |
| `pre_speech_ms` | `500` | Audio from before the detected start of speech that the model gets as context |
| `fallback_secs` | `None` (off) | Earlier, softer timeout: after this much silence the turn ends if the highest score in that silence reached `fallback_threshold`. See [Long waits after an ending](#long-waits-after-an-ending) |
| `fallback_threshold` | `0.1` | The score the fallback requires |

`LocalTurn1MiniAnalyzer(...)`:

| Argument | Default | Meaning |
|---|---|---|
| `params` | `Turn1MiniParams()` | See above |
| `variant` | `"float32"` | `"float32"`: 21.8 MB graph, the weights behind the published benchmark numbers. `"int8"`: 7.0 MB graph with 8-bit weights, smaller and somewhat faster |
| `model_path` | `None` | Path to a local `turn-1-mini.step*.onnx` file. Skips the Hugging Face cache and download |
| `revision` | tested commit | Revision of `p99lab/turn-1-mini` to load. Pass `"main"` to follow the repository |
| `repo_id` | `"p99lab/turn-1-mini"` | Hugging Face repository |
| `cpu_count` | `1` | CPU threads for inference. One is enough |
| `sample_rate` | `None` | Fix the input sample rate. By default the pipeline's input rate is used |

Input at 8, 16, 24 or 48 kHz (or any other rate) just works: anything other than 16 kHz is resampled with a
streaming resampler. Do not put a low-pass filter in front: the model's own input filter is inside the ONNX graph.

## Choosing a threshold

The table below is the full sweep behind the presets: each turn was played in 20 ms frames through Pipecat's Silero
VAD (default settings) and this analyzer with `stop_secs=3`, exactly as a pipeline would. "Wrong endings" is the share
of mid-turn pauses (0.2 to 5 s long) in which the turn was ended. The delay runs from the end of speech to the
end-of-turn decision and includes the 200 ms the VAD takes to report that the speech stopped.

| Threshold | Wrong endings | Delay, median | Delay, 90th percentile | Turns ended by the timeout |
|---:|---:|---:|---:|---:|
| 0.1 | 20.8% | 200 ms | 200 ms | 0.5% |
| 0.2 (`fast`) | 16.4% | 200 ms | 200 ms | 0.9% |
| 0.3 | 15.0% | 200 ms | 320 ms | 2.1% |
| 0.4 | 12.4% | 200 ms | 444 ms | 2.7% |
| 0.5 (`balanced`) | 11.3% | 200 ms | 746 ms | 3.6% |
| 0.6 | 10.3% | 200 ms | 1,142 ms | 6.2% |
| 0.7 (`patient`) | 7.9% | 200 ms | 3,200 ms | 10.2% |

What to take from it:

- These are p99lab's own measurements on its own development audio (English phone-call turns: 608 turns, 379
  mid-turn pauses; a few hundred pauses, so each rate is uncertain by a few points). They are not a benchmark result.
- No threshold got wrong endings down to 5% on this audio: 2 to 3 points of every row are pauses that outlast the
  3 s `stop_secs` timeout, and the model scores many long pauses as ends. If interruptions are costly in your product,
  use `patient()` and consider a longer `stop_secs`.
- On a harder slice (multi-party meeting speech) wrong endings were roughly twice as high at `fast` and `balanced`,
  and at `patient` most turns ended by the timeout.
- The scores are not calibrated. Run [`examples/offline_simulation.py`](examples/offline_simulation.py) on recordings
  of your own users to see where the scores fall, then choose.

## How it decides

- While the VAD reports speech, audio is streamed through the model and nothing is decided.
- When the VAD reports that the user stopped, the analyzer reads the model's score at that moment (without waiting
  for the model's next 160 ms step) and ends the turn if it is at or above `threshold`.
- If not, it keeps streaming the silence and checks the newest score after every 160 ms step.
- If the user starts speaking again, the turn simply continues.
- If the score never reaches the threshold, the turn ends after `stop_secs` of silence, or earlier by the fallback
  if `fallback_secs` is set.

### Long waits after an ending

The model scores some real endings below the threshold, most often short answers such as a name or a number. Those
turns then wait for the `stop_secs` timeout. Mid-sentence pauses usually score far lower (near zero), which the
fallback uses: `Turn1MiniParams(fallback_secs=1.3)` ends the turn after 1.3 s of silence if the score reached 0.1 at
any point in that silence, and leaves pauses scored below 0.1 to `stop_secs`.

Measured at the default threshold, same method as the table above (silence counted from the VAD's stop):

| Audio | | Wrong endings | Turns ended by the 3 s timeout | Delay p90 |
|---|---|---:|---:|---:|
| Phone-call turns | off | 11.3% | 3.6% | 746 ms |
| | `fallback_secs=1.3` | 13.5% | 0.5% | 746 ms |
| Short telephone answers (names, numbers, postcodes) | off | 14.5% | 24.7% | 3,200 ms |
| | `fallback_secs=1.3` | 16.5% | 7.6% | 1,620 ms |

It is off by default because it adds about 2 points of wrong endings. Turn it on if your users give short answers
and silence after them costs more than an occasional early reply.

The VAD's own `stop_secs` (0.2 s by default in Pipecat) is therefore the shortest silence after which a turn can end.

**Live stream or re-scoring?** turn-1-mini was benchmarked by scoring the audio of the turn so far from a zero
state, at any moment of a pause. This integration gets the same scores with a live stream: at the start of each turn
the model state is reset to zero and the stream is seeded with the `pre_speech_ms` of audio before the speech, so the
score at any moment is the score of the turn's audio prefix up to that moment, at the constant cost of streaming
instead of re-running the whole prefix at every check. State never carries over from one turn to the next.

## Limits

- **English only.** turn-1-mini has been measured on English only; other languages are untested.
- **Scores are not calibrated, and the preset figures come from p99lab's own development audio.** Calibrate for your
  audio and users, and keep `stop_secs` as a fallback.
- The model ends the turn in a share of mid-turn pauses at every threshold (see the table above), most often in long
  pauses. Audio only: it does not read words, so it can end a turn inside a spoken number, a spelled-out word or an
  email address.
- The `"int8"` graph has not itself been run on eot-bench; the published numbers are for the float32 weights, which
  are the default here.
- Not evaluated with overlapping speech, far-field audio or agent echo.
- Inference runs inside the audio path: about 1 to 2 ms per 160 ms of audio on one CPU core (our development
  machine; measure on yours).

See the [model card](https://huggingface.co/p99lab/turn-1-mini) for the full evaluation and limits.

## Tests

```bash
pip install ".[test]"
pytest -m "not network"     # unit tests: no network, no keys, model faked
pytest                      # also loads the real model from Hugging Face (or the local cache)
```

## Licence

This integration is released under the **BSD 2-Clause License**, the same licence as Pipecat (see [LICENSE](LICENSE)).
The turn-1-mini model it downloads is released separately by p99lab under Apache-2.0.

## Maintainer

p99lab, research@p99lab.com. Issues and questions are welcome by email or on this repository's issue tracker.
See [CHANGELOG.md](CHANGELOG.md) for version history.
