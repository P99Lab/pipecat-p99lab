# Changelog

All notable changes to `pipecat-p99lab` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.2] - 2026-10-06

### Changed

- Installs alongside Pipecat 0.0.108 and later (previously required 1.12). On 0.0.x, pass the analyzer as
  `turn_analyzer=` in the transport params. Tested with Pipecat 0.0.108 and 1.12.0.

## [0.1.1] - 2026-10-06

### Fixed

- Audio chunks containing NaN, infinity or samples far outside [-1, 1] no longer corrupt the stream's cached
  state: bad samples are zeroed or clipped before the model sees them.

## [0.1.0] - 2026-10-06

### Added

- `LocalTurn1MiniAnalyzer`: a Pipecat turn analyzer that runs the turn-1-mini
  ONNX model locally as a stream on the user's audio.
- `Turn1MiniParams`: `threshold`, `stop_secs` and `pre_speech_ms`, with three
  presets: `fast()` (0.2), `balanced()` (0.5, the default) and `patient()` (0.7).
- The float32 graph is the default; `variant="int8"` selects the 7 MB graph.
- Foundational example: `examples/foundational/turn-management-turn-1-mini.py`.
- Offline simulation that needs no API keys: `examples/offline_simulation.py`.
- Tested with Pipecat v1.12.0 and turn-1-mini revision `22bcc75`.
