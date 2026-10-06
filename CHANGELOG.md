# Changelog

All notable changes to `pipecat-p99lab` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-06

### Added

- `LocalTurn1MiniAnalyzer`: a Pipecat turn analyzer that runs the turn-1-mini
  ONNX model locally as a stream on the user's audio.
- `Turn1MiniParams`: `threshold`, `stop_secs` and `pre_speech_ms`.
- Foundational example: `examples/foundational/turn-management-turn-1-mini.py`.
- Offline simulation that needs no API keys: `examples/offline_simulation.py`.
- Tested with Pipecat v1.12.0 and turn-1-mini revision `22bcc75`.
