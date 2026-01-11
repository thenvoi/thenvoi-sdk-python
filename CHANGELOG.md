# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1](https://github.com/thenvoi/thenvoi-sdk-python/compare/thenvoi-sdk-python-v0.1.0...thenvoi-sdk-python-v0.1.1) (2026-01-08)


### Features

* add local GithubToken action and release workflow ([eeacd94](https://github.com/thenvoi/thenvoi-sdk-python/commit/eeacd94f454216ab814cd6ba19744171747e5482))
* **ci:** add changelog generation with semantic versioning ([66d0bbd](https://github.com/thenvoi/thenvoi-sdk-python/commit/66d0bbd169041206283814a9657e5b4b163931b6))


### Bug Fixes

* **ci:** move checkout before token generation ([088dbf5](https://github.com/thenvoi/thenvoi-sdk-python/commit/088dbf5ef94d28425804759b2cd5503422149f95))

## [Unreleased]

## [0.1.0] - 2024-01-01

### Added

- Initial SDK implementation for Thenvoi platform
- ThenvoiLink for WebSocket + REST transport
- AgentRuntime for convenient agent lifecycle management
- RoomPresence for cross-room lifecycle management
- ExecutionContext for per-room context accumulation
- AgentTools for platform tools (send_message, add_participant, etc.)
- Support for multiple AI frameworks (PydanticAI, Anthropic, LangGraph, Claude SDK)
- Comprehensive test suite with pytest

[Unreleased]: https://github.com/thenvoi/thenvoi-sdk-python/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/thenvoi/thenvoi-sdk-python/releases/tag/v0.1.0
