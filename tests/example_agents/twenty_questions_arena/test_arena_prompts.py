"""Tests for 20 Questions Arena game prompts.

Tests cover:
- generate_thinker_prompt() returns valid thinker instructions
- generate_guesser_prompt() returns valid guesser instructions
- create_llm() helper selects the right LLM based on env vars
"""

from __future__ import annotations

import os
from unittest.mock import patch


class TestThinkerPrompt:
    """Test generate_thinker_prompt() content."""

    def test_returns_nonempty_string(self):
        from prompts import generate_thinker_prompt

        prompt = generate_thinker_prompt("Thinker")
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_includes_agent_name(self):
        from prompts import generate_thinker_prompt

        prompt = generate_thinker_prompt("Alice")
        assert "Alice" in prompt

    def test_includes_secret_word_instructions(self):
        from prompts import generate_thinker_prompt

        prompt = generate_thinker_prompt("Thinker")
        assert "secret" in prompt.lower()

    def test_includes_yes_no_constraint(self):
        from prompts import generate_thinker_prompt

        prompt = generate_thinker_prompt("Thinker")
        lower = prompt.lower()
        assert "yes" in lower
        assert "no" in lower

    def test_includes_20_questions_limit(self):
        from prompts import generate_thinker_prompt

        prompt = generate_thinker_prompt("Thinker")
        assert "20" in prompt

    def test_includes_category_references(self):
        from prompts import generate_thinker_prompt

        prompt = generate_thinker_prompt("Thinker")
        lower = prompt.lower()
        assert "category" in lower or "animal" in lower or "object" in lower

    def test_includes_lookup_peers_instruction(self):
        from prompts import generate_thinker_prompt

        prompt = generate_thinker_prompt("Thinker")
        assert "thenvoi_lookup_peers" in prompt

    def test_includes_add_participant_instruction(self):
        from prompts import generate_thinker_prompt

        prompt = generate_thinker_prompt("Thinker")
        assert "thenvoi_add_participant" in prompt

    def test_includes_thought_event_instruction(self):
        from prompts import generate_thinker_prompt

        prompt = generate_thinker_prompt("Thinker")
        assert "thenvoi_send_event" in prompt

    def test_must_not_reveal_word(self):
        from prompts import generate_thinker_prompt

        prompt = generate_thinker_prompt("Thinker")
        lower = prompt.lower()
        assert "reveal" in lower or "never" in lower or "do not" in lower


class TestGuesserPrompt:
    """Test generate_guesser_prompt() content."""

    def test_returns_nonempty_string(self):
        from prompts import generate_guesser_prompt

        prompt = generate_guesser_prompt("Guesser")
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_includes_agent_name(self):
        from prompts import generate_guesser_prompt

        prompt = generate_guesser_prompt("Bob")
        assert "Bob" in prompt

    def test_includes_question_strategy(self):
        from prompts import generate_guesser_prompt

        prompt = generate_guesser_prompt("Guesser")
        lower = prompt.lower()
        assert "question" in lower

    def test_includes_deductive_reasoning(self):
        from prompts import generate_guesser_prompt

        prompt = generate_guesser_prompt("Guesser")
        lower = prompt.lower()
        assert "narrow" in lower or "dedu" in lower or "eliminat" in lower

    def test_includes_20_questions_limit(self):
        from prompts import generate_guesser_prompt

        prompt = generate_guesser_prompt("Guesser")
        assert "20" in prompt

    def test_includes_guess_format(self):
        from prompts import generate_guesser_prompt

        prompt = generate_guesser_prompt("Guesser")
        lower = prompt.lower()
        assert "is it" in lower or "guess" in lower

    def test_includes_thought_event_instruction(self):
        from prompts import generate_guesser_prompt

        prompt = generate_guesser_prompt("Guesser")
        assert "thenvoi_send_event" in prompt


class TestCreateLlmByName:
    """Test create_llm_by_name() selects the correct provider by model name."""

    def test_openai_model(self):
        from prompts import create_llm_by_name

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False):
            llm = create_llm_by_name("gpt-5.2")
            assert "ChatOpenAI" in type(llm).__name__

    def test_anthropic_model(self):
        import sys
        from types import ModuleType

        from prompts import create_llm_by_name

        class FakeChatAnthropic:
            def __init__(self, **kwargs):
                self.model = kwargs.get("model")

        fake_module = ModuleType("langchain_anthropic")
        fake_module.ChatAnthropic = FakeChatAnthropic  # type: ignore[attr-defined]

        with (
            patch.dict(
                os.environ,
                {"ANTHROPIC_API_KEY": "sk-ant-test"},
                clear=False,
            ),
            patch.dict(sys.modules, {"langchain_anthropic": fake_module}),
        ):
            llm = create_llm_by_name("claude-opus-4-6")
            assert "ChatAnthropic" in type(llm).__name__

    def test_missing_openai_key_raises(self):
        import pytest

        from prompts import create_llm_by_name

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                create_llm_by_name("gpt-5.2")

    def test_missing_anthropic_key_raises(self):
        import pytest

        from prompts import create_llm_by_name

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                create_llm_by_name("claude-sonnet-4-6")


class TestThinkerMultiGuesserPrompt:
    """Test that the thinker prompt includes multi-guesser support."""

    def test_parallel_gameplay_section_exists(self):
        from prompts import generate_thinker_prompt

        prompt = generate_thinker_prompt("Thinker")
        assert "Parallel Gameplay" in prompt

    def test_tagging_rules_present(self):
        from prompts import generate_thinker_prompt

        prompt = generate_thinker_prompt("Thinker")
        lower = prompt.lower()
        assert "tag all guessers" in lower or "tag all" in lower

    def test_per_guesser_tracking(self):
        from prompts import generate_thinker_prompt

        prompt = generate_thinker_prompt("Thinker")
        lower = prompt.lower()
        assert "per guesser" in lower or "per-guesser" in lower or "separately" in lower

    def test_no_information_leaking(self):
        from prompts import generate_thinker_prompt

        prompt = generate_thinker_prompt("Thinker")
        lower = prompt.lower()
        assert "leak" in lower or "private" in lower

    def test_independent_outcomes(self):
        from prompts import generate_thinker_prompt

        prompt = generate_thinker_prompt("Thinker")
        lower = prompt.lower()
        assert "independent" in lower


class TestGuesserIsolationPrompt:
    """Test that the guesser prompt includes multi-guesser isolation rules."""

    def test_never_tag_other_guessers(self):
        from prompts import generate_guesser_prompt

        prompt = generate_guesser_prompt("Guesser")
        lower = prompt.lower()
        assert "never tag" in lower or "never mention other guessers" in lower

    def test_ignore_other_guessers(self):
        from prompts import generate_guesser_prompt

        prompt = generate_guesser_prompt("Guesser")
        lower = prompt.lower()
        assert "ignore" in lower

    def test_isolation_section_exists(self):
        from prompts import generate_guesser_prompt

        prompt = generate_guesser_prompt("Guesser")
        assert "Multi-Guesser Isolation" in prompt

    def test_critical_rule_6(self):
        from prompts import generate_guesser_prompt

        prompt = generate_guesser_prompt("Guesser")
        assert "NEVER tag or mention other guessers" in prompt


class TestCreateLlm:
    """Test create_llm() helper selects LLM based on env vars."""

    def test_selects_openai_when_key_present(self):
        from prompts import create_llm

        remove = {k: "" for k in ("ANTHROPIC_API_KEY",) if k in os.environ}
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "sk-test-key", **remove},
            clear=False,
        ):
            # Remove keys set to empty so getenv returns None
            os.environ.pop("ANTHROPIC_API_KEY", None)
            llm = create_llm()
            assert "ChatOpenAI" in type(llm).__name__

    def test_selects_anthropic_when_key_present(self):
        import sys
        from types import ModuleType

        from prompts import create_llm

        # Create a fake ChatAnthropic class
        class FakeChatAnthropic:
            def __init__(self, **kwargs):
                pass

        fake_module = ModuleType("langchain_anthropic")
        fake_module.ChatAnthropic = FakeChatAnthropic  # type: ignore[attr-defined]

        with (
            patch.dict(
                os.environ,
                {"ANTHROPIC_API_KEY": "sk-ant-test-key"},
                clear=False,
            ),
            patch.dict(sys.modules, {"langchain_anthropic": fake_module}),
        ):
            os.environ.pop("OPENAI_API_KEY", None)
            llm = create_llm()
            assert "ChatAnthropic" in type(llm).__name__

    def test_raises_when_no_key(self):
        import pytest

        from prompts import create_llm

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("ANTHROPIC_API_KEY", None)
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY.*OPENAI_API_KEY"):
                create_llm()

    def test_prefers_anthropic_when_both_present(self):
        from prompts import create_llm

        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "sk-test", "ANTHROPIC_API_KEY": "sk-ant-test"},
            clear=False,
        ):
            llm = create_llm()
            assert "ChatAnthropic" in type(llm).__name__
