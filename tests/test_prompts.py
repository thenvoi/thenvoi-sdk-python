"""
Tests for system prompt rendering.

Tests cover:
- render_system_prompt() with include_base_instructions parameter
"""

from thenvoi.core.prompts import render_system_prompt, BASE_INSTRUCTIONS


class TestRenderSystemPromptDefaults:
    """Test render_system_prompt() default behavior."""

    def test_includes_base_instructions_by_default(self):
        """Default should include BASE_INSTRUCTIONS."""
        prompt = render_system_prompt(
            agent_name="TestBot",
            agent_description="A test agent",
        )

        assert "## Environment" in prompt
        assert "Multi-participant chat" in prompt
        assert "send_event" in prompt

    def test_includes_agent_identity(self):
        """Should include agent name and description."""
        prompt = render_system_prompt(
            agent_name="MyAgent",
            agent_description="A helpful assistant",
        )

        assert "MyAgent" in prompt
        assert "A helpful assistant" in prompt

    def test_includes_custom_section(self):
        """Should include custom section."""
        prompt = render_system_prompt(
            agent_name="TestBot",
            custom_section="Focus on Python programming.",
        )

        assert "Focus on Python programming." in prompt


class TestRenderSystemPromptWithoutBaseInstructions:
    """Test render_system_prompt() with include_base_instructions=False."""

    def test_excludes_base_instructions_when_disabled(self):
        """Should NOT include BASE_INSTRUCTIONS when disabled."""
        prompt = render_system_prompt(
            agent_name="TestBot",
            agent_description="A test agent",
            include_base_instructions=False,
        )

        # Should NOT have the opinionated instructions
        assert "## Environment" not in prompt
        assert "CRITICAL: Always Relay Information" not in prompt
        assert "send_event" not in prompt
        assert "thought" not in prompt

    def test_still_includes_agent_identity_when_disabled(self):
        """Should still include agent name and description."""
        prompt = render_system_prompt(
            agent_name="MinimalBot",
            agent_description="A minimal agent",
            include_base_instructions=False,
        )

        assert "MinimalBot" in prompt
        assert "A minimal agent" in prompt

    def test_still_includes_custom_section_when_disabled(self):
        """Should still include custom section."""
        prompt = render_system_prompt(
            agent_name="TestBot",
            agent_description="Test",
            custom_section="Custom behavior instructions here.",
            include_base_instructions=False,
        )

        assert "Custom behavior instructions here." in prompt

    def test_minimal_prompt_format(self):
        """Minimal prompt should have expected format."""
        prompt = render_system_prompt(
            agent_name="Bot",
            agent_description="helper",
            custom_section="Be helpful.",
            include_base_instructions=False,
        )

        # Should start with "You are" pattern
        assert prompt.startswith("You are Bot, helper.")
        assert "Be helpful." in prompt

    def test_empty_custom_section_when_disabled(self):
        """Should handle empty custom section gracefully."""
        prompt = render_system_prompt(
            agent_name="Bot",
            agent_description="helper",
            custom_section="",
            include_base_instructions=False,
        )

        # Should still work, just with empty section
        assert "Bot" in prompt
        assert "helper" in prompt


class TestBaseInstructionsConstant:
    """Test BASE_INSTRUCTIONS constant."""

    def test_base_instructions_exported(self):
        """BASE_INSTRUCTIONS should be importable."""
        assert BASE_INSTRUCTIONS is not None
        assert len(BASE_INSTRUCTIONS) > 0

    def test_base_instructions_has_key_sections(self):
        """BASE_INSTRUCTIONS should have expected sections."""
        assert "## Environment" in BASE_INSTRUCTIONS
        assert "## CRITICAL" in BASE_INSTRUCTIONS
        assert "## IMPORTANT" in BASE_INSTRUCTIONS
        assert "## Examples" in BASE_INSTRUCTIONS
