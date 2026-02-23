"""Tests for role-based prompt profiles."""

from __future__ import annotations

import pytest

from thenvoi.prompts import AVAILABLE_ROLES, get_role_prompt


class TestGetRolePrompt:
    """Tests for get_role_prompt function."""

    def test_get_planner_role(self) -> None:
        """Test getting planner role prompt."""
        prompt = get_role_prompt("planner")

        assert "# Role: Planner" in prompt
        assert "Design Document" in prompt
        assert "Multi-Agent Collaboration" in prompt
        assert "@username/agent-name" in prompt
        assert "Planning complete" in prompt

    def test_get_reviewer_role(self) -> None:
        """Test getting reviewer role prompt."""
        prompt = get_role_prompt("reviewer")

        assert "# Role: Code Reviewer" in prompt
        assert "Review Checklist" in prompt
        assert "[Critical]" in prompt
        assert "[Suggestion]" in prompt
        assert "Approved. Ready to merge." in prompt

    def test_get_implementer_role(self) -> None:
        """Test getting implementer role prompt."""
        prompt = get_role_prompt("implementer")

        assert "# Role: Implementer" in prompt
        assert "Progress Updates" in prompt
        assert "Implementation complete" in prompt

    def test_custom_agent_name(self) -> None:
        """Test that custom agent name is used in prompt."""
        prompt = get_role_prompt("planner", agent_name="Design Bot")

        assert "**Design Bot**" in prompt
        assert "Planner" not in prompt.split("# Role: Planner")[1].split("\n")[0]

    def test_default_agent_name(self) -> None:
        """Test that default agent name is used when not specified."""
        prompt = get_role_prompt("planner")

        assert "**Planner**" in prompt

    def test_unknown_role_raises_error(self) -> None:
        """Test that unknown role raises ValueError with helpful message."""
        with pytest.raises(ValueError) as exc_info:
            get_role_prompt("unknown_role")

        assert "Unknown role 'unknown_role'" in str(exc_info.value)
        assert "Available roles:" in str(exc_info.value)
        assert "planner" in str(exc_info.value)

    def test_all_available_roles_work(self) -> None:
        """Test that all roles in AVAILABLE_ROLES can be loaded."""
        for role in AVAILABLE_ROLES:
            prompt = get_role_prompt(role)
            assert isinstance(prompt, str)
            assert len(prompt) > 100  # Should have substantial content


class TestAvailableRoles:
    """Tests for AVAILABLE_ROLES list."""

    def test_contains_expected_roles(self) -> None:
        """Test that expected roles are available."""
        assert "planner" in AVAILABLE_ROLES
        assert "reviewer" in AVAILABLE_ROLES
        assert "implementer" in AVAILABLE_ROLES

    def test_is_list(self) -> None:
        """Test that AVAILABLE_ROLES is a list."""
        assert isinstance(AVAILABLE_ROLES, list)

    def test_all_strings(self) -> None:
        """Test that all roles are strings."""
        for role in AVAILABLE_ROLES:
            assert isinstance(role, str)


class TestRolePromptContent:
    """Tests for specific content in role prompts."""

    def test_planner_has_design_doc_format(self) -> None:
        """Test that planner includes design document template."""
        prompt = get_role_prompt("planner")

        assert "## Summary" in prompt
        assert "## Problem Statement" in prompt
        assert "## Technical Design" in prompt
        assert "## Testing Strategy" in prompt

    def test_planner_has_human_escalation(self) -> None:
        """Test that planner includes when to involve humans."""
        prompt = get_role_prompt("planner")

        assert "When to Involve Humans" in prompt
        assert "Architecture decisions" in prompt
        assert "Security-sensitive" in prompt

    def test_planner_has_termination_signals(self) -> None:
        """Test that planner includes conversation termination signals."""
        prompt = get_role_prompt("planner")

        assert "Conversation Termination Signals" in prompt
        assert "Planning complete" in prompt
        assert "Handing off to" in prompt

    def test_reviewer_has_feedback_format(self) -> None:
        """Test that reviewer includes feedback format guidance."""
        prompt = get_role_prompt("reviewer")

        assert "[Critical]" in prompt
        assert "[Suggestion]" in prompt
        assert "[Nit]" in prompt

    def test_all_roles_have_mention_format(self) -> None:
        """Test that all roles explain how to mention other agents."""
        for role in AVAILABLE_ROLES:
            prompt = get_role_prompt(role)
            assert "@username/agent-name" in prompt or "@agent" in prompt
