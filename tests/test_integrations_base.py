"""
Tests for base integration utilities.

Tests cover:
- check_and_format_participants() helper function
"""

from unittest.mock import MagicMock


class TestCheckAndFormatParticipants:
    """Test check_and_format_participants() helper."""

    def test_returns_none_when_no_change(self):
        """Should return None when participants haven't changed."""
        from thenvoi.integrations.base import check_and_format_participants

        session = MagicMock()
        session.participants_changed.return_value = False

        result = check_and_format_participants(session)

        assert result is None
        session.build_participants_message.assert_not_called()
        session.mark_participants_sent.assert_not_called()

    def test_returns_message_when_changed(self):
        """Should return formatted message when participants changed."""
        from thenvoi.integrations.base import check_and_format_participants

        session = MagicMock()
        session.participants_changed.return_value = True
        session.build_participants_message.return_value = (
            "## Current Participants\n- Alice"
        )

        result = check_and_format_participants(session)

        assert result == "## Current Participants\n- Alice"
        session.build_participants_message.assert_called_once()
        session.mark_participants_sent.assert_called_once()

    def test_marks_participants_sent_automatically(self):
        """Should automatically call mark_participants_sent() when returning message."""
        from thenvoi.integrations.base import check_and_format_participants

        session = MagicMock()
        session.participants_changed.return_value = True
        session.build_participants_message.return_value = "Participants list"

        check_and_format_participants(session)

        session.mark_participants_sent.assert_called_once()


class TestIntegrationsImport:
    """Test that integrations module exports base utilities."""

    def test_can_import_from_integrations(self):
        """Should be able to import check_and_format_participants from integrations."""
        from thenvoi.integrations import check_and_format_participants

        assert check_and_format_participants is not None

    def test_check_and_format_participants_in_all(self):
        """Should be listed in __all__."""
        from thenvoi import integrations

        assert "check_and_format_participants" in integrations.__all__
