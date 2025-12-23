"""
Framework-specific adapters for normalized messages.

Each adapter converts NormalizedMessage → framework-specific format.
"""

# Adapters are imported directly from their modules to avoid
# requiring all framework dependencies when only using one.
#
# Usage:
#   from thenvoi.integrations.history.adapters.anthropic import to_anthropic_messages
#   from thenvoi.integrations.history.adapters.langgraph import to_langgraph_messages
