"""Context Provider.

Provides the agent with contextual information like:
- Current time/timezone
- User's location (if available)
- Browser/device information
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AgentContext:
    """Contextual information about the user's environment."""
    current_time: datetime | None = None
    timezone: str | None = None
    user_location: str | None = None  # City, Country
    user_coordinates: dict[str, float] | None = None  # lat, lon
    device_type: str | None = None  # mobile, desktop, tablet
    browser: str | None = None


def get_current_context(
    timezone_str: str | None = None,
    user_location: str | None = None,
    device_info: dict[str, Any] | None = None,
) -> AgentContext:
    """Get current contextual information.

    Args:
        timezone_str: User's timezone (e.g., "Asia/Kolkata")
        user_location: User's location (e.g., "Mumbai, India")
        device_info: Device information from browser

    Returns:
        AgentContext with current context
    """
    # Get current time
    current_time = datetime.now(timezone.utc)

    # Parse device info
    device_type = None
    browser = None
    if device_info:
        device_type = device_info.get("device_type")
        browser = device_info.get("browser")

    return AgentContext(
        current_time=current_time,
        timezone=timezone_str,
        user_location=user_location,
        device_type=device_type,
        browser=browser,
    )


def format_context_for_llm(context: AgentContext) -> str:
    """Format context as a string for LLM prompts.

    Returns a string that can be included in system prompts.
    """
    parts = []

    if context.current_time:
        parts.append(f"Current time: {context.current_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    if context.timezone:
        parts.append(f"User timezone: {context.timezone}")

    if context.user_location:
        parts.append(f"User location: {context.user_location}")

    if context.device_type:
        parts.append(f"Device: {context.device_type}")

    return "\n".join(parts) if parts else "No context available"


# Singleton instance
_agent_context: AgentContext | None = None


def get_agent_context() -> AgentContext:
    """Get or create the agent context singleton."""
    global _agent_context
    if _agent_context is None:
        _agent_context = get_current_context()
    return _agent_context


def update_agent_context(**kwargs) -> AgentContext:
    """Update the agent context with new information."""
    global _agent_context
    if _agent_context is None:
        _agent_context = get_current_context()

    # Update fields
    for key, value in kwargs.items():
        if hasattr(_agent_context, key):
            setattr(_agent_context, key, value)

    return _agent_context