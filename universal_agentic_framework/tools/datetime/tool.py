"""Date and time utilities tool."""

from datetime import datetime, timezone
from typing import Literal, Optional
from zoneinfo import ZoneInfo
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger()


class DateTimeInput(BaseModel):
    """Input for datetime operations."""

    operation: Literal["current_time", "convert_timezone"] = Field(
        default="current_time", description="Operation to perform"
    )
    timezone: Optional[str] = Field(
        default=None, description="Timezone (e.g., 'Europe/Berlin', 'America/New_York')"
    )


class DateTimeTool(BaseTool):
    """Get current date/time and perform datetime operations."""

    name: str = "datetime_tool"
    description: str = (
        "Get current date and time, convert timezones, or perform date calculations. "
        "Use this when the user asks 'what time is it', 'current date', or timezone questions."
    )
    args_schema: type[BaseModel] = DateTimeInput

    # Injected by registry (from fork config)
    default_timezone: str = "UTC"

    def _run(
        self, operation: str = "current_time", timezone: Optional[str] = None
    ) -> str:
        """Execute datetime operation."""

        tz = timezone or self.default_timezone

        try:
            if operation == "current_time":
                return self._get_current_time(tz)
            elif operation == "convert_timezone":
                return self._convert_timezone(tz)
            else:
                return f"Operation '{operation}' not yet implemented"

        except Exception as e:
            logger.error("DateTime operation failed", error=str(e), operation=operation)
            return f"Error: {str(e)}"

    async def _arun(
        self, operation: str = "current_time", timezone: Optional[str] = None
    ) -> str:
        """Async execution (uses sync implementation)."""
        return self._run(operation, timezone)

    def _get_current_time(self, tz: str) -> str:
        """Get current time in specified timezone."""
        try:
            zone = ZoneInfo(tz)
            now = datetime.now(zone)

            logger.info("Current time retrieved", timezone=tz)

            return (
                f"Current date and time:\n"
                f"• Date: {now.strftime('%A, %B %d, %Y')}\n"
                f"• Time: {now.strftime('%H:%M:%S')}\n"
                f"• Timezone: {tz}\n"
                f"• UTC Offset: {now.strftime('%z')}\n"
                f"• ISO format: {now.isoformat()}"
            )
        except Exception as e:
            logger.warning("Invalid timezone requested", timezone=tz, error=str(e))
            return f"Invalid timezone '{tz}': {str(e)}"

    def _convert_timezone(self, target_tz: str) -> str:
        """Convert current UTC time to target timezone."""
        try:
            # Get current UTC time
            utc_now = datetime.now(timezone.utc)

            # Convert to target timezone
            target_zone = ZoneInfo(target_tz)
            converted = utc_now.astimezone(target_zone)

            logger.info("Timezone conversion performed", target_timezone=target_tz)

            return (
                f"Current time in {target_tz}:\n"
                f"• Date: {converted.strftime('%A, %B %d, %Y')}\n"
                f"• Time: {converted.strftime('%H:%M:%S')}\n"
                f"• Timezone: {target_tz}\n"
                f"• UTC Offset: {converted.strftime('%z')}"
            )
        except Exception as e:
            logger.warning("Timezone conversion failed", timezone=target_tz, error=str(e))
            return f"Invalid timezone '{target_tz}': {str(e)}"
