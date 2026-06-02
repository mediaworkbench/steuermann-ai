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
        default=None, description="Target timezone (e.g., 'Europe/Berlin', 'America/New_York')"
    )
    time: Optional[str] = Field(
        default=None,
        description="Time to convert, e.g. '14:30' or '2:30 PM'. If omitted, uses current time.",
    )
    from_timezone: Optional[str] = Field(
        default=None,
        description="Source timezone for the time value (convert_timezone only), e.g. 'America/New_York'. Defaults to UTC.",
    )


class DateTimeTool(BaseTool):
    """Get current date/time and perform datetime operations."""

    name: str = "datetime_tool"
    description: str = (
        "Get current date and time, convert timezones, or convert a specific time between timezones. "
        "Use this when the user asks 'what time is it', 'current date', timezone questions, "
        "or 'what time is 3pm Berlin in New York?'."
    )
    args_schema: type[BaseModel] = DateTimeInput

    # Injected by registry (from profile config)
    default_timezone: str = "UTC"

    def _run(
        self,
        operation: str = "current_time",
        timezone: Optional[str] = None,
        time: Optional[str] = None,
        from_timezone: Optional[str] = None,
    ) -> str:
        """Execute datetime operation."""

        tz = timezone or self.default_timezone

        try:
            if operation == "current_time":
                return self._get_current_time(tz)
            elif operation == "convert_timezone":
                return self._convert_timezone(tz, time, from_timezone)
            else:
                return f"Operation '{operation}' not yet implemented"

        except Exception as e:
            logger.error("DateTime operation failed", error=str(e), operation=operation)
            return f"Error: {str(e)}"

    async def _arun(
        self,
        operation: str = "current_time",
        timezone: Optional[str] = None,
        time: Optional[str] = None,
        from_timezone: Optional[str] = None,
    ) -> str:
        """Async execution (uses sync implementation)."""
        return self._run(operation, timezone, time, from_timezone)

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

    def _convert_timezone(
        self, target_tz: str, time_str: Optional[str], from_tz: Optional[str]
    ) -> str:
        """Convert a time (or current time) from one timezone to another."""
        try:
            source_zone = ZoneInfo(from_tz) if from_tz else timezone.utc
            target_zone = ZoneInfo(target_tz)

            if time_str:
                # Parse the supplied time string against today's date in the source zone
                today = datetime.now(source_zone).date()
                parsed = None
                for fmt in ("%H:%M", "%I:%M %p", "%I:%M%p"):
                    try:
                        parsed = datetime.strptime(time_str.strip(), fmt)
                        break
                    except ValueError:
                        continue
                if parsed is None:
                    return (
                        f"Could not parse time '{time_str}'. "
                        "Use formats like '14:30', '2:30 PM', or '09:00'."
                    )
                source_dt = datetime(
                    today.year, today.month, today.day,
                    parsed.hour, parsed.minute, parsed.second,
                    tzinfo=source_zone,
                )
            else:
                source_dt = datetime.now(source_zone)

            converted = source_dt.astimezone(target_zone)
            source_label = from_tz or "UTC"

            logger.info(
                "Timezone conversion performed",
                source_timezone=source_label,
                target_timezone=target_tz,
            )

            return (
                f"Time conversion:\n"
                f"• Source: {source_dt.strftime('%H:%M:%S')} {source_label}\n"
                f"• Target: {converted.strftime('%H:%M:%S')} {target_tz}\n"
                f"• Date: {converted.strftime('%A, %B %d, %Y')}\n"
                f"• UTC Offset: {converted.strftime('%z')}"
            )
        except Exception as e:
            logger.warning(
                "Timezone conversion failed",
                target_timezone=target_tz,
                from_timezone=from_tz,
                error=str(e),
            )
            return f"Error during timezone conversion: {str(e)}"
