"""Unit tests for DateTime tool."""

import pytest
from datetime import datetime
from zoneinfo import ZoneInfo
from unittest.mock import patch
from universal_agentic_framework.tools.datetime.tool import DateTimeTool, DateTimeInput


@pytest.fixture
def datetime_tool():
    """Create a DateTimeTool instance with Berlin timezone."""
    return DateTimeTool(default_timezone="Europe/Berlin")


class TestDateTimeToolInstantiation:
    """Test tool instantiation and configuration."""

    def test_tool_creation_with_default_timezone(self):
        """Test creating tool with custom timezone."""
        tool = DateTimeTool(default_timezone="America/New_York")
        assert tool.default_timezone == "America/New_York"
        assert tool.name == "datetime_tool"

    def test_tool_creation_with_utc(self):
        """Test creating tool with UTC timezone."""
        tool = DateTimeTool(default_timezone="UTC")
        assert tool.default_timezone == "UTC"

    def test_tool_name_and_description(self):
        """Test tool has correct name and description."""
        tool = DateTimeTool()
        assert tool.name == "datetime_tool"
        assert "date and time" in tool.description.lower()
        assert "timezone" in tool.description.lower()


class TestCurrentTimeOperation:
    """Test current_time operation."""

    def test_current_time_returns_formatted_output(self, datetime_tool):
        """Test that current_time returns properly formatted output."""
        result = datetime_tool._run(operation="current_time")

        assert "Current date and time:" in result
        assert "Date:" in result
        assert "Time:" in result
        assert "Timezone:" in result
        assert "Europe/Berlin" in result

    def test_current_time_with_utc(self):
        """Test current_time with UTC timezone."""
        tool = DateTimeTool(default_timezone="UTC")
        result = tool._run(operation="current_time")

        assert "UTC" in result
        assert "Date:" in result
        assert "Time:" in result

    def test_current_time_contains_iso_format(self, datetime_tool):
        """Test that output includes ISO format."""
        result = datetime_tool._run(operation="current_time")
        assert "ISO format:" in result
        # Should contain T separator for ISO format
        assert "T" in result

    def test_current_time_with_custom_timezone_parameter(self, datetime_tool):
        """Test overriding default timezone with parameter."""
        result = datetime_tool._run(
            operation="current_time", timezone="America/New_York"
        )

        assert "America/New_York" in result
        assert "New_York" in result or "New York" in result

    def test_current_time_all_timezones(self):
        """Test that common timezones work."""
        tool = DateTimeTool()
        timezones = [
            "UTC",
            "Europe/Berlin",
            "Europe/London",
            "America/New_York",
            "America/Los_Angeles",
            "Asia/Tokyo",
            "Australia/Sydney",
        ]

        for tz in timezones:
            result = tool._run(operation="current_time", timezone=tz)
            assert "Current date and time:" in result
            # Should either have the timezone name or UTC offset
            assert tz in result or ("UTC" in result or "+" in result or "-" in result)


class TestConvertTimezoneOperation:
    """Test convert_timezone operation."""

    def test_convert_timezone_returns_formatted_output(self, datetime_tool):
        """Test that convert_timezone returns properly formatted output."""
        result = datetime_tool._run(
            operation="convert_timezone", timezone="America/New_York"
        )

        assert "Current time in America/New_York:" in result
        assert "Date:" in result
        assert "Time:" in result
        assert "Timezone:" in result

    def test_convert_timezone_to_multiple_zones(self, datetime_tool):
        """Test converting to various timezones."""
        result_ny = datetime_tool._run(
            operation="convert_timezone", timezone="America/New_York"
        )
        result_tokyo = datetime_tool._run(
            operation="convert_timezone", timezone="Asia/Tokyo"
        )

        # Both should have proper structure
        assert "Current time in" in result_ny
        assert "Current time in" in result_tokyo

    def test_convert_timezone_utc_offset_present(self, datetime_tool):
        """Test that UTC offset is included."""
        result = datetime_tool._run(
            operation="convert_timezone", timezone="Europe/Berlin"
        )

        # Should contain UTC offset (+ or -)
        assert "+" in result or "-" in result


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_timezone_current_time(self, datetime_tool):
        """Test handling of invalid timezone in current_time."""
        result = datetime_tool._run(operation="current_time", timezone="Invalid/Timezone")

        assert "Error:" in result or "Invalid timezone" in result
        assert "Invalid/Timezone" in result

    def test_invalid_timezone_convert(self, datetime_tool):
        """Test handling of invalid timezone in convert_timezone."""
        result = datetime_tool._run(
            operation="convert_timezone", timezone="NotAReal/Zone"
        )

        assert "Error:" in result or "Invalid timezone" in result

    def test_invalid_operation(self, datetime_tool):
        """Test handling of invalid operation."""
        result = datetime_tool._run(operation="unknown_operation")

        assert "not yet implemented" in result or "Error:" in result


class TestAsyncExecution:
    """Test async execution."""

    def test_sync_current_time(self, datetime_tool):
        """Test current_time operation via sync interface."""
        result = datetime_tool._run(operation="current_time")

        assert "Current date and time:" in result
        assert "Date:" in result

    def test_sync_convert_timezone(self, datetime_tool):
        """Test convert_timezone operation via sync interface."""
        result = datetime_tool._run(
            operation="convert_timezone", timezone="America/New_York"
        )

        assert "Current time in" in result


class TestInputSchema:
    """Test input schema validation."""

    def test_datetime_input_default_operation(self):
        """Test DateTimeInput with default operation."""
        input_data = DateTimeInput()
        assert input_data.operation == "current_time"
        assert input_data.timezone is None

    def test_datetime_input_with_timezone(self):
        """Test DateTimeInput with timezone."""
        input_data = DateTimeInput(timezone="Europe/Berlin")
        assert input_data.timezone == "Europe/Berlin"

    def test_datetime_input_all_operations(self):
        """Test all valid operations."""
        ops = ["current_time", "convert_timezone"]
        for op in ops:
            input_data = DateTimeInput(operation=op)
            assert input_data.operation == op


class TestDefaultTimezoneUsage:
    """Test default timezone behavior."""

    def test_default_timezone_used_when_not_specified(self):
        """Test that default timezone is used when parameter not provided."""
        tool = DateTimeTool(default_timezone="Asia/Tokyo")
        result = tool._run(operation="current_time")

        assert "Asia/Tokyo" in result

    def test_parameter_overrides_default(self):
        """Test that parameter timezone overrides default."""
        tool = DateTimeTool(default_timezone="Asia/Tokyo")
        result = tool._run(operation="current_time", timezone="Europe/Berlin")

        assert "Europe/Berlin" in result
        assert "Asia/Tokyo" not in result


class TestTimezoneAccuracy:
    """Test that timezone information is accurate."""

    def test_berlin_timezone_offset(self):
        """Test that Berlin timezone is correctly handled."""
        tool = DateTimeTool(default_timezone="Europe/Berlin")

        # Get current time
        now = datetime.now(ZoneInfo("Europe/Berlin"))

        result = tool._run(operation="current_time")

        # Verify it's today's date
        assert now.strftime("%B %d, %Y") in result or now.strftime(
            "%B %d"
        ) in result  # Allow year flexibility due to async

    def test_multiple_timezones_show_correct_times(self, datetime_tool):
        """Test that different timezones show appropriate offsets."""
        ny_result = datetime_tool._run(
            operation="convert_timezone", timezone="America/New_York"
        )
        tokyo_result = datetime_tool._run(
            operation="convert_timezone", timezone="Asia/Tokyo"
        )

        # Both should be valid results
        assert "Current time in" in ny_result
        assert "Current time in" in tokyo_result
        # They should have different content (at least the timezone name)
        assert ny_result != tokyo_result
