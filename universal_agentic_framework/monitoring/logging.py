"""Structured logging configuration with structlog."""

import logging
import sys
from typing import Any
import structlog


def configure_logging(
    level: str = "INFO",
    json_logs: bool = False,
    context: dict[str, Any] | None = None
) -> None:
    """Configure structured logging for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_logs: Whether to output logs as JSON (for production)
        context: Additional context to add to all log messages
    """
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )
    
    # Build processor chain
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    # Add context processor if provided
    if context:
        processors.insert(0, structlog.processors.merge_contextvars)
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(**context)
    
    # Choose renderer based on environment
    if json_logs:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


def bind_context(**kwargs: Any) -> None:
    """Bind context variables to current thread/task.
    
    Args:
        **kwargs: Context key-value pairs to bind
        
    Example:
        bind_context(user_id="user_123", session_id="sess_abc")
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def unbind_context(*keys: str) -> None:
    """Unbind context variables from current thread/task.
    
    Args:
        *keys: Context keys to unbind
        
    Example:
        unbind_context("user_id", "session_id")
    """
    structlog.contextvars.unbind_contextvars(*keys)


def clear_context() -> None:
    """Clear all context variables for current thread/task."""
    structlog.contextvars.clear_contextvars()
