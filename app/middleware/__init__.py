"""Middleware package."""

from app.middleware.guardrails import guardrails_middleware

__all__ = ["guardrails_middleware"]
