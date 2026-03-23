"""Convenience module exposing the default FastAPI application."""

from license_server.api.app import app, create_app

__all__ = ["app", "create_app"]
