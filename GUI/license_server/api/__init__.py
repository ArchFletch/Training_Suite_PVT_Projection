"""FastAPI app wiring and route modules for the on-prem license server."""

from license_server.api.app import app, create_app

__all__ = ["app", "create_app"]
