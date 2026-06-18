"""FastAPI application factory for the customer-facing license server."""

from __future__ import annotations

from fastapi import FastAPI

from license_server.api.routes_client import router as client_router
from license_server.service import LicenseServerRuntime, ServerConfig, create_runtime


def create_app(
    *,
    config: ServerConfig | None = None,
    runtime: LicenseServerRuntime | None = None,
    runtime_root: str | None = None,
    vendor_public_key_pem: str = "",
) -> FastAPI:
    service_runtime = runtime or create_runtime(
        config=config,
        runtime_root=runtime_root,
        vendor_public_key_pem=vendor_public_key_pem,
    )
    app = FastAPI(
        title="MLP License Server",
        version="0.1.0",
        description="On-prem floating license server for Surrogate Model Training Suite.",
    )
    app.state.runtime = service_runtime
    app.include_router(client_router)
    return app


app = create_app()
