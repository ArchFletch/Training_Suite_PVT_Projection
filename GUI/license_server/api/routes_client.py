"""Customer-facing client routes for seat checkout and status inspection."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from license_server.schemas import (
    CheckoutRequest,
    CheckoutResponse,
    HeartbeatRequest,
    HeartbeatResponse,
    ReleaseRequest,
    ReleaseResponse,
    StatusResponse,
)
from license_server.service import LicenseServerRuntime


router = APIRouter(prefix="/api/v1", tags=["client"])


def get_runtime(request: Request) -> LicenseServerRuntime:
    return request.app.state.runtime


@router.post("/checkout", response_model=CheckoutResponse, response_model_exclude_none=True)
def checkout(
    payload: CheckoutRequest,
    runtime: LicenseServerRuntime = Depends(get_runtime),
) -> CheckoutResponse:
    return runtime.lease_service.checkout(payload).to_dict()


@router.post("/heartbeat", response_model=HeartbeatResponse, response_model_exclude_none=True)
def heartbeat(
    payload: HeartbeatRequest,
    runtime: LicenseServerRuntime = Depends(get_runtime),
) -> HeartbeatResponse:
    return runtime.lease_service.heartbeat(payload).to_dict()


@router.post("/release", response_model=ReleaseResponse, response_model_exclude_none=True)
def release(
    payload: ReleaseRequest,
    runtime: LicenseServerRuntime = Depends(get_runtime),
) -> ReleaseResponse:
    return runtime.lease_service.release(payload).to_dict()


@router.get("/status", response_model=StatusResponse, response_model_exclude_none=True)
def status(runtime: LicenseServerRuntime = Depends(get_runtime)) -> StatusResponse:
    return runtime.license_service.get_status().to_dict()
