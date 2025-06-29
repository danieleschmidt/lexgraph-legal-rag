from __future__ import annotations

from fastapi import APIRouter, FastAPI, Depends, Header, HTTPException, status
import os
import time
from collections import deque
from pydantic import BaseModel

import logging

from .sample import add as add_numbers

"""FastAPI application with API key auth and rate limiting."""

SUPPORTED_VERSIONS = ("v1",)

API_KEY_ENV = "API_KEY"
RATE_LIMIT = 60  # requests per minute


def verify_api_key(x_api_key: str = Header(...), api_key: str | None = None) -> None:
    if api_key is None:
        api_key = os.environ.get(API_KEY_ENV)
    if not api_key or x_api_key != api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
        )


def enforce_rate_limit(request_times: deque[float], limit: int = RATE_LIMIT) -> None:
    now = time.time()
    while request_times and now - request_times[0] > 60:
        request_times.popleft()
    if len(request_times) >= limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
        )
    request_times.append(now)


logger = logging.getLogger(__name__)


class PingResponse(BaseModel):
    version: str
    ping: str


class AddResponse(BaseModel):
    result: int


def create_api(
    version: str = SUPPORTED_VERSIONS[0],
    api_key: str | None = None,
    rate_limit: int = RATE_LIMIT,
) -> FastAPI:
    """Return a FastAPI app configured for the given API ``version``."""
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(f"Unsupported API version: {version}")

    if api_key is None:
        api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        raise ValueError("API key not provided")

    app = FastAPI(title="LexGraph Legal RAG", version=version)

    request_times: deque[float] = deque()

    def register_routes(router: APIRouter) -> None:
        @router.get("/ping", response_model=PingResponse)
        async def ping() -> PingResponse:
            logger.debug("/ping called")
            return PingResponse(version=version, ping="pong")

        @router.get("/add", response_model=AddResponse)
        async def add(a: int, b: int) -> AddResponse:
            """Return the sum of two integers."""
            logger.debug("/add called with a=%s b=%s", a, b)
            return AddResponse(result=add_numbers(a, b))

    def auth_dep(x_api_key: str | None = Header(None)) -> None:
        verify_api_key(x_api_key, api_key)

    def rate_dep() -> None:
        enforce_rate_limit(request_times, rate_limit)

    dependencies = [Depends(auth_dep), Depends(rate_dep)]
    versioned_router = APIRouter(prefix=f"/{version}", dependencies=dependencies)
    register_routes(versioned_router)
    app.include_router(versioned_router)

    if version == SUPPORTED_VERSIONS[0]:
        default_router = APIRouter(dependencies=dependencies)
        register_routes(default_router)
        app.include_router(default_router)

    return app
