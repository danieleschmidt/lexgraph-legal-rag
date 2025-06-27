from __future__ import annotations

from fastapi import APIRouter, FastAPI
from pydantic import BaseModel

import logging

from .sample import add as add_numbers

"""FastAPI application with basic versioned routing."""

SUPPORTED_VERSIONS = ("v1",)


logger = logging.getLogger(__name__)


class PingResponse(BaseModel):
    version: str
    ping: str


class AddResponse(BaseModel):
    result: int


def create_api(version: str = SUPPORTED_VERSIONS[0]) -> FastAPI:
    """Return a FastAPI app configured for the given API ``version``."""
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(f"Unsupported API version: {version}")

    app = FastAPI(title="LexGraph Legal RAG", version=version)

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

    versioned_router = APIRouter(prefix=f"/{version}")
    register_routes(versioned_router)
    app.include_router(versioned_router)

    if version == SUPPORTED_VERSIONS[0]:
        default_router = APIRouter()
        register_routes(default_router)
        app.include_router(default_router)

    return app
