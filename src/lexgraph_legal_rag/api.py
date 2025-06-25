from __future__ import annotations

from fastapi import APIRouter, FastAPI

from .sample import add as add_numbers

"""FastAPI application with basic versioned routing."""

SUPPORTED_VERSIONS = ("v1",)


def create_api(version: str = SUPPORTED_VERSIONS[0]) -> FastAPI:
    """Return a FastAPI app configured for the given API ``version``."""
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(f"Unsupported API version: {version}")

    app = FastAPI(title="LexGraph Legal RAG", version=version)

    def register_routes(router: APIRouter) -> None:
        @router.get("/ping")
        async def ping() -> dict[str, str]:
            return {"version": version, "ping": "pong"}

        @router.get("/add")
        async def add(a: int, b: int) -> dict[str, int]:
            """Return the sum of two integers."""
            return {"result": add_numbers(a, b)}

    versioned_router = APIRouter(prefix=f"/{version}")
    register_routes(versioned_router)
    app.include_router(versioned_router)

    if version == SUPPORTED_VERSIONS[0]:
        default_router = APIRouter()
        register_routes(default_router)
        app.include_router(default_router)

    return app
