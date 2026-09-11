"""Creation studio capability routes."""

from __future__ import annotations

from fastapi import APIRouter

from dreamverse.creation_capabilities import LOBBY_CREATION_CAPABILITIES

creation_router = APIRouter(tags=["creation"])


@creation_router.get("/creation-capabilities")
async def creation_capabilities() -> dict[str, object]:
    return LOBBY_CREATION_CAPABILITIES.as_dict()
