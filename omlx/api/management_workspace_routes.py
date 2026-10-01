"""Authenticated mounting is owned by the parent management router."""

from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from ..services.management_workspace import WorkspaceService
from .management_dependencies import get_runtime

router = APIRouter(tags=["management-workspace"])


def workspace(runtime=Depends(get_runtime)):
    return WorkspaceService(runtime)


class StrictBody(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Selection(StrictBody):
    model_ids: list[str] = Field(max_length=100)


class Collection(Selection):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)
    preload: bool = False


class ImportBundle(StrictBody):
    bundle: dict
    dry_run: bool = True


class Verify(StrictBody):
    mode: Literal["structural", "smoke"] = "structural"


class Drain(StrictBody):
    drain: bool = False


class Revision(Drain):
    revision: str


class Move(Drain):
    root_id: str


class Delete(Drain):
    plan_token: str


@router.get("/workspace/registry")
async def registry(service=Depends(workspace)):
    return await service.registry()


@router.get("/workspace/storage")
async def storage(service=Depends(workspace)):
    return await service.storage()


@router.post("/workspace/plan")
async def plan(body: Selection, service=Depends(workspace)):
    return service.plan(body.model_ids)


@router.get("/workspace/collections")
async def collections(service=Depends(workspace)):
    return service.collections()


@router.put("/workspace/collections/{collection_id}")
async def save_collection(
    collection_id: str, body: Collection, service=Depends(workspace)
):
    return await service.save_collection(collection_id, body.model_dump())


@router.delete("/workspace/collections/{collection_id}")
async def delete_collection(collection_id: str, service=Depends(workspace)):
    return await service.delete_collection(collection_id)


@router.post("/workspace/collections/{collection_id}/load")
async def load_collection(collection_id: str, service=Depends(workspace)):
    return await service.load_collection(collection_id)


@router.get("/workspace/export")
async def export(service=Depends(workspace)):
    return await service.export()


@router.post("/workspace/import")
async def import_bundle(body: ImportBundle, service=Depends(workspace)):
    return await service.import_bundle(body.bundle, dry_run=body.dry_run)


@router.post("/workspace/models/{model_id:path}/verify")
async def verify(model_id: str, body: Verify, service=Depends(workspace)):
    return await service.verify(model_id, mode=body.mode)


@router.post("/workspace/models/{model_id:path}/check-update")
async def check_update(model_id: str, service=Depends(workspace)):
    return await service.check_update(model_id)


@router.post("/workspace/models/{model_id:path}/stage-update")
async def stage_update(model_id: str, service=Depends(workspace)):
    return await service.stage_update(model_id)


@router.post("/workspace/models/{model_id:path}/revision")
async def revision(model_id: str, body: Revision, service=Depends(workspace)):
    return await service.activate_revision(model_id, body.revision, drain=body.drain)


@router.post("/workspace/models/{model_id:path}/move")
async def move(model_id: str, body: Move, service=Depends(workspace)):
    return await service.move(
        model_id, destination_root_id=body.root_id, drain=body.drain
    )


@router.get("/workspace/models/{model_id:path}/delete-plan")
async def delete_plan(model_id: str, service=Depends(workspace)):
    return await service.delete_plan(model_id)


@router.delete("/workspace/models/{model_id:path}/delete")
async def delete_model(model_id: str, body: Delete, service=Depends(workspace)):
    return await service.delete(model_id, plan_token=body.plan_token, drain=body.drain)
