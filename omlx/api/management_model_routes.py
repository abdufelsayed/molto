"""Model metadata, templates and explicit settings helpers."""

from fastapi import APIRouter, Depends

from ..services.management import ManagementService
from ..services.management_model_options import ModelHelpers, options
from ..services.management_models import (
    OptimalApply,
    RecipeApply,
    TemplateUpdate,
    TemplateWrite,
)
from .management_dependencies import get_runtime, get_service

router = APIRouter(tags=["management-models"])


@router.get("/model-options")
def model_options(service: ManagementService = Depends(get_service)):
    return options(service)


@router.get("/models/{model_id:path}/options")
def model_options_for_id(
    model_id: str, service: ManagementService = Depends(get_service)
):
    return options(service, model_id)


@router.get("/templates")
def templates(service: ManagementService = Depends(get_service)):
    return ModelHelpers(service).templates()


@router.post("/templates")
def create_template(
    body: TemplateWrite, service: ManagementService = Depends(get_service)
):
    return ModelHelpers(service).write_template(body)


@router.put("/templates/{name}")
def update_template(
    name: str, body: TemplateUpdate, service: ManagementService = Depends(get_service)
):
    return ModelHelpers(service).write_template(body, name)


@router.delete("/templates/{name}")
def delete_template(name: str, service: ManagementService = Depends(get_service)):
    return ModelHelpers(service).delete_template(name)


@router.post("/models/{model_id:path}/templates/{name}/apply")
async def apply_template(
    model_id: str, name: str, service: ManagementService = Depends(get_service)
):
    return await ModelHelpers(service).apply_template(model_id, name)


@router.get("/presets")
def presets(service: ManagementService = Depends(get_service)):
    import json
    from pathlib import Path

    return json.loads(
        (Path(__file__).parents[1] / "services" / "management_presets.json").read_text()
    )


@router.post("/presets/refresh")
async def refresh_presets(service: ManagementService = Depends(get_service)):
    return await ModelHelpers(service).refresh_presets()


@router.get("/models/{model_id:path}/generation-config")
def generation_config(model_id: str, service: ManagementService = Depends(get_service)):
    return ModelHelpers(service).generation_config(model_id)


@router.post("/models/{model_id:path}/generation-config")
async def import_generation_config(
    model_id: str, service: ManagementService = Depends(get_service)
):
    from ..services.management_models import ModelSettingsPatch

    helper = ModelHelpers(service)
    return await service.update_model_settings(
        model_id, ModelSettingsPatch(**helper.generation_config(model_id)["settings"])
    )


@router.post("/models/{model_id:path}/settings/reset")
async def reset(model_id: str, service: ManagementService = Depends(get_service)):
    return await ModelHelpers(service).reset(model_id)


@router.post("/models/{model_id:path}/settings/recipe")
async def apply_recipe(
    model_id: str, body: RecipeApply, service: ManagementService = Depends(get_service)
):
    return await ModelHelpers(service).recipe(model_id, body.recipe)


@router.post("/models/{model_id:path}/settings/optimal")
async def apply_optimal(
    model_id: str, body: OptimalApply, service: ManagementService = Depends(get_service)
):
    return await ModelHelpers(service).optimal(model_id, body.benchmark_id)


@router.get("/models/{model_id:path}/settings/optimal")
async def optimal_candidates(
    model_id: str, service: ManagementService = Depends(get_service)
):
    return await ModelHelpers(service).optimal_candidates(model_id)


@router.post("/models/{model_id:path}/presets/{name}/apply")
async def apply_preset(
    model_id: str, name: str, service: ManagementService = Depends(get_service)
):
    return await ModelHelpers(service).apply_preset(model_id, name)


@router.post("/models/{model_id:path}/import-mtplx")
async def import_mtplx(
    model_id: str,
    service: ManagementService = Depends(get_service),
    runtime=Depends(get_runtime),
):
    return await ModelHelpers(service).import_mtplx(model_id, runtime)
