from app.modules.training.router import router as training_router
from fastapi import APIRouter

from app.core.settings import settings
from app.modules.system.router import router as system_router
from app.modules.workspaces.router import router as workspaces_router
from app.modules.datasets.router import router as datasets_router


api_router = APIRouter(prefix=settings.API_PREFIX)

api_router.include_router(system_router)
api_router.include_router(workspaces_router)
api_router.include_router(datasets_router)

api_router.include_router(training_router)
