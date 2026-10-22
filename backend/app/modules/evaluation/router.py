from uuid import UUID

from fastapi import APIRouter, File, Response, UploadFile, status
from fastapi.responses import FileResponse

from app.modules.evaluation.schemas import (
    ClassifierCreate,
    ClassifierResponse,
    EvaluationRunCreate,
    EvaluationRunResponse,
)
from app.modules.evaluation.service import (
    get_evaluation_dataset,
    upload_evaluation_images,
    list_evaluation_images,
    get_evaluation_image_path,
    delete_evaluation_image,
    start_classifier_training,
    list_classifiers,
    get_classifier,
    start_evaluation,
    list_evaluation_runs,
    get_evaluation_run,
    get_evaluation_results,
    get_chart_path,
)


router = APIRouter(
    prefix="/workspaces/{workspace_id}/evaluation",
    tags=["Evaluation"],
)


# Evaluation dataset: real images per workspace class, used only to train the classifier

@router.get("/dataset")
def get_dataset_endpoint(workspace_id: UUID):
    return get_evaluation_dataset(workspace_id)


@router.post(
    "/dataset/classes/{class_id}/images",
    status_code=status.HTTP_201_CREATED,
)
def upload_images_endpoint(
    workspace_id: UUID,
    class_id: UUID,
    files: list[UploadFile] = File(...),
):
    return upload_evaluation_images(workspace_id, class_id, files)


@router.get("/dataset/classes/{class_id}/images")
def list_images_endpoint(workspace_id: UUID, class_id: UUID):
    return list_evaluation_images(workspace_id, class_id)


@router.get("/dataset/classes/{class_id}/images/{image_id}")
def get_image_endpoint(workspace_id: UUID, class_id: UUID, image_id: UUID):
    path = get_evaluation_image_path(workspace_id, class_id, image_id)
    return FileResponse(
        path=path,
        media_type={
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".webp": "image/webp",
        }[path.suffix.lower()],
    )


@router.delete("/dataset/classes/{class_id}/images/{image_id}", status_code=204)
def delete_image_endpoint(workspace_id: UUID, class_id: UUID, image_id: UUID):
    delete_evaluation_image(workspace_id, class_id, image_id)
    return Response(status_code=204)


# Classifier

@router.post(
    "/classifiers",
    response_model=ClassifierResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def train_classifier_endpoint(workspace_id: UUID, data: ClassifierCreate):
    return start_classifier_training(workspace_id, data)


@router.get("/classifiers")
def list_classifiers_endpoint(workspace_id: UUID):
    return list_classifiers(workspace_id)


@router.get("/classifiers/{classifier_id}")
def get_classifier_endpoint(workspace_id: UUID, classifier_id: UUID):
    return get_classifier(workspace_id, classifier_id)


# Evaluation runs: score one generation run

@router.post(
    "/runs",
    response_model=EvaluationRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_evaluation_endpoint(workspace_id: UUID, data: EvaluationRunCreate):
    return start_evaluation(workspace_id, data)


@router.get("/runs")
def list_runs_endpoint(workspace_id: UUID):
    return list_evaluation_runs(workspace_id)


@router.get("/runs/{run_id}")
def get_run_endpoint(workspace_id: UUID, run_id: UUID):
    return get_evaluation_run(workspace_id, run_id)


@router.get("/runs/{run_id}/results")
def get_results_endpoint(workspace_id: UUID, run_id: UUID):
    return get_evaluation_results(workspace_id, run_id)


@router.get("/runs/{run_id}/charts/{chart}")
def get_chart_endpoint(workspace_id: UUID, run_id: UUID, chart: str):
    return FileResponse(get_chart_path(workspace_id, run_id, chart), media_type="image/png")
