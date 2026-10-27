from celery import Celery

from app.core.settings import settings

celery_app = Celery(
    "synthetic_image_platform",
    broker=f"{settings.REDIS_URL}/0",
    backend=f"{settings.REDIS_URL}/1",
    include=[
        "app.workers.training_tasks",
        "app.workers.generation_tasks",
    ],
)

celery_app.conf.update(
    task_routes={
        "training.train_lora": {"queue": "gpu"},
        "generation.generate_images": {"queue": "gpu"},
    },
    task_default_queue="default",
    worker_prefetch_multiplier=1,
    task_acks_late=True,

    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=3600,
)