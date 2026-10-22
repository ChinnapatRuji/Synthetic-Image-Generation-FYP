from celery import Celery

celery_app = Celery(
    "synthetic_image_platform",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1",
    include=[
        "app.workers.training_tasks",
        "app.workers.generation_tasks",
        "app.workers.evaluation_tasks",
    ],
)

celery_app.conf.update(
    task_routes={
        "training.train_lora": {"queue": "gpu"},
        "generation.generate_images": {"queue": "gpu"},
        "evaluation.train_classifier": {"queue": "gpu"},
        "evaluation.evaluate_generation": {"queue": "gpu"},
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