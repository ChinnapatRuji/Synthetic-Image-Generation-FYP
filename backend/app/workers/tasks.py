import time

from app.workers.celery_app import celery_app


@celery_app.task(name="test_task")
def test_task():
    time.sleep(5)

    return {
        "status": "completed",
        "message": "Celery is working!",
    }