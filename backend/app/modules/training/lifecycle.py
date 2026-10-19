import json
from pathlib import Path

from filelock import FileLock


TERMINAL = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "SUBMISSION_FAILED",
}


class TrainingCancelled(Exception):
    pass


def get_lock(run_dir):
    run_dir = Path(run_dir)
    return FileLock(str(run_dir / "run.lock"), timeout=30)


def read_run(run_dir):
    run_dir = Path(run_dir)

    with get_lock(run_dir):
        path = run_dir / "run.json"
        return json.loads(path.read_text(encoding="utf-8"))


def update_run(run_dir, status=None, **extra):
    run_dir = Path(run_dir)
    path = run_dir / "run.json"

    with get_lock(run_dir):
        data = json.loads(path.read_text(encoding="utf-8"))

        current = data.get("status")
        cancelling = (
            current == "CANCELLING"
            or (run_dir / "cancel.requested").exists()
        )

        if status is not None:
            if current in TERMINAL and status != current:
                raise RuntimeError(
                    f"Cannot transition terminal run "
                    f"{current} to {status}"
                )

            if cancelling and status in {
                "PREPARING",
                "TRAINING",
                "COMPLETED",
            }:
                raise TrainingCancelled(
                    "Cancellation requested"
                )

            data["status"] = status

        data.update(extra)

        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(data, indent=4),
            encoding="utf-8",
        )
        temporary.replace(path)

        return data


def request_cancellation(run_dir):
    run_dir = Path(run_dir)

    with get_lock(run_dir):
        path = run_dir / "run.json"
        data = json.loads(path.read_text(encoding="utf-8"))

        if data["status"] in TERMINAL:
            return data

        (run_dir / "cancel.requested").touch(exist_ok=True)

        data["status"] = "CANCELLING"

        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(data, indent=4),
            encoding="utf-8",
        )
        temporary.replace(path)

        return data


def cancellation_requested(run_dir):
    return (Path(run_dir) / "cancel.requested").exists()


def check_cancellation(run_dir):
    if cancellation_requested(run_dir):
        raise TrainingCancelled(
            "Training cancelled by user"
        )
