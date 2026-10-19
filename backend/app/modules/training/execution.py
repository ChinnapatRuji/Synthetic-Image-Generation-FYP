from pathlib import Path

from filelock import FileLock, Timeout


def execution_lock(run_dir):
    run_dir = Path(run_dir).resolve()

    lock_dir = run_dir.parent.parent / ".training_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)

    lock_path = lock_dir / f"{run_dir.name}.lock"

    return FileLock(str(lock_path), timeout=0)


def is_execution_active(run_dir):
    lock = execution_lock(run_dir)

    try:
        with lock.acquire(timeout=0):
            return False
    except Timeout:
        return True
