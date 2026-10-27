from datetime import datetime, timezone


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def elapsed_seconds(started_at, ended_at):
    if not started_at or not ended_at:
        return None

    start = datetime.fromisoformat(started_at)
    end = datetime.fromisoformat(ended_at)

    return round(max(0, (end - start).total_seconds()), 2)
