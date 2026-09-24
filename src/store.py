import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import UUID

from .models import Job


class JobStore:
    def __init__(self, jobs_dir: Path):
        self.jobs_dir = jobs_dir
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def create(self, job: Job) -> Job:
        return self.save(job)

    def save(self, job: Job) -> Job:
        job.updated_at = datetime.now(timezone.utc).isoformat()
        path = self.jobs_dir / f"{job.id}.json"
        with self._lock:
            path.write_text(job.model_dump_json(indent=2), encoding="utf-8")
        return job

    def get(self, job_id: UUID) -> Job | None:
        path = self.jobs_dir / f"{job_id}.json"
        if not path.exists():
            return None
        return Job.model_validate_json(path.read_text(encoding="utf-8"))
