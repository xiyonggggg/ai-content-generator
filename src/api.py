from concurrent.futures import ThreadPoolExecutor
from uuid import UUID

from fastapi import FastAPI, HTTPException

from .config import settings
from .models import ContentRequest, Job
from .pipeline import ContentPipeline
from .store import JobStore


app = FastAPI(title="AI Automatic Content Creator", version="0.1.0")
store = JobStore(settings.jobs_dir)
pipeline = ContentPipeline(settings, store)
executor = ThreadPoolExecutor(max_workers=settings.max_workers)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mode": "demo" if settings.demo_mode else "production"}


@app.post("/jobs", response_model=Job, status_code=202)
def create_job(request: ContentRequest) -> Job:
    job = store.create(Job(request=request))
    executor.submit(pipeline.run, job)
    return job


@app.get("/jobs/{job_id}", response_model=Job)
def get_job(job_id: UUID) -> Job:
    job = store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
