from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class PipelineStage(str, Enum):
    queued = "queued"
    script_generation = "script_generation"
    audio_generation = "audio_generation"
    video_generation = "video_generation"
    media_compilation = "media_compilation"
    publishing = "publishing"
    completed = "completed"
    failed = "failed"


class ContentRequest(BaseModel):
    prompt: str = Field(min_length=8, description="Natural-language description of the content to create")
    platforms: list[str] = Field(default_factory=lambda: ["youtube", "tiktok"])
    duration_seconds: int = Field(default=45, ge=15, le=600)
    language: str = Field(default="en", min_length=2, max_length=12)
    tone: str = Field(default="engaging", min_length=2, max_length=40)
    publish: bool = Field(default=False, description="Publish after rendering when platform credentials are configured")

    @field_validator("platforms")
    @classmethod
    def validate_platforms(cls, value: list[str]) -> list[str]:
        allowed = {"youtube", "tiktok"}
        normalized = [item.strip().lower() for item in value]
        if not normalized or any(item not in allowed for item in normalized):
            raise ValueError("platforms must contain only youtube and/or tiktok")
        return list(dict.fromkeys(normalized))


class Job(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    request: ContentRequest
    status: JobStatus = JobStatus.queued
    stage: PipelineStage = PipelineStage.queued
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    result: dict[str, Any] | None = None
    error: str | None = None
