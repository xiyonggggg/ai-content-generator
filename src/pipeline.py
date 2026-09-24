from pathlib import Path

from .config import Settings
from .models import Job, JobStatus, PipelineStage
from .store import JobStore
from .tasks.audio_generation import AudioGenerator
from .tasks.media_compilation import MediaCompiler
from .tasks.script_generation import ScriptGenerator
from .tasks.video_generation import VideoGenerator


class ContentPipeline:
    def __init__(self, settings: Settings, store: JobStore):
        self.settings = settings
        self.store = store
        self.script_generator = ScriptGenerator(settings)
        self.audio_generator = AudioGenerator(settings)
        self.video_generator = VideoGenerator(settings)
        self.media_compiler = MediaCompiler()

    def run(self, job: Job) -> None:
        output_dir = self.settings.jobs_dir / str(job.id)
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            job.status = JobStatus.running
            job.stage = PipelineStage.script_generation
            self.store.save(job)
            script = self.script_generator.generate(job.request)
            script_path = output_dir / "script.json"
            script_path.write_text(script.model_dump_json(indent=2), encoding="utf-8")

            artifacts = {
                "script": str(script_path.relative_to(self.settings.data_dir)),
            }
            next_tasks = ["audio_generation", "video_generation", "publishing"]
            audio_path = None
            video_path = None
            if self.settings.audio_enabled:
                job.stage = PipelineStage.audio_generation
                self.store.save(job)
                audio_path = self.audio_generator.generate(
                    script,
                    output_dir,
                    language=job.request.language,
                )
                artifacts["audio"] = str(audio_path.relative_to(self.settings.data_dir))
                next_tasks.remove("audio_generation")
            if self.settings.video_enabled:
                job.stage = PipelineStage.video_generation
                self.store.save(job)
                video_path = self.video_generator.generate(script, output_dir)
                artifacts["video"] = str(video_path.relative_to(self.settings.data_dir))
                next_tasks.remove("video_generation")

            if audio_path and video_path:
                job.stage = PipelineStage.media_compilation
                self.store.save(job)
                captions_path = self.media_compiler.generate_captions(
                    script,
                    output_dir / "audio-manifest.json",
                    output_dir,
                )
                final_video_path = self.media_compiler.compile(
                    video_path=video_path,
                    audio_path=audio_path,
                    captions_path=captions_path,
                    output_dir=output_dir,
                )
                artifacts["captions"] = str(captions_path.relative_to(self.settings.data_dir))
                artifacts["final_video"] = str(final_video_path.relative_to(self.settings.data_dir))

            job.status = JobStatus.completed
            job.stage = PipelineStage.completed
            job.result = {
                "script": script.model_dump(),
                "artifacts": artifacts,
                "next_tasks": next_tasks,
            }
            self.store.save(job)
        except Exception as exc:
            job.status = JobStatus.failed
            job.stage = PipelineStage.failed
            job.error = str(exc)
            self.store.save(job)
