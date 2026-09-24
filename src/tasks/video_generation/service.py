from pathlib import Path
import json
import shutil
import subprocess
from threading import Lock
from typing import Any

from ...config import Settings
from ..script_generation.models import GeneratedScript


class VideoGenerator:
    """Generate silent scene clips with Wan 2.1 T2V 1.3B through Diffusers.

    The heavyweight ML dependencies are imported only when generation starts so
    the API can still run for script-only jobs on machines without a GPU.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._pipeline: Any | None = None
        self._pipeline_lock = Lock()

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        with self._pipeline_lock:
            if self._pipeline is not None:
                return self._pipeline

            try:
                import torch
                from diffusers import AutoencoderKLWan, WanPipeline
                from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
            except ImportError as exc:
                raise RuntimeError(
                    "Video generation dependencies are missing. Install requirements-video.txt."
                ) from exc

            device = self.settings.video_device
            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("VIDEO_DEVICE=cuda but no CUDA GPU is available")

            dtype = torch.bfloat16 if device != "cpu" else torch.float32
            model_id = self.settings.video_model_id
            local_model = Path(model_id)
            if not local_model.exists() and (model_id.startswith(".") or model_id.startswith("/")):
                raise RuntimeError(
                    f"VIDEO_MODEL_ID local directory does not exist: {local_model}. "
                    "Download Wan2.1-T2V-1.3B-Diffusers into this path."
                )
            if local_model.is_dir() and not (local_model / "model_index.json").exists():
                raise RuntimeError(
                    f"VIDEO_MODEL_ID points to a non-Diffusers Wan checkpoint: {local_model}. "
                    "This adapter expects the Wan2.1-T2V-1.3B-Diffusers directory "
                    "containing model_index.json. Download the Diffusers variant or "
                    "use the original Wan inference implementation."
                )
            vae = AutoencoderKLWan.from_pretrained(
                model_id,
                subfolder="vae",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            pipeline = WanPipeline.from_pretrained(
                model_id,
                vae=vae,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                offload_state_dict=True,
            )
            pipeline.scheduler = UniPCMultistepScheduler.from_config(
                pipeline.scheduler.config,
                flow_shift=self.settings.video_flow_shift,
            )

            if device == "cuda" and self.settings.video_cpu_offload:
                if self.settings.video_sequential_cpu_offload:
                    pipeline.enable_sequential_cpu_offload()
                else:
                    pipeline.enable_model_cpu_offload()
            else:
                pipeline.to(device)

            self._pipeline = pipeline
            return pipeline

    def _ffmpeg(self) -> str:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            return ffmpeg
        try:
            from imageio_ffmpeg import get_ffmpeg_exe
            return get_ffmpeg_exe()
        except ImportError as exc:
            raise RuntimeError("ffmpeg is required to combine generated video clips") from exc

    def _combine_clips(self, clips: list[Path], output_path: Path) -> None:
        if len(clips) == 1:
            shutil.copy2(clips[0], output_path)
            return

        concat_file = output_path.parent / "concat.txt"
        entries = [f"file '{clip.resolve().as_posix().replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'" for clip in clips]
        concat_file.write_text("\n".join(entries) + "\n", encoding="utf-8")
        try:
            result = subprocess.run(
                [self._ffmpeg(), "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-c", "copy", str(output_path)],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg could not combine clips: {result.stderr[-1000:]}")
        finally:
            concat_file.unlink(missing_ok=True)

    def generate(self, script: GeneratedScript, output_dir: Path) -> Path:
        try:
            from diffusers.utils import export_to_video
        except ImportError as exc:
            raise RuntimeError(
                "Video generation dependencies are missing. Install requirements-video.txt."
            ) from exc

        pipeline = self._load_pipeline()
        clips_dir = output_dir / "video_clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        clips: list[Path] = []
        manifest: list[dict[str, Any]] = []

        for scene in script.scenes:
            prompt = (
                f"{script.title}. {scene.visual_direction} "
                "Cinematic short-form video, natural motion, coherent subject, detailed lighting, "
                "vertical composition, no on-screen text."
            )
            clip_path = clips_dir / f"scene_{scene.scene_number:03d}.mp4"
            output = pipeline(
                prompt=prompt,
                negative_prompt=self.settings.video_negative_prompt,
                width=self.settings.video_width,
                height=self.settings.video_height,
                num_frames=self.settings.video_num_frames,
                num_inference_steps=self.settings.video_num_inference_steps,
                guidance_scale=self.settings.video_guidance_scale,
            )
            export_to_video(output.frames[0], str(clip_path), fps=self.settings.video_fps)
            clips.append(clip_path)
            manifest.append({
                "scene_number": scene.scene_number,
                "prompt": prompt,
                "clip": str(clip_path.name),
                "frames": self.settings.video_num_frames,
            })

        video_path = output_dir / "video.mp4"
        self._combine_clips(clips, video_path)
        (output_dir / "video-manifest.json").write_text(json.dumps({
            "model": self.settings.video_model_id,
            "width": self.settings.video_width,
            "height": self.settings.video_height,
            "fps": self.settings.video_fps,
            "clips": manifest,
            "audio": "not included; audio_generation is a separate task",
        }, indent=2), encoding="utf-8")
        return video_path
