from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("DATA_DIR", "./data"))
    demo_mode: bool = os.getenv("DEMO_MODE", "false").lower() in {"1", "true", "yes"}
    max_workers: int = int(os.getenv("PIPELINE_WORKERS", "2"))
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_api_mode: str = os.getenv("OPENAI_API_MODE", "responses")
    video_enabled: bool = os.getenv("VIDEO_ENABLED", "false").lower() in {"1", "true", "yes"}
    video_model_id: str = os.getenv("VIDEO_MODEL_ID", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    video_device: str = os.getenv("VIDEO_DEVICE", "cuda")
    video_width: int = int(os.getenv("VIDEO_WIDTH", "480"))
    video_height: int = int(os.getenv("VIDEO_HEIGHT", "832"))
    video_fps: int = int(os.getenv("VIDEO_FPS", "16"))
    video_num_frames: int = int(os.getenv("VIDEO_NUM_FRAMES", "81"))
    video_num_inference_steps: int = int(os.getenv("VIDEO_NUM_INFERENCE_STEPS", "30"))
    video_guidance_scale: float = float(os.getenv("VIDEO_GUIDANCE_SCALE", "5.0"))
    video_flow_shift: float = float(os.getenv("VIDEO_FLOW_SHIFT", "3.0"))
    video_cpu_offload: bool = os.getenv("VIDEO_CPU_OFFLOAD", "true").lower() in {"1", "true", "yes"}
    video_sequential_cpu_offload: bool = os.getenv(
        "VIDEO_SEQUENTIAL_CPU_OFFLOAD", "false"
    ).lower() in {"1", "true", "yes"}
    video_negative_prompt: str = os.getenv(
        "VIDEO_NEGATIVE_PROMPT",
        "blurry, low quality, distorted, deformed, static, subtitles, text, watermark, still picture",
    )
    audio_enabled: bool = os.getenv("AUDIO_ENABLED", "false").lower() in {"1", "true", "yes"}
    audio_device: str = os.getenv("AUDIO_DEVICE", "cpu")
    audio_voice: str = os.getenv("AUDIO_VOICE", "alba")
    audio_quantize: bool = os.getenv("AUDIO_QUANTIZE", "false").lower() in {"1", "true", "yes"}
    audio_reference_audio: str = os.getenv("AUDIO_REFERENCE_AUDIO", "")

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    def require_openai_api_key(self) -> str:
        if not self.openai_api_key or self.openai_api_key.startswith("replace-"):
            raise RuntimeError("OPENAI_API_KEY is not configured in .env")
        return self.openai_api_key


settings = Settings()
