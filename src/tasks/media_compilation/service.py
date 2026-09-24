from pathlib import Path
import json
import shutil
import subprocess
import textwrap
from typing import Any

from ..script_generation.models import GeneratedScript


class MediaCompiler:
    """Create captions and combine the generated media into a final MP4."""

    @staticmethod
    def _ffmpeg() -> str:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            return ffmpeg
        try:
            from imageio_ffmpeg import get_ffmpeg_exe
            return get_ffmpeg_exe()
        except ImportError as exc:
            raise RuntimeError(
                "FFmpeg is required. Install requirements-media.txt or install ffmpeg on the system."
            ) from exc

    @staticmethod
    def _timestamp(seconds: float) -> str:
        milliseconds = max(0, round(seconds * 1000))
        hours, remainder = divmod(milliseconds, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        seconds_part, millis = divmod(remainder, 1_000)
        return f"{hours:02d}:{minutes:02d}:{seconds_part:02d},{millis:03d}"

    def generate_captions(
        self,
        script: GeneratedScript,
        audio_manifest_path: Path,
        output_dir: Path,
    ) -> Path:
        if not audio_manifest_path.exists():
            raise RuntimeError(f"Audio manifest not found: {audio_manifest_path}")

        manifest = json.loads(audio_manifest_path.read_text(encoding="utf-8"))
        durations = {
            int(item["scene_number"]): float(item["duration_seconds"])
            for item in manifest.get("scenes", [])
        }
        captions = []
        cursor = 0.0
        for index, scene in enumerate(script.scenes, start=1):
            duration = durations.get(scene.scene_number)
            if duration is None:
                raise RuntimeError(f"Missing audio duration for scene {scene.scene_number}")
            end = cursor + duration
            caption_text = "\n".join(textwrap.wrap(scene.voiceover, width=42))
            captions.extend([
                str(index),
                f"{self._timestamp(cursor)} --> {self._timestamp(end)}",
                caption_text,
                "",
            ])
            cursor = end

        captions_path = output_dir / "captions.srt"
        captions_path.write_text("\n".join(captions), encoding="utf-8")
        return captions_path

    @staticmethod
    def _subtitle_filter_path(captions_path: Path) -> str:
        # Escape characters interpreted by the FFmpeg subtitles filter.
        return (
            str(captions_path.resolve())
            .replace("\\", "\\\\")
            .replace(":", "\\:")
            .replace("'", "\\'")
        )

    def _run(self, command: list[str]) -> tuple[bool, str]:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        return result.returncode == 0, result.stderr[-1500:]

    def compile(
        self,
        video_path: Path,
        audio_path: Path,
        captions_path: Path,
        output_dir: Path,
    ) -> Path:
        for path in (video_path, audio_path, captions_path):
            if not path.exists():
                raise RuntimeError(f"Compilation input not found: {path}")

        output_path = output_dir / "final_video.mp4"
        ffmpeg = self._ffmpeg()
        subtitle_filter = f"subtitles='{self._subtitle_filter_path(captions_path)}'"
        burned_in_command = [
            ffmpeg, "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-vf", subtitle_filter,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            str(output_path),
        ]
        succeeded, stderr = self._run(burned_in_command)
        caption_mode = "burned_in"

        if not succeeded:
            # Some FFmpeg builds do not include libass/subtitles. Preserve the
            # captions as a selectable mov_text subtitle stream in that case.
            selectable_command = [
                ffmpeg, "-y",
                "-i", str(video_path),
                "-i", str(audio_path),
                "-i", str(captions_path),
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-map", "2:0",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-c:s", "mov_text",
                "-shortest",
                "-movflags", "+faststart",
                str(output_path),
            ]
            succeeded, selectable_stderr = self._run(selectable_command)
            stderr = f"burned-in attempt: {stderr}\nselectable attempt: {selectable_stderr}"
            caption_mode = "selectable_track"

        if not succeeded:
            raise RuntimeError(f"FFmpeg compilation failed: {stderr}")

        (output_dir / "compile-manifest.json").write_text(json.dumps({
            "video": video_path.name,
            "audio": audio_path.name,
            "captions": captions_path.name,
            "output": output_path.name,
            "caption_mode": caption_mode,
        }, indent=2), encoding="utf-8")
        return output_path
