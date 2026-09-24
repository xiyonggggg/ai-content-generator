from pathlib import Path
import json
from threading import Lock
from typing import Any

from ...config import Settings
from ..script_generation.models import GeneratedScript


class AudioGenerator:
    """Generate scene voiceovers with Kyutai Pocket TTS."""

    _LANGUAGE_CODES = {
        "en": "english",
        "fr": "french",
        "de": "german",
        "pt": "portuguese",
        "it": "italian",
        "es": "spanish",
    }

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model: Any | None = None
        self._model_language: str | None = None
        self._voice_state: Any | None = None
        self._voice_source: str | None = None
        self._model_lock = Lock()

    def _load_model(self, language: str) -> Any:
        if self._model is not None and self._model_language == language:
            return self._model

        with self._model_lock:
            if self._model is not None and self._model_language == language:
                return self._model

            try:
                from pocket_tts import TTSModel
            except ImportError as exc:
                raise RuntimeError(
                    "Pocket TTS is not installed. Run: "
                    "pip install -r requirements-audio.txt"
                ) from exc

            if self.settings.audio_device != "cpu" and self.settings.audio_quantize:
                raise RuntimeError(
                    "AUDIO_QUANTIZE=true is CPU-only. Set AUDIO_DEVICE=cpu or "
                    "disable quantization."
                )

            try:
                model = TTSModel.load_model(
                    language=language,
                    quantize=self.settings.audio_quantize,
                )
                if self.settings.audio_device not in {"cpu", "auto"}:
                    import torch

                    if not torch.cuda.is_available():
                        raise RuntimeError(
                            f"AUDIO_DEVICE={self.settings.audio_device}, but CUDA is unavailable."
                        )
                    model = model.to(self.settings.audio_device)
            except TypeError as exc:
                raise RuntimeError(
                    "The installed Pocket TTS version does not support the expected "
                    "Python API. Install pocket-tts>=3.1,<4."
                ) from exc

            self._model = model
            self._model_language = language
            self._voice_state = None
            self._voice_source = None
            return model

    def _get_voice_state(self, model: Any) -> Any:
        reference_audio = self.settings.audio_reference_audio.strip()
        if reference_audio:
            reference_path = Path(reference_audio)
            if not reference_path.exists():
                raise RuntimeError(
                    f"AUDIO_REFERENCE_AUDIO does not exist: {reference_path}"
                )
            voice_source = str(reference_path)
        else:
            voice_source = self.settings.audio_voice

        if self._voice_state is None or self._voice_source != voice_source:
            self._voice_state = model.get_state_for_audio_prompt(voice_source)
            self._voice_source = voice_source
        return self._voice_state

    @staticmethod
    def _to_numpy(audio: Any) -> Any:
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.squeeze()
        return audio

    def generate(
        self,
        script: GeneratedScript,
        output_dir: Path,
        language: str = "en",
    ) -> Path:
        try:
            import numpy as np
            import soundfile as sf
        except ImportError as exc:
            raise RuntimeError(
                "Audio generation dependencies are missing. Install requirements-audio.txt."
            ) from exc

        language_code = self._LANGUAGE_CODES.get(language.lower().split("-")[0])
        if language_code is None:
            supported = ", ".join(sorted(self._LANGUAGE_CODES))
            raise RuntimeError(
                f"Pocket TTS does not support language '{language}'. "
                f"Supported languages: {supported}."
            )

        model = self._load_model(language_code)
        voice_state = self._get_voice_state(model)
        scene_dir = output_dir / "scene_audio"
        scene_dir.mkdir(parents=True, exist_ok=True)
        sample_rate = int(model.sample_rate)
        scene_manifest: list[dict[str, Any]] = []
        scene_wavs = []

        for scene in script.scenes:
            audio = model.generate_audio(
                voice_state,
                scene.voiceover,
                copy_state=True,
            )
            wav = self._to_numpy(audio).astype(np.float32, copy=False)
            scene_path = scene_dir / f"scene_{scene.scene_number:03d}.wav"
            sf.write(scene_path, wav, sample_rate)
            scene_wavs.append(wav)
            scene_manifest.append({
                "scene_number": scene.scene_number,
                "audio": scene_path.name,
                "sample_rate": sample_rate,
                "duration_seconds": round(len(wav) / sample_rate, 3),
            })

        if not scene_wavs:
            raise RuntimeError("The generated script did not contain any scenes.")

        combined = np.concatenate(scene_wavs)
        audio_path = output_dir / "voiceover.wav"
        sf.write(audio_path, combined, sample_rate)
        (output_dir / "audio-manifest.json").write_text(
            json.dumps(
                {
                    "model": "Kyutai Pocket TTS",
                    "language": language_code,
                    "device": self.settings.audio_device,
                    "voice": self._voice_source,
                    "sample_rate": sample_rate,
                    "scenes": scene_manifest,
                    "duration_seconds": round(len(combined) / sample_rate, 3),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return audio_path
