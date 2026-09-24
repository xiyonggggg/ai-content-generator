# AI Automatic Content Creator

This backend accepts a natural-language brief, tracks an asynchronous job, and uses the OpenAI Python SDK to generate a validated short-form script. It can generate Pocket TTS voiceover and Wan 2.1 video as separate local tasks. Publishing remains separate.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Install this as well on the GPU machine that will render video:
pip install -r requirements-video.txt
# Install this as well on the GPU machine that will synthesize audio:
pip install -r requirements-audio.txt
# Install this as well for media assembly when system ffmpeg is unavailable:
pip install -r requirements-media.txt
uvicorn main:app --reload
```

Configure OpenAI in `.env`:

```dotenv
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
OPENAI_API_MODE=responses
```

For Gemini's OpenAI-compatible endpoint, use `OPENAI_API_MODE=chat_completions`.

Create a job:

```bash
curl -X POST http://127.0.0.1:8000/jobs \
  -H 'content-type: application/json' \
  -d '{"prompt":"3 practical ways to save money on groceries", "platforms":["youtube","tiktok"]}'
```

Poll the returned job ID at `GET /jobs/{id}`. The generated script is stored under `data/jobs/{id}/script.json`.

## Enable Wan video generation

Wan 2.1 T2V 1.3B uses the Diffusers model layout in this application. Download the Diffusers variant (the original `Wan2.1-T2V-1.3B` checkpoint directory is not interchangeable):

```bash
pip install -r requirements-video.txt
hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --local-dir models/video/Wan2.1-T2V-1.3B-Diffusers
```

Set this in `.env` on a CUDA machine:

```dotenv
VIDEO_ENABLED=true
VIDEO_MODEL_ID=./models/video/Wan2.1-T2V-1.3B-Diffusers
VIDEO_DEVICE=cuda
VIDEO_CPU_OFFLOAD=true
VIDEO_SEQUENTIAL_CPU_OFFLOAD=true
```

The implementation generates one silent MP4 clip per script scene, then concatenates them into `data/jobs/{id}/video.mp4`. The default is 480×832 vertical output, 16 FPS, and 81 frames per scene. `81` follows Wan's recommended `4*k+1` frame format and is approximately five seconds at 16 FPS. Increase `VIDEO_NUM_FRAMES` only when the GPU has enough memory.

The Wan 2.1 project recommends 480P for the 1.3B model because 720P is less stable, and documents memory-saving options for OOM situations. `VIDEO_SEQUENTIAL_CPU_OFFLOAD=true` uses the slower, lower-GPU-memory offload mode. [Wan 2.1 official usage](https://github.com/Wan-Video/Wan2.1), [Wan Diffusers pipeline documentation](https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/wan.md)

For the first isolated test, use the included one-scene smoke test. It uses 17 frames and 8 inference steps to reduce runtime:

```bash
python temp_video.py
```

If the process is still killed while `Loading pipeline components` is running, it is a WSL host-RAM OOM. Check `free -h` and `dmesg -T | grep -i oom`; configure WSL with at least 16 GB RAM and 8 GB swap, then run `wsl --shutdown` from Windows PowerShell before retrying. The 8 GB GPU alone is not enough to load the T5 checkpoint.

## Enable Pocket TTS voiceover

Pocket TTS is a small CPU-oriented TTS model, so it does not require the GPU or the large IndexTTS checkpoints. Install it into the same virtual environment as this application:

```bash
pip install -r requirements-audio.txt
```

Then set this in `.env`:

```dotenv
AUDIO_ENABLED=true
AUDIO_DEVICE=cpu
AUDIO_VOICE=alba
AUDIO_QUANTIZE=false
AUDIO_REFERENCE_AUDIO=
```

The task synthesizes each script scene into `scene_audio/scene_*.wav`, concatenates them into `voiceover.wav`, and writes `audio-manifest.json` with the actual duration of every scene. Set `AUDIO_REFERENCE_AUDIO` to a local WAV file for voice cloning, or leave it empty to use the configured Pocket TTS voice such as `alba`. Pocket TTS supports English, French, German, Portuguese, Italian, and Spanish. [Pocket TTS official repository and API](https://github.com/kyutai-labs/pocket-tts)

## Compile final media with FFmpeg

When both `AUDIO_ENABLED=true` and `VIDEO_ENABLED=true`, the pipeline automatically:

1. Generates an SRT caption file from the real scene audio durations.
2. Muxes `video.mp4` and `voiceover.wav`.
3. Burns captions into the final MP4 when the FFmpeg subtitles filter is available.
4. Falls back to a selectable subtitle track when burned-in captions are unavailable.

The final artifact is `data/jobs/{id}/final_video.mp4`. Compilation metadata is saved in `compile-manifest.json`.

Task code is separated under `src/tasks/`:

- `script_generation/`: OpenAI SDK script generation
- `audio_generation/`: Pocket TTS scene voiceover generation
- `video_generation/`: Wan 2.1 scene video generation and concatenation
- `media_compilation/`: FFmpeg audio, video, and caption assembly

## Production roadmap

1. Replace scene-level captions with word-level timestamps.
2. Add moderation, human approval, retries, and idempotency.
3. Implement OAuth-based YouTube publishing and TikTok Content Posting API publishing; store refresh tokens encrypted and request the minimum scopes.
4. Move job state to a durable queue/database (for example Redis + a worker and Postgres) before multi-user deployment.
