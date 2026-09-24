from openai import OpenAI

from ...config import Settings
from ...models import ContentRequest
from .models import GeneratedScript


class ScriptGenerator:
    """Generate a validated short-form video script with the OpenAI SDK."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        if self.client is None:
            self.client = OpenAI(
                api_key=self.settings.require_openai_api_key(),
                base_url=self.settings.openai_base_url,
            )
        return self.client

    def generate(self, request: ContentRequest) -> GeneratedScript:
        system_prompt = (
            "You are an expert short-form video scriptwriter. "
            "Turn the user's natural-language brief into an engaging, accurate script "
            "for YouTube Shorts and TikTok. Keep the narration natural to speak aloud. "
            "Plan a strong first-second hook, clear scene-by-scene visuals, concise on-screen text, "
            "and a useful call to action. Do not invent specific facts when the brief does not provide "
            "enough information; keep claims general or phrase them as suggestions."
        )
        user_prompt = (
            f"Brief: {request.prompt}\n"
            f"Target duration: {request.duration_seconds} seconds\n"
            f"Language: {request.language}\n"
            f"Tone: {request.tone}\n"
            "Return a complete script suitable for the requested duration."
        )

        if self.settings.openai_api_mode == "chat_completions":
            response = self._get_client().chat.completions.parse(
                model=self.settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=GeneratedScript,
            )
            parsed_script = response.choices[0].message.parsed
        else:
            response = self._get_client().responses.parse(
                model=self.settings.openai_model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                text_format=GeneratedScript,
            )
            parsed_script = response.output_parsed

        if parsed_script is None:
            raise RuntimeError("OpenAI returned no parsed script")
        return parsed_script
