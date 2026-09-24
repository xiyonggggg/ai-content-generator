from pydantic import BaseModel, Field


class ScriptScene(BaseModel):
    scene_number: int = Field(ge=1)
    duration_seconds: int = Field(ge=1, le=60)
    voiceover: str = Field(min_length=1)
    on_screen_text: str = Field(min_length=1)
    visual_direction: str = Field(min_length=1)


class GeneratedScript(BaseModel):
    title: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1)
    hook: str = Field(min_length=1)
    narration: str = Field(min_length=1)
    scenes: list[ScriptScene] = Field(min_length=1)
    hashtags: list[str] = Field(min_length=1, max_length=15)
    call_to_action: str = Field(min_length=1)

