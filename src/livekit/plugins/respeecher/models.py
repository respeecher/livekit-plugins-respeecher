from dataclasses import dataclass
from typing import Any, Literal

TTSModels = Literal[
    # Respeecher's public English model
    "/public/tts/en-rt",
    # Respeecher's public Ukrainian model
    "/public/tts/ua-rt",
]

TTSEncoding = Literal["pcm_s16le"]

SamplingParams = dict[str, Any]


@dataclass
class VoiceSettings:
    """Voice settings for Respeecher TTS.

    See https://space.respeecher.com/docs/api/tts/sampling-params-guide for details.
    """

    sampling_params: SamplingParams | None = None


class Voice(dict[str, Any]):
    """Voice model returned by Respeecher.

    The API may include additional fields, so this type behaves like a dictionary while
    guaranteeing an `id` property for common usage.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if "id" not in self:
            raise ValueError("Voice must have an 'id' field")

    @property
    def id(self) -> str:
        return str(self["id"])

    @property
    def sampling_params(self) -> SamplingParams | None:
        sampling_params = self.get("sampling_params")
        return sampling_params if isinstance(sampling_params, dict) else None
