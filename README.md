# Respeecher LiveKit Plugin

Respeecher TTS integration for [LiveKit Agents](https://github.com/livekit/agents).

This package is maintained by Respeecher. It uses the normal LiveKit plugin import path,
but it is distributed outside the LiveKit monorepo while the upstream PR is under review.

## Installation

```bash
pip install livekit-agents livekit-plugins-respeecher
```

## Configure

```bash
export LIVEKIT_URL="wss://..."
export LIVEKIT_API_KEY="..."
export LIVEKIT_API_SECRET="..."
export RESPEECHER_API_KEY="..."
```

`RESPEECHER_API_KEY` is used when `api_key` is not passed to the constructor; you can
also pass it directly: `respeecher.TTS(api_key="...", voice_id="...")`.

## Usage

```python
from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import respeecher

load_dotenv()


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    session = AgentSession(
        tts=respeecher.TTS(voice_id="marta"),
    )

    await session.start(
        room=ctx.room,
        agent=Agent(instructions="You are a concise voice assistant."),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

`voice_id` is required because each Respeecher model exposes its own voice list. The
plugin streams over WebSocket by default; chunked HTTP synthesis is also available via
`TTS.synthesize(text)`.

## Discover Voices

```python
import asyncio

from livekit.plugins import respeecher


async def main() -> None:
    tts = respeecher.TTS(voice_id="placeholder")
    voices = await tts.list_voices()
    for voice in voices:
        print(voice.id)


asyncio.run(main())
```

Use a `voice_id` returned by `list_voices()` for the model you plan to use.

## Overriding Sampling Parameters

See the [Sampling Parameters Guide](https://space.respeecher.com/docs/api/tts/sampling-params-guide).

```python
from livekit.plugins import respeecher

tts = respeecher.TTS(
    voice_id="marta",
    voice_settings=respeecher.VoiceSettings(
        sampling_params={
            "min_p": 0.01,
        },
    ),
)
```

## Models

See [Models & Languages](https://space.respeecher.com/docs/models-and-languages).
Supported models:

- `/public/tts/en-rt` (default)
- `/public/tts/ua-rt`

To use the Ukrainian model, pass it explicitly:

```python
from livekit.plugins import respeecher

tts = respeecher.TTS(
    voice_id="olesia-conversation",
    model="/public/tts/ua-rt",
)
```

## Compatibility

Requires `livekit-agents>=1.5,<2`.

## Support

For plugin issues, open an issue in this repository; for LiveKit Agents itself, use the
[LiveKit repo](https://github.com/livekit/agents). If this plugin helps your use case,
feedback on the [upstream PR](https://github.com/livekit/agents/pull/3233) is appreciated.
