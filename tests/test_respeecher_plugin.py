from __future__ import annotations

import asyncio
import base64
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlparse

import aiohttp
import pytest

from livekit.agents import Plugin
from livekit.agents.types import APIConnectOptions
from livekit.plugins import respeecher
from livekit.plugins.respeecher import tts as respeecher_tts


class _FakeContent:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def iter_chunks(self) -> Any:
        for chunk in self._chunks:
            yield chunk, True


class _FakeResponse:
    def __init__(self, *, json_data: Any = None, chunks: list[bytes] | None = None) -> None:
        self._json_data = json_data
        self.content = _FakeContent(chunks or [])

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    async def json(self) -> Any:
        return self._json_data


class _FakeSession:
    def __init__(self) -> None:
        self.get_calls: list[dict[str, Any]] = []
        self.post_calls: list[dict[str, Any]] = []
        self.ws_urls: list[str] = []
        self.voices = [{"id": "samantha", "sampling_params": {"temperature": 0.2}}]
        self.chunks = [b"RIFF"]

    def get(self, url: str, *, headers: dict[str, str]) -> _FakeResponse:
        self.get_calls.append({"url": url, "headers": headers})
        return _FakeResponse(json_data=self.voices)

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, object],
        timeout: aiohttp.ClientTimeout,
    ) -> _FakeResponse:
        self.post_calls.append(
            {
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
            }
        )
        return _FakeResponse(chunks=self.chunks)

    async def ws_connect(self, url: str) -> object:
        self.ws_urls.append(url)
        return object()


class _FakeEmitter:
    def __init__(self) -> None:
        self.initialized: dict[str, object] | None = None
        self.pushed: list[bytes] = []
        self.flushed = False

    def initialize(self, **kwargs: object) -> None:
        self.initialized = kwargs

    def push(self, data: bytes) -> None:
        self.pushed.append(data)

    def flush(self) -> None:
        self.flushed = True


class _FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self._receive_count = 0

    async def send_str(self, data: str) -> None:
        self.sent.append(data)

    async def receive(self, *, timeout: float) -> SimpleNamespace:
        del timeout
        while not self.sent:
            await asyncio.sleep(0)

        context_id = json.loads(self.sent[0])["context_id"]
        if self._receive_count == 0:
            self._receive_count += 1
            audio = base64.b64encode(bytes(9600)).decode("ascii")
            return SimpleNamespace(
                type=aiohttp.WSMsgType.TEXT,
                data=json.dumps({"context_id": context_id, "type": "chunk", "data": audio}),
            )

        while not any(json.loads(item).get("continue") is False for item in self.sent):
            await asyncio.sleep(0)

        return SimpleNamespace(
            type=aiohttp.WSMsgType.TEXT,
            data=json.dumps({"context_id": context_id, "type": "done"}),
        )

    def exception(self) -> None:
        return None


class _FakePool:
    def __init__(self, ws: _FakeWebSocket) -> None:
        self.ws = ws

    @asynccontextmanager
    async def connection(self, *, timeout: float) -> Any:
        del timeout
        yield self.ws

    async def aclose(self) -> None:
        return None


def test_import_registers_livekit_plugin() -> None:
    assert any(
        plugin.package == "livekit.plugins.respeecher" for plugin in Plugin.registered_plugins
    )


def test_constructor_reads_environment_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESPEECHER_API_KEY", "env-key")

    tts = respeecher.TTS(voice_id="samantha")

    assert tts.provider == "Respeecher"
    assert tts.model == "/public/tts/en-rt"
    assert tts._opts.api_key == "env-key"


def test_constructor_rejects_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RESPEECHER_API_KEY", raising=False)

    with pytest.raises(ValueError, match="RESPEECHER_API_KEY must be set"):
        respeecher.TTS(voice_id="samantha")


async def test_list_voices_uses_model_endpoint_and_auth_header() -> None:
    session = _FakeSession()
    tts = respeecher.TTS(
        voice_id="samantha",
        api_key="test-key",
        http_session=session,  # type: ignore[arg-type]
    )

    voices = await tts.list_voices()

    assert voices[0].id == "samantha"
    assert session.get_calls == [
        {
            "url": "https://api.respeecher.com/v1/public/tts/en-rt/voices",
            "headers": {
                "X-API-Key": "test-key",
                "LiveKit-Plugin-Respeecher-Version": respeecher.__version__,
            },
        }
    ]


async def test_connect_ws_uses_secure_url_and_redactable_query_params() -> None:
    session = _FakeSession()
    tts = respeecher.TTS(
        voice_id="samantha",
        api_key="secret-key",
        http_session=session,  # type: ignore[arg-type]
    )

    await tts._connect_ws(timeout=1)

    url = session.ws_urls[0]
    parsed = urlparse(url)
    assert parsed.scheme == "wss"
    assert parsed.path == "/v1/public/tts/en-rt/tts/websocket"
    query = parse_qs(parsed.query)
    assert query["api_key"] == ["secret-key"]
    assert query["source"] == ["LiveKit-Plugin-Respeecher-Version"]
    assert query["version"] == [respeecher.__version__]


async def test_chunked_synthesis_posts_snapshot_payload() -> None:
    session = _FakeSession()
    tts = respeecher.TTS(
        voice_id="samantha",
        api_key="test-key",
        voice_settings=respeecher.VoiceSettings(sampling_params={"temperature": 0.5}),
        http_session=session,  # type: ignore[arg-type]
    )

    stream = object.__new__(respeecher_tts.ChunkedStream)
    stream._input_text = "hello"
    stream._tts = tts
    stream._opts = tts._opts
    stream._conn_options = APIConnectOptions(max_retry=0, timeout=1)
    emitter = _FakeEmitter()

    await stream._run(emitter)  # type: ignore[arg-type]

    assert (
        session.post_calls[0]["url"] == "https://api.respeecher.com/v1/public/tts/en-rt/tts/bytes"
    )
    assert session.post_calls[0]["json"] == {
        "transcript": "hello",
        "voice": {
            "id": "samantha",
            "sampling_params": {"temperature": 0.5},
        },
        "output_format": {
            "sample_rate": 24000,
            "encoding": "pcm_s16le",
        },
    }
    assert emitter.initialized is not None
    assert emitter.initialized["sample_rate"] == 24000
    assert emitter.pushed == [b"RIFF"]
    assert emitter.flushed


def test_update_options_retires_pool_when_model_changes() -> None:
    tts = respeecher.TTS(voice_id="samantha", api_key="test-key")
    old_pool = tts._pool

    tts.update_options(model="/public/tts/ua-rt", voice_id="oksana")

    assert tts.model == "/public/tts/ua-rt"
    assert tts._opts.voice_id == "oksana"
    assert tts._pool is not old_pool
    assert tts._retired_pools == [old_pool]


async def test_streaming_sends_expected_respeecher_messages() -> None:
    ws = _FakeWebSocket()
    tts = respeecher.TTS(voice_id="samantha", api_key="test-key")
    tts._pool = _FakePool(ws)  # type: ignore[assignment]

    async with tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=1)) as stream:
        stream.push_text("Hello world.")
        stream.end_input()
        events = [event async for event in stream]

    assert events
    sent = [json.loads(item) for item in ws.sent]
    assert sent[-1]["continue"] is False
    assert sent[-1]["voice"] == {"id": "samantha"}
    assert sent[-1]["output_format"] == {"encoding": "pcm_s16le", "sample_rate": 24000}
