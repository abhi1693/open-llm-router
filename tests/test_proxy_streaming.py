import asyncio
import logging
from typing import Any

import httpx

from open_llm_router.gateway.proxy import BackendProxy


class _TimeoutingUpstream:
    def __init__(self) -> None:
        self.request = httpx.Request("POST", "http://upstream.example/v1/responses")

    async def aiter_lines(self) -> Any:
        if False:
            yield ""
        msg = "timed out"
        raise httpx.ReadTimeout(msg, request=self.request)


class _LineUpstream:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    async def aiter_lines(self) -> Any:
        for line in self._lines:
            yield line


async def _collect_events(upstream: Any) -> list[dict[str, Any]]:
    return [event async for event in BackendProxy._iter_sse_data_json(upstream)]


def test_iter_sse_data_json_handles_upstream_timeout(caplog: Any) -> None:
    with caplog.at_level(logging.WARNING):
        events = asyncio.run(_collect_events(_TimeoutingUpstream()))
    assert events == []
    assert "proxy_upstream_stream_error" in caplog.text


def test_iter_sse_data_json_filters_non_json_and_done_events() -> None:
    events = asyncio.run(
        _collect_events(
            _LineUpstream(
                [
                    "",
                    "event: ping",
                    "data: not-json",
                    "data: [DONE]",
                    'data: {"type":"response.output_text.delta","delta":"hello"}',
                    "data: [1,2,3]",
                ],
            ),
        ),
    )

    assert events == [{"type": "response.output_text.delta", "delta": "hello"}]
