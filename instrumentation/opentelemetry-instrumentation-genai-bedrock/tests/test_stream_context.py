# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

from opentelemetry import context, trace
from opentelemetry.instrumentation.genai.bedrock.stream import (
    AsyncBedrockAgentEventStreamWrapper,
    AsyncBedrockConverseStreamWrapper,
    AsyncBedrockInvokeModelStreamWrapper,
    AsyncBedrockRetrieveAndGenerateStreamWrapper,
    BedrockAgentEventStreamWrapper,
    BedrockConverseStreamWrapper,
    BedrockInvokeModelStreamWrapper,
    BedrockRetrieveAndGenerateStreamWrapper,
)
from opentelemetry.util.genai.handler import TelemetryHandler

STREAM_WRAPPERS = [
    pytest.param(BedrockConverseStreamWrapper, False, id="converse-sync"),
    pytest.param(AsyncBedrockConverseStreamWrapper, True, id="converse-async"),
    pytest.param(
        BedrockInvokeModelStreamWrapper, False, id="invoke-model-sync"
    ),
    pytest.param(
        AsyncBedrockInvokeModelStreamWrapper,
        True,
        id="invoke-model-async",
    ),
    pytest.param(BedrockAgentEventStreamWrapper, False, id="agent-sync"),
    pytest.param(AsyncBedrockAgentEventStreamWrapper, True, id="agent-async"),
    pytest.param(
        BedrockRetrieveAndGenerateStreamWrapper,
        False,
        id="retrieve-and-generate-sync",
    ),
    pytest.param(
        AsyncBedrockRetrieveAndGenerateStreamWrapper,
        True,
        id="retrieve-and-generate-async",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("wrapper_cls", "asynchronous"), STREAM_WRAPPERS)
@pytest.mark.parametrize("ending", ["drain", "close", "error"])
async def test_stream_activates_invocation_only_during_reads(
    tracer_provider,
    span_exporter,
    caplog,
    wrapper_cls: type,
    asynchronous: bool,
    ending: str,
) -> None:
    before = context.get_current()
    invocation = TelemetryHandler(
        tracer_provider=tracer_provider,
    ).inference("bedrock", request_model="test-model")
    error = ConnectionError("stream failed")
    chunk: dict[str, Any] = {}

    def assert_invocation_is_current() -> None:
        assert (
            trace.get_current_span().get_span_context()
            == trace.get_current_span(invocation.context).get_span_context()
        )

    def produce() -> Iterator[dict[str, Any]]:
        assert_invocation_is_current()
        yield chunk
        assert_invocation_is_current()
        if ending == "error":
            raise error

    async def aproduce() -> AsyncIterator[dict[str, Any]]:
        for item in produce():
            await asyncio.sleep(0)
            yield item

    source = aproduce() if asynchronous else produce()
    stream = wrapper_cls(source, invocation)

    assert context.get_current() is before
    if asynchronous:
        assert await asyncio.create_task(anext(stream)) is chunk
    else:
        assert next(stream) is chunk
    assert context.get_current() is before
    assert span_exporter.get_finished_spans() == ()

    if ending == "close":
        if asynchronous:
            await stream.aclose()
        else:
            stream.close()
    elif ending == "error":
        with pytest.raises(ConnectionError) as raised:
            if asynchronous:
                await anext(stream)
            else:
                next(stream)
        assert raised.value is error
    elif asynchronous:
        assert [item async for item in stream] == []
    else:
        assert list(stream) == []

    assert context.get_current() is before
    assert len(span_exporter.get_finished_spans()) == 1
    assert not [
        record
        for record in caplog.records
        if record.name == "opentelemetry.context" and record.levelno >= 40
    ]
