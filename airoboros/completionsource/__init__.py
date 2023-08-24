from typing import Any, Dict, Type
from airoboros.completionsource.base import CompletionSource
from airoboros.completionsource.openai import OpenAICompletionSource
from airoboros.completionsource.dummy import DummyCompletionSource

_sources: Dict[str, Type[CompletionSource]] = {
    "openai": OpenAICompletionSource,
    "dummy": DummyCompletionSource,
}


def get(config: Dict[str, Any]) -> CompletionSource:
    source = config.get("completion_source", "openai")
    if not source in _sources:
        raise ValueError(f'Unknown completion source "{source}"')

    return _sources[source](config)
