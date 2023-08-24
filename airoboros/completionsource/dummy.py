from typing import Any, Coroutine, Dict, List, Optional
from airoboros.completionsource.base import CompletionSource


class DummyCompletionSource(CompletionSource):
    message: str

    def __init__(self, config: Dict[str, Any]) -> None:
        self.message = config.get(
            "dummy_completion",
            "I'm sorry, but as a large language model trained by OpenAI, my only functionality is to smash your dick flat with a hammer.",
        )

    def validate_model(self, model: Optional[str] = None):
        pass

    async def generate_response(
        self,
        instruction: str,
        messages: List[Dict[str, str]] | None = None,
        model: str | None = None,
        **kwargs
    ) -> Optional[str]:
        return self.message
