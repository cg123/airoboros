from abc import ABC
from typing import Any, Dict, List, Optional


class CompletionSource(ABC):
    def __init__(self, config: Dict[str, Any]) -> None:
        ...

    def validate_model(self, model: Optional[str] = None):
        ...

    async def generate_response(
        self,
        instruction: str,
        messages: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        ...
