import copy
import json
import os
import re
from typing import Any, Dict, Optional
from uuid import uuid4
from airoboros.completionsource.base import CompletionSource
from loguru import logger
import backoff
import aiohttp
import requests
from time import sleep

from airoboros.exceptions import (
    RateLimitError,
    TooManyRequestsError,
    TokensExhaustedError,
    ServerOverloadedError,
    ServerError,
    ContextLengthExceededError,
    BadResponseError,
)

OPENAI_API_BASE_URL = "https://api.openai.com/v1"


class OpenAICompletionSource(CompletionSource):
    model: str
    api_base: str
    api_key: str
    organization_id: Optional[str]
    used_tokens: int
    max_tokens: Optional[int]

    def __init__(self, config: Dict[str, Any]) -> None:
        self.model = config.get("model") or "gpt-4"
        self.api_key = config.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable or openai_api_key must be provided"
            )
        self.api_base = config.get("openai_api_base", OPENAI_API_BASE_URL)

        self.organization_id = config.get("organization_id")
        self.used_tokens = 0
        self.max_tokens = (
            int(config["max_tokens"]) if config.get("max_tokens") else None
        )

    @backoff.on_exception(
        backoff.fibo,
        (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            ServerError,
            RateLimitError,
            TooManyRequestsError,
            ServerOverloadedError,
        ),
        max_value=19,
    )
    async def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Perform a post request to OpenAI API.

        :param path: URL path to send request to.
        :type path: str

        :param payload: Dict containing request body/payload.
        :type payload: Dict[str, Any]

        :return: Response object.
        :rtype: Dict[str, Any]
        """
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if self.organization_id:
            headers["OpenAI-Organization"] = self.organization_id
        request_id = str(uuid4())
        logger.debug(f"POST [{request_id}] with payload {json.dumps(payload)}")
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_base}{path}",
                headers=headers,
                json=payload,
                timeout=600.0,
            ) as result:
                if result.status != 200:
                    text = await result.text()
                    logger.error(f"OpenAI request error: {text}")
                    if "too many requests" in text.lower():
                        raise TooManyRequestsError(text)
                    if (
                        "rate limit reached" in text.lower()
                        or "rate_limit_exceeded" in text.lower()
                    ):
                        sleep(30)
                        raise RateLimitError(text)
                    elif "context_length_exceeded" in text.lower():
                        raise ContextLengthExceededError(text)
                    elif "server_error" in text and "overloaded" in text.lower():
                        raise ServerOverloadedError(text)
                    elif (
                        "bad gateway" in text.lower() or "server_error" in text.lower()
                    ):
                        raise ServerError(text)
                    else:
                        raise BadResponseError(text)
                result = await result.json()
                logger.debug(f"POST [{request_id}] response: {json.dumps(result)}")
                self.used_tokens += result["usage"]["total_tokens"]
                if self.max_tokens and self.used_tokens > self.max_tokens:
                    raise TokensExhaustedError(
                        f"Max token usage exceeded: {self.used_tokens}"
                    )
                logger.debug(f"token usage: {self.used_tokens}")
                return result

    async def _post_no_exc(self, *a, **k):
        """Post, ignoring all exceptions."""
        try:
            return await self._post(*a, **k)
        except Exception as ex:
            logger.error(f"Error performing post: {ex}")
        return None

    async def generate_response(self, instruction: str, **kwargs) -> str:
        """Call OpenAI with the specified instruction and return the text response.

        :param instruction: The instruction to respond to.
        :type instruction: str

        :return: Response text.
        :rtype: str
        """
        messages = copy.deepcopy(kwargs.pop("messages", None) or [])
        model = kwargs.get("model", self.model)
        path = "/chat/completions"
        payload = {**kwargs}
        if "model" not in payload:
            payload["model"] = model
        payload["messages"] = messages
        if instruction:
            payload["messages"].append({"role": "user", "content": instruction})
        response = await self._post_no_exc(path, payload)
        if (
            not response
            or not response.get("choices")
            or response["choices"][0]["finish_reason"] == "length"
        ):
            return None
        return response["choices"][0]["message"]["content"]

    def validate_model(self, model: Optional[str] = None):
        """Ensure the specified model is available."""
        if model is None:
            model = self.model

        headers = {"Authorization": f"Bearer {self.api_key}"}
        if self.organization_id:
            headers["OpenAI-Organization"] = self.organization_id
        result = requests.get(f"{self.api_base}/models", headers=headers)
        print(f"{self.api_base}/models")
        print(repr(headers))
        if result.status_code != 200:
            raise ValueError(
                f"Invalid openai API key [{result.status_code}: {result.text}]"
            )
        available = {item["id"] for item in result.json()["data"]}
        print(repr(available))
        if model not in available:
            raise ValueError(f"Model is not available to your API key: {model}")
        logger.success(f"Successfully validated model: {model}")
