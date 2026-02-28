import openai

from typing import Any
from pydantic import BaseModel, Field, ConfigDict

from abc import ABC, abstractmethod


class ModelConfig(BaseModel):
    client_config: dict = Field(default_factory=dict)
    request_config: dict = Field(default_factory=dict)


class LLMResponse(BaseModel):
    message: Any
    content: str
    reasoning_content: str
    tool_calls: list = Field(default_factory=list)
    meta: dict = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class LLM(ABC):
    def __init__(self, model_config: ModelConfig):
        self.client_config = model_config.client_config
        self.request_config = model_config.request_config

        self._build_model()

    @abstractmethod
    def _build_model(self):
        raise NotImplementedError

    @abstractmethod
    async def call(self, messages: list[dict[str, Any]], **kwargs: Any) -> LLMResponse:
        raise NotImplementedError


# TODO: minimal example, using openai compatible model
# in future, split into multiple different files to support different types of models
class OpenAI(LLM):
    def _build_model(self):
        self.model = openai.AsyncOpenAI(**self.client_config)

    async def call(self, messages: list[dict[str, Any]], **kwargs: Any) -> LLMResponse:
        # TODO: build messages from prompt, use more robust definition of message in future
        response = await self.model.chat.completions.create(
            messages=messages,
            **self.request_config,
            **kwargs,
        )

        message = response.choices[0].message
        content = response.choices[0].message.content
        reasoning_content = response.choices[0].message.reasoning_content
        tool_calls = response.choices[0].message.tool_calls if response.choices[0].message.tool_calls else []
        meta = response.usage

        resp = LLMResponse(
            message=message,
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            meta=meta.model_dump(),
        )
        return resp


class AzureOpenAI(LLM):
    def _build_model(self):
        self.model = openai.AsyncAzureOpenAI(**self.client_config)

    async def call(self, messages: list[dict[str, Any]], **kwargs: Any) -> LLMResponse:
        response = await self.model.chat.completions.create(
            messages=messages,
            **self.request_config,
            **kwargs,
        )

        message = response.choices[0].message
        content = response.choices[0].message.content
        reasoning_content = response.choices[0].message.reasoning_content
        tool_calls = response.choices[0].message.tool_calls if response.choices[0].message.tool_calls else []
        meta = response.usage

        resp = LLMResponse(
            message=message,
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            meta=meta.model_dump(),
        )
        return resp
