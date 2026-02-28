import json

from typing import Any
from pydantic import BaseModel, ConfigDict

from abc import ABC, abstractmethod
from ubot.tools.base import Tool


class ToolCall(BaseModel):
    tool: Tool
    tool_call_id: str
    params: dict[str, Any]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ToolTemplate(ABC):
    def __init__(self, tools: list[Tool]):
        self.tools = tools
        self.tools_mapping = {tool.name: tool for tool in tools}

    @abstractmethod
    def generate_prompt(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    # TODO: define a more robust definition of tool call in future
    # for now, we use list[any] which is not a good idea
    def parse_tool_calls(self, tool_calls: list[Any]) -> list[ToolCall]:
        raise NotImplementedError


class OpenAIToolTemplate(ToolTemplate):
    def generate_prompt(self) -> list[dict[str, Any]]:
        return [tool.schema for tool in self.tools]

    def parse_tool_calls(self, tool_calls: list[Any]) -> list[ToolCall]:
        parsed_tool_calls = []
        for tool_call in tool_calls:
            name = tool_call.function.name

            try:
                params = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                params = {}

            if name in self.tools_mapping:
                tool = self.tools_mapping[name]
                parsed_tool_calls.append(ToolCall(tool=tool, tool_call_id=tool_call.id, params=params))
            else:
                # TODO: better handle tool not found error
                raise NameError(f"Tool {name} not found in tool registry.")

        return parsed_tool_calls
