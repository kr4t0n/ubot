import inspect
import logging
import subprocess

from typing import Any
from pydantic import BaseModel, Field

from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)


class ToolResponse(BaseModel):
    content: str | list[str]
    meta: dict = Field(default_factory=dict)


class Tool(ABC):
    @abstractmethod
    async def run(self, params: dict[str, Any]) -> ToolResponse:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__tool_schema__["function"]["name"]

    @property
    def schema(self) -> dict[str, Any]:
        return self.__tool_schema__


class ToolRegistry:
    _registry: dict[str, Tool] = {}
    _init_params: dict[str, dict[str, Any]] = {}

    @classmethod
    def register(cls, tool_class: Tool) -> Tool:
        tool_class_name = tool_class.__name__
        cls._registry[tool_class_name] = tool_class

        init_sig = inspect.signature(tool_class.__init__)
        params_info = {}
        for param_name, param in init_sig.parameters.items():
            if param_name == "self":
                continue
            params_info[param_name] = {
                "default": param.default,
                "annotation": param.annotation,
                "required": param.default == inspect.Parameter.empty,
            }
        cls._init_params[tool_class_name] = params_info

        return tool_class

    @classmethod
    def create(cls, tool_class_name: str, **kwargs: Any) -> Tool:
        if tool_class_name not in cls._registry:
            raise ValueError(f"Tool class {tool_class_name} not found")

        tool_class = cls._registry[tool_class_name]
        try:
            init_kwargs = {
                param_name: kwargs.get(param_name, param_info["default"])
                for param_name, param_info in cls._init_params[tool_class_name].items()
                if param_name in kwargs or not param_info["required"]
            }
            return tool_class(**init_kwargs)
        except Exception as e:
            logger.error(f"Failed to create tool class {tool_class_name}: {e}")
            raise e

    @classmethod
    def tools(cls) -> list[Tool]:
        tools = [cls.create(tool_class_name) for tool_class_name in cls._registry]
        return tools


@ToolRegistry.register
class ReadTool(Tool):
    def __init__(self):
        self.__tool_schema__ = {
            "type": "function",
            "function": {
                "name": "read_tool",
                "description": "Read a file and return its content",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "The path to the file to read"},
                    },
                },
                "required": ["path"],
            },
        }

    async def run(self, params: dict[str, Any]) -> ToolResponse:
        path = params.get("path", "").strip()

        try:
            with open(path, "r") as f:
                content = f.read()
        except Exception as e:
            content = f"Failed to read file {path}: {e}"

        return ToolResponse(content=content)


@ToolRegistry.register
class WriteTool(Tool):
    def __init__(self):
        self.__tool_schema__ = {
            "type": "function",
            "function": {
                "name": "write_tool",
                "description": "Write content to a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "The path to the file to write to"},
                        "content": {"type": "string", "description": "The content to write to the file"},
                    },
                },
                "required": ["path", "content"],
            },
        }

    async def run(self, params: dict[str, Any]) -> ToolResponse:
        path = params.get("path", "").strip()
        content = params.get("content", "").strip()

        try:
            with open(path, "w") as f:
                f.write(content)
                content = f"File {path} written successfully."
        except Exception as e:
            content = f"Failed to write file {path}: {e}"

        return ToolResponse(content=content)


@ToolRegistry.register
class BashTool(Tool):
    def __init__(self):
        self.__tool_schema__ = {
            "type": "function",
            "function": {
                "name": "bash_tool",
                "description": "Execute a command and return the output",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "The command to execute"},
                    },
                },
                "required": ["command"],
            },
        }

    async def run(self, params: dict[str, Any]) -> ToolResponse:
        command = params.get("command", "").strip()

        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            content = "".join([result.stdout, result.stderr])
        except Exception as e:
            content = f"Failed to execute command {command}: {e}"

        return ToolResponse(content=content)
