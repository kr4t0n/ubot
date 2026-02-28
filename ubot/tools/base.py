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


@ToolRegistry.register
class ReadTool(Tool):
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
    async def run(self, params: dict[str, Any]) -> ToolResponse:
        command = params.get("command", "").strip()

        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            content = "".join([result.stdout, result.stderr])
        except Exception as e:
            content = f"Failed to execute command {command}: {e}"

        return ToolResponse(content=content)
