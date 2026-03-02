from typing import Any

from ubot.llms.base import LLM
from ubot.tools.base import ToolRegistry
from ubot.tools.template import OpenAIToolTemplate


class UBot:
    def __init__(self, model: LLM):
        self.model = model
        self.tools = ToolRegistry.tools()
        self.model.bind_tool_template(OpenAIToolTemplate(self.tools))

    async def run(self, messages: list[dict[str, Any]], max_iterations: int = 20):
        iteration = 0
        finished = False

        while not finished and iteration < max_iterations:
            tool_schemas = self.model.tool_template.generate_prompt()
            response = await self.model.call(messages, tools=tool_schemas)

            messages.append(response.message.model_dump())
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    result = await tool_call.tool.run(tool_call.params)
                    tool_message = {"role": "tool", "content": result, "tool_call_id": tool_call.tool_call_id}
                    messages.append(tool_message)
            else:
                finished = True

            iteration += 1

        return messages
