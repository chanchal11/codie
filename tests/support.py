"""Scripted chat models for tests that should not call Ollama or Bedrock."""

from langchain_core.messages import AIMessage


class ScriptedModel:
    def __init__(self, responses: list[AIMessage]) -> None:
        self.responses = list(responses)
        self.prompts: list[list] = []

    def bind_tools(self, tools, **_kwargs):
        self.tools = tools
        return self

    def invoke(self, messages):
        self.prompts.append(list(messages))
        if not self.responses:
            return AIMessage(content="(no more scripted responses)")
        return self.responses.pop(0)

    def stream(self, messages):
        yield self.invoke(messages)
