from typing import AsyncGenerator

import openai
import tiktoken

from .base import Base
from ..config import OpenAIModelConfig
from ..models import Message, Update


openai.log = "debug"


class OpenAIModel(Base):
    config: OpenAIModelConfig

    async def query(self, messages: list[Message]) -> AsyncGenerator[Update, None]:
        msgs = [
            {"role": message.role, "content": message.content}
            for message in messages
            if message.role in ("user", "assistant", "system")
        ]
        if self.config.system_message:
            msgs.insert(0, {"role": "system", "content": self.config.system_message})
        try:
            response = await openai.ChatCompletion.acreate(
                messages=msgs,
                stream=True,
                **self.config.params(),
            )
            async for chunk in response:
                choice = chunk.choices[0]
                delta = choice.delta
                if "role" in delta:
                    yield Update(role=delta["role"])
                if "content" in delta:
                    yield Update(content=delta["content"])
                if choice.get("finish_reason"):
                    yield Update(finish_reason=choice["finish_reason"])
        except openai.InvalidRequestError as ex:
            yield Update(role="error", content=str(ex))

    def token_count(self, s: str) -> int:
        enc = tiktoken.encoding_for_model(self.config.model)
        return len(enc.encode(s))


def create(config: OpenAIModelConfig) -> Base:
    return OpenAIModel(config=config)
