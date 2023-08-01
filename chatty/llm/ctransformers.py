import asyncio
from typing import Any, AsyncGenerator
from threading import Thread

from ctransformers import AutoModelForCausalLM

from .base import Base
from ..config import CTransformersModelConfig
from ..models import Message, Update


class CTransformers(Base):
    config: CTransformersModelConfig

    def model_post_init(self, __context: Any) -> None:
        params = self.config.params()
        self._llm = AutoModelForCausalLM.from_pretrained(**params)
        return super().model_post_init(__context)

    async def query(self, messages: list[Message]) -> AsyncGenerator[Update, None]:
        msgs = []
        for i, message in enumerate(messages):
            content = message.content
            if message.role in "user":
                if i == 0 and self.config.system_message:
                    content = self.config.system_message + " " + content
                if self.config.prefix:
                    content = self.config.prefix + " " + content
                if self.config.suffix:
                    content = content + " " + self.config.suffix
                msgs.append(content)
            elif message.role in "assistant":
                msgs.append(message.content)
        text = "\n".join(msgs)

        queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def generate():
            tokens = self._llm.tokenize(text)
            for token in self._llm.generate(tokens):
                content = self._llm.detokenize(token)
                loop.call_soon_threadsafe(queue.put_nowait, content)
            loop.call_soon_threadsafe(queue.put_nowait, None)

        thread = Thread(target=generate)
        thread.start()
        yield Update(role="Assistant")
        while True:
            content = await queue.get()
            if content is None:
                break
            yield Update(content=content)

        thread.join()

    def token_count(self, s: str) -> int:
        return len(self._llm.tokenize(s))


def create(config: CTransformersModelConfig) -> Base:
    return CTransformers(config=config)
