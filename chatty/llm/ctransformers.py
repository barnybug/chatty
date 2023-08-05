import asyncio
import time
from typing import Any, AsyncGenerator
import threading

from ctransformers import AutoModelForCausalLM

from .base import Base
from ..config import CTransformersModelConfig
from ..models import Message, Update


class ResultGenerator:
    def cancel(self):
        pass

    def __aiter__(self):
        pass


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

        queue = asyncio.Queue[str]()
        loop = asyncio.get_event_loop()
        stop_event = threading.Event()

        def generate():
            # Run in thread
            last_update = 0
            tokens = self._llm.tokenize(text)
            items = []
            for token in self._llm.generate(tokens):
                content = self._llm.detokenize([token])
                items.append(content)
                if time.time() > last_update + 0.3:
                    loop.call_soon_threadsafe(queue.put_nowait, "".join(items))
                    items = []
                    last_update = time.time()

                if stop_event.is_set():
                    break

            if items:
                loop.call_soon_threadsafe(queue.put_nowait, "".join(items))

            loop.call_soon_threadsafe(queue.put_nowait, None)

        thread = threading.Thread(target=generate)
        thread.start()
        try:
            yield Update(role="Assistant")
            while True:
                content = await queue.get()
                if content is None:
                    break
                yield Update(content=content)
        except GeneratorExit:
            # Interrupted by caller (user)
            pass

        # Signal thread to stop
        stop_event.set()

        thread.join()

    def token_count(self, s: str) -> int:
        return len(self._llm.tokenize(s))


def create(config: CTransformersModelConfig) -> Base:
    return CTransformers(config=config)
