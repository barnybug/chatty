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
        for message in messages:
            content = message.content
            if message.role == "system":
                if self.config.system_format:
                    content = self.config.system_format.replace("{message}", content)
            elif message.role == "user":
                if self.config.user_format:
                    content = self.config.user_format.replace("{message}", content)
            elif message.role == "assistant":
                if self.config.assistant_format:
                    content = self.config.assistant_format.replace("{message}", content)
            msgs.append(content)
        text = "".join(msgs)

        print("PROMPT:\n", text)

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
            yield Update(role="assistant")
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
