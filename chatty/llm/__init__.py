from .base import Base
from ..config import ModelConfig


def load_llm(config: ModelConfig) -> Base:
    mod = __import__("chatty.llm.%s" % config.module, fromlist=["chatty.llm"])
    return mod.create(config)
