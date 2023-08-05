from abc import ABC
from typing import AsyncGenerator
from pydantic import BaseModel

from chatty import models


class Base(BaseModel, ABC):
    def query(
        self, messages: list[models.Message]
    ) -> AsyncGenerator[models.Update, None]:
        raise NotImplementedError()

    def token_count(self, s: str) -> int:
        raise NotImplementedError()
