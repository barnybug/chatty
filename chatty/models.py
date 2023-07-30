from dataclasses import dataclass
from typing import List
from sqlalchemy import ForeignKey
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship


class Base(DeclarativeBase):
    pass


class Session(Base):
    __tablename__ = "session"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str | None]
    model: Mapped[str]
    messages: Mapped[List["Message"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by="Message.id",
    )

    @property
    def label(self) -> str:
        if self.name:
            return self.name
        if self.messages:
            return self.messages[0].content
        return "New session"


class Message(Base):
    __tablename__ = "message"
    id: Mapped[int] = mapped_column(primary_key=True)
    role: Mapped[str]
    content: Mapped[str]
    tokens: Mapped[int | None]
    session_id: Mapped[int] = mapped_column(ForeignKey("session.id"))
    session: Mapped["Session"] = relationship(back_populates="messages")

    def append(self, b: "Message"):
        self.content = (self.content or "") + b.content


@dataclass
class Update:
    role: str | None = None
    content: str | None = None
    finish_reason: str | None = None
