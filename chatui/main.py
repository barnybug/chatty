from dataclasses import dataclass
from operator import is_
from typing import AsyncGenerator

from rich.text import Text
from sqlalchemy import inspect, orm
from textual import on
from textual.reactive import reactive
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.message import Message as TextualMessage
from textual.validation import Length
from textual.widgets import (
    Header,
    Footer,
    Input,
    Label,
    ListItem,
    ListView,
    Static,
)
import openai
import tiktoken
from pydantic import BaseModel

from . import db
from . import models

openai.log = "debug"
engine = db.engine()


@dataclass
class Update:
    role: str | None = None
    content: str | None = None
    finish_reason: str | None = None


class OpenAIModel(BaseModel):
    model: str = "gpt-3.5-turbo"

    async def query(
        self, messages: list[models.Message]
    ) -> AsyncGenerator[Update, None]:
        msgs = [
            {"role": message.role, "content": message.content}
            for message in messages
            if message.role in ("user", "assistant", "system")
        ]
        try:
            response = await openai.ChatCompletion.acreate(
                # model="gpt-4",
                model="gpt-3.5-turbo",
                messages=msgs,
                stream=True,
            )
            async for chunk in response:
                choice = chunk["choices"][0]
                delta = choice["delta"]
                if "role" in delta:
                    yield Update(role=delta["role"])
                if "content" in delta:
                    yield Update(content=delta["content"])
                if choice.get("finish_reason"):
                    yield Update(finish_reason=choice["finish_reason"])
        except openai.InvalidRequestError as ex:
            yield Update(role="error", content=str(ex))

    def token_count(self, s: str) -> int:
        enc = tiktoken.encoding_for_model(self.model)
        return len(enc.encode(s))


class ChatLog(ListView):
    class Action(TextualMessage):
        def __init__(self, index: int, delete=False, edit=False) -> None:
            super().__init__()
            self.index = index
            self.delete = delete
            self.edit = edit

    BINDINGS = [
        ("r", "rerun", "Rerun"),
        ("enter", "edit", "Edit"),
        ("backspace", "delete", "Delete"),
    ]

    messages = reactive(list[models.Message], always_update=True)

    def watch_messages(self, messages: list[models.Message]):
        self.clear()
        for message in messages:
            parts = []
            if message.role:
                colour = (
                    "blue"
                    if message.role == "user"
                    else "red"
                    if message.role == "error"
                    else "green"
                )
                parts.append((f"{message.role.title()}: ", colour))
            if message.content:
                parts.append(message.content)
            if not message.content and not message.role:
                parts.append("…")

            content = Text.assemble(*parts)
            self.append(ListItem(Static(content)))
        self.index = None

    def action_delete(self):
        if self.index is not None:
            message = self.messages[self.index]
            if message.role in ("user", "system"):
                self.post_message(ChatLog.Action(self.index, delete=True))

    def action_edit(self):
        if self.index is not None:
            message = self.messages[self.index]
            if message.role in ("user", "system"):
                self.post_message(ChatLog.Action(self.index, edit=True))


class SessionWidget(Static):
    session: reactive[models.Session] = reactive(models.Session)
    editing: int | None = None

    def __init__(self, session: models.Session):
        super().__init__(id="right")
        self.session = session

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Header()
            yield ChatLog()
            yield Input(
                placeholder="Ask me a question...",
                id="input",
                validators=[Length(1)],
            )

    def on_mount(self):
        self.render_log()

    def render_log(self):
        log = self.query_one(ChatLog)
        log.messages = self.session.messages

    def watch_session(self, session: models.Session):
        if self.is_attached:
            self.render_log()

    @on(Input.Submitted)
    async def send(self, event: Input.Submitted):
        if event.validation_result is None or not event.validation_result.is_valid:
            return

        model = OpenAIModel()
        i = self.query_one(Input)
        text = i.value

        if self.editing is None:
            message = models.Message(
                role="user",
                content=text,
                session=self.session,
                tokens=model.token_count(text),
            )
        else:
            message = self.session.messages[self.editing]
            message.content = text

        with orm.Session(engine) as sess:
            sess.add(self.session)
            sess.commit()
            sess.refresh(self.session)

        self.render_log()

        i.disabled = True

        messages = (
            self.session.messages[: self.editing + 1]
            if self.editing is not None
            else self.session.messages
        )
        if self.editing is None:
            message = models.Message(role="", content="")
            self.session.messages.append(message)
        else:
            message = self.session.messages[self.editing + 1]
            message.role = ""
            message.content = ""
        self.render_log()  # show "…" initially

        async for update in model.query(messages):
            if update.role:
                message.role = update.role
            if update.content:
                message.content += update.content
            self.render_log()

        message.tokens = model.token_count(message.content)
        with orm.Session(engine) as sess:
            sess.add(self.session)
            sess.commit()
            sess.refresh(self.session)

        i.value = ""
        i.disabled = False
        i.focus()
        self.editing = None

    @on(ChatLog.Action)
    def action_message(self, event: ChatLog.Action):
        message = self.session.messages[event.index]
        if event.edit:
            input = self.query_one(Input)
            input.value = message.content
            input.focus()
            self.editing = event.index
        elif event.delete:
            if event.index + 1 < len(self.session.messages):
                self.session.messages.pop(event.index + 1)
            self.session.messages.pop(event.index)
            with orm.Session(engine) as sess:
                sess.add(self.session)
                sess.commit()
                sess.refresh(self.session)
            self.render_log()


class ChatApp(App):
    AUTO_FOCUS = "Input"
    BINDINGS = [
        ("d", "toggle_dark", "Toggle dark mode"),
        ("ctrl+n", "new_session", "New session"),
        ("ctrl+x", "delete_session", "Delete session"),
        ("ctrl+q", "quit", "Quit"),
    ]
    CSS_PATH = "chatui.css"

    sessions: list[models.Session] = []

    def compose(self) -> ComposeResult:
        with orm.Session(engine) as sess:
            self.sessions = sess.query(models.Session).all()
            if self.sessions:
                session = self.sessions[0]
            else:
                session = models.Session()
                self.sessions.append(session)
                sess.add(session)  # ?
                sess.commit()
                sess.refresh(session)

        items = [ListItem(Label(session.label)) for session in self.sessions]
        yield ListView(
            *items,
            id="sessions",
        )
        yield SessionWidget(session)
        yield Footer()

    @on(ListView.Highlighted)
    def session_changed(self):
        list_view = self.query_one("#sessions", ListView)
        if (i := list_view.index) is not None:
            widget = self.query_one(SessionWidget)
            widget.session = self.sessions[i]

    def action_new_session(self) -> None:
        list_view = self.query_one("#sessions", ListView)
        session = models.Session()
        self.sessions.append(session)
        list_view.append(ListItem(Label(session.label)))
        list_view.index = len(list_view.children) - 1
        widget = self.query_one(SessionWidget)
        widget.session = session

    def action_delete_session(self) -> None:
        list_view = self.query_one("#sessions", ListView)
        if (index := list_view.index) is not None:
            session = self.sessions.pop(index)
            if session.id is not None:
                with orm.Session(engine) as sess:
                    sess.delete(session)
                    sess.commit()
            item = list_view._nodes[index]
            item.remove()
            if index >= len(list_view):
                list_view.index = index - 1
            else:
                list_view.index = index

    def action_toggle_dark(self) -> None:
        self.dark = not self.dark

    def action_quit(self) -> None:
        self.exit()


def main():
    app = ChatApp()
    app.run()
