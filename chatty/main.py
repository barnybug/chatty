from functools import cache
from rich.text import Text
from sqlalchemy import orm
from textual import on
from textual.reactive import reactive
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.message import Message as TextualMessage
from textual.screen import ModalScreen
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

from . import config
from . import db
from . import models
from . import llm
from .llm.base import Base


engine = db.engine()


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


@cache
def create_model(c: config.ModelConfig) -> Base:
    return llm.load_llm(c)


class SessionWidget(Static):
    session = reactive(models.Session)
    editing: int | None = None

    def __init__(self, config: config.Config, session: models.Session):
        super().__init__(id="right")
        self.config = config
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
            config = self.config.model[self.session.model]
            self.app.title = config.title

    @property
    def model_instance(self) -> Base:
        return create_model(self.config.model[self.session.model])

    @on(Input.Submitted)
    async def send(self, event: Input.Submitted):
        if event.validation_result is None or not event.validation_result.is_valid:
            return

        model = self.model_instance
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


class ModelPickerModal(ModalScreen[str]):
    AUTO_FOCUS = "ListView"

    def __init__(self, config: config.Config) -> None:
        self.config = config
        super().__init__()

    def compose(self) -> ComposeResult:
        items = [ListItem(Label(model.title)) for model in self.config.model.values()]
        self.models = list(self.config.model.keys())
        yield ListView(*items)

    @on(ListView.Selected)
    def on_selected(self, event: ListView.Selected) -> None:
        model = self.models[event.list_view.index]
        self.dismiss(model)


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

    def __init__(self):
        self.config = config.load_or_default()
        super().__init__()

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

        items = [
            ListItem(Label(session.label, markup=False)) for session in self.sessions
        ]
        yield ListView(
            *items,
            id="sessions",
        )
        yield SessionWidget(self.config, session)
        yield Footer()

    @on(ListView.Highlighted, "#sessions")
    def session_changed(self):
        list_view = self.query_one("#sessions", ListView)
        if (i := list_view.index) is not None:
            widget = self.query_one(SessionWidget)
            widget.session = self.sessions[i]

    def action_new_session(self) -> None:
        def model_picked(model: str) -> None:
            list_view = self.query_one("#sessions", ListView)
            session = models.Session(model=model)
            self.sessions.append(session)
            list_view.append(ListItem(Label(session.label)))
            list_view.index = len(list_view.children) - 1
            widget = self.query_one(SessionWidget)
            widget.session = session
            input = self.query_one("#input", Input)
            input.focus()

        self.push_screen(ModelPickerModal(self.config), model_picked)

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
