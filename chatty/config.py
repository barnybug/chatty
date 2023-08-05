from pathlib import Path
from typing import Any, Literal
from pydantic import BaseModel, Field
import tomllib


DEFAULT_CONFIG = """
"""


class ModelConfig(BaseModel):
    title: str
    module: Literal["ctransformers", "openai"]
    system_message: str | None = None
    completion: bool = False

    def __hash__(self):
        return hash(self.model_dump_json())

    def params(self, exclude: set | None = None) -> dict[str, Any]:
        return self.model_dump(
            exclude=(exclude or set()) | set(ModelConfig.model_fields),
            exclude_none=True,
            by_alias=True,
        )


class CTransformersModelConfig(ModelConfig):
    module: Literal["ctransformers"]
    model: str = Field(..., serialization_alias="model_path_or_repo_id")
    model_file: str | None = None
    model_type: str | None = None
    lib: Literal["avx2", "avx", "basic"] | None = None
    # see https://github.com/marella/ctransformers#config
    top_k: int | None = None
    top_p: float | None = None
    temperature: float | None = None
    repetition_penalty: float | None = None
    last_n_tokens: int | None = None
    seed: int | None = None
    max_new_tokens: int | None = None
    stop: list[str] | None = None
    batch_size: int | None = None
    threads: int | None = None
    context_length: int | None = None
    gpu_layers: int | None = None

    # prompts
    prefix: str | None = None
    suffix: str | None = None

    def params(self) -> dict[str, Any]:
        return super().params({"prefix", "suffix"})


class OpenAIModelConfig(ModelConfig):
    module: Literal["openai"]
    model: str = "gpt-3.5-turbo"
    temperature: float | None = None


class Config(BaseModel):
    model: dict[str, CTransformersModelConfig | OpenAIModelConfig]


def load(path: Path) -> Config:
    with path.open("rb") as f:
        obj = tomllib.load(f)
        return Config.model_validate(obj)


def load_or_default() -> Config:
    path = Path.home() / ".chatty.toml"
    # TODO default
    return load(path)
