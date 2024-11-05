# Copyright (c) 2024 Allan Saddi <allan@saddi.com>
import base64
from io import BytesIO
import os
from typing import Any, Optional

from aiohttp import web
from aiohttp.web_request import Request
import openai
from PIL.Image import Image
from pydantic import BaseModel
import torchvision.transforms.functional as F
import yaml

from server import PromptServer


BASE_PATH = os.path.dirname(os.path.realpath(__file__))


class ProviderDefinition(BaseModel):
    name: str
    base_url: str
    api_key: Optional[str]

    def resolve_base_url(self) -> str:
        return ProviderDefinition._get_env_or_value(self.base_url)

    def resolve_api_key(self) -> str | None:
        api_key = self.api_key
        if api_key:
            api_key = ProviderDefinition._get_env_or_value(api_key)
        if not api_key or api_key == "none":
            return None
        return api_key

    @staticmethod
    def _get_env_or_value(value: str) -> str:
        """
        Retrieves the value from either an environment variable or a provided string.

        Args:
            value (str): The value to retrieve. It can be a string literal or a string
                starting with 'os.environ/' followed by an environment variable name.

        Returns:
            str: The retrieved value.
        """
        PREFIX = "os.environ/"
        if value.startswith(PREFIX):
            env_var = value[len(PREFIX) :]
            value = os.environ[env_var]
        return value


class ModelDefinition(ProviderDefinition):
    model: str


class DefinitionsConfig:
    _mtime = None

    def load(self, def_type) -> None:
        models_file = self._get_models_file()

        with open(models_file) as inp:
            d = yaml.load(inp, yaml.Loader)
        self._mtime = (models_file, os.path.getmtime(models_file))

        self.LIST = []
        for value in d[self._root]:
            self.LIST.append(def_type.model_validate(value))
        if not self.LIST:
            raise RuntimeError(f"{models_file}: Need at least one definition")
        self.BY_NAME = {d.name: d for d in self.LIST}
        self.CHOICES = [d.name for d in self.LIST]

    def _get_models_file(self):
        models_file = os.path.join(BASE_PATH, self._config)
        if not os.path.exists(models_file):
            # Just read from the example for now
            # print(f'{models_file} does not exist; using default')
            models_file = os.path.join(BASE_PATH, self._config_default)
        return models_file

    def refresh(self):
        models_file = self._get_models_file()
        if self._mtime != (models_file, os.path.getmtime(models_file)):
            self.load()


class Providers(DefinitionsConfig):
    _root = "providers"
    _config = "providers.yaml"
    _config_default = "providers.yaml.example"

    def load(self):
        return super().load(ProviderDefinition)


class Models(DefinitionsConfig):
    _root = "models"
    _config = "models.yaml"
    _config_default = "models.yaml.example"

    def load(self):
        return super().load(ModelDefinition)


PROVIDERS = Providers()
PROVIDERS.load()

MODELS = Models()
MODELS.load()


class LLMTextLatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                    },
                ),
                "replace": (
                    "BOOLEAN",
                    {
                        "default": True,
                    },
                ),
            },
            "optional": {
                "text_input": (
                    "STRING",
                    {
                        "forceInput": True,
                        "multiline": True,
                    },
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, text):
        # Always valid, even if empty
        return True

    TITLE = "Text Latch"

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)

    FUNCTION = "execute"

    OUTPUT_NODE = True

    CATEGORY = "YALLM"

    def execute(self, text, replace, text_input=None):
        if text_input is not None and replace:
            text = text_input

        if text is None:
            text = ""

        return {"ui": {"text": text}, "result": (text,)}


class SamplerDefinition(BaseModel):
    name: str
    is_int: bool
    min: Optional[float | int] = None
    default: float | int
    max: Optional[float | int] = None

    def to_input_type(self):
        constraints = {}
        if self.min is not None:
            constraints["min"] = self.min
        if self.max is not None:
            constraints["max"] = self.max
        constraints["default"] = self.default
        return {self.name: ("INT" if self.is_int else "FLOAT", constraints)}


class LLMTemperature:
    _DEF = SamplerDefinition(
        name="temperature",
        is_int=False,
        min=0.0,
        default=0.8,
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": cls._DEF.to_input_type(),
            "optional": {
                "previous": ("LLMSAMPLER",),
            },
        }

    TITLE = "LLM Temperature"

    RETURN_TYPES = ("LLMSAMPLER",)
    RETURN_NAMES = ("llm_sampler",)

    FUNCTION = "execute"

    CATEGORY = "YALLM/samplers"

    def execute(self, previous=None, **args):
        value = args[self._DEF.name]

        if previous is None:
            previous = []

        previous.append((self._DEF.name, value))

        return (previous,)


class LLMTopP(LLMTemperature):
    _DEF = SamplerDefinition(
        name="top_p",
        is_int=False,
        min=0.0,
        default=0.95,
        max=1.0,
    )

    TITLE = "LLM Top-P"


class LLMTopK(LLMTemperature):
    _DEF = SamplerDefinition(
        name="top_k",
        is_int=True,
        min=1,
        default=40,
    )

    TITLE = "LLM Top-K"


class LLMMinP(LLMTemperature):
    _DEF = SamplerDefinition(
        name="min_p",
        is_int=False,
        min=0.0,
        default=0.05,
        max=1.0,
    )

    TITLE = "LLM Min-P"


# TODO typical_p? tfs_z? Any others? mirostat?


# TODO Will we need a more complex definition? (e.g. multi-modal)
ChatMessage = dict[str, str]
ChatHistory = list[ChatMessage]

SamplerSetting = tuple[str, Any]


class LLMModel:
    def __init__(self, prov_def: ProviderDefinition, model: str | None = None):
        self._llm = openai.OpenAI(
            base_url=prov_def.resolve_base_url(),
            api_key=(
                prov_def.resolve_api_key() or "none"
            ),  # openai package requires something for API key
        )
        if model is None:
            self._model = getattr(prov_def, "model", None)
        else:
            self._model = model

    def get_models(self):
        models = list(self._llm.models.list())
        return [m.id for m in models]

    def chat_completion(
        self,
        messages: ChatHistory,
        image: Image | None = None,
        samplers: list[SamplerSetting] | None = None,
        seed: int | None = None,
    ) -> str:
        assert self._model is not None

        if image is not None:
            # Work backwards through history
            for msg in reversed(messages):
                # And grab the very last user message we see
                if msg.get("role") == "user":
                    parts = msg.get("content", [])
                    if isinstance(parts, str):
                        parts = [{"type": "text", "text": parts}]

                    # Convert image to data URI format
                    png_io = BytesIO()
                    image.save(png_io, "PNG")
                    png_b64 = base64.b64encode(png_io.getbuffer()).decode("ascii")

                    # And insert image just before the user prompt
                    parts.insert(
                        0,
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{png_b64}"},
                        },
                    )

                    msg["content"] = parts
                    break

        # print(messages)

        if samplers is None:
            samplers = []

        sampler_order = []
        extra_args = {}
        extra_body = {}
        for k, v in samplers:
            sampler_order.append(k)
            if k in (
                "temperature",
                "top_p",
            ):  # The only ones supported directly by openai package
                extra_args[k] = v
            else:
                # The rest have to go into "extra_body"
                extra_body[k] = v
        extra_body["samplers"] = (
            sampler_order  # Hopefully won't cause an issue for non-llama.cpp endpoints
        )

        if seed is not None:
            extra_args["seed"] = seed

        # print(f'extra_args = {repr(extra_args)}')
        # print(f'extra_body = {repr(extra_body)}')

        output = self._llm.chat.completions.create(
            messages=messages, model=self._model, extra_body=extra_body, **extra_args
        )

        # TODO Since we're not streaming, we'll need some sort of timeout. How to specify that?

        return output.choices[0].message.content


class LLMProvider:
    @classmethod
    def INPUT_TYPES(cls):
        PROVIDERS.refresh()

        return {
            "required": {
                "provider": (PROVIDERS.CHOICES,),
                "model": (["fetch.models.first"],),
            },
        }

    @classmethod
    def IS_CHANGED(cls, provider, model):
        PROVIDERS.refresh()
        return PROVIDERS._mtime

    @classmethod
    def VALIDATE_INPUTS(cls, model):
        if model in ("fetch.models.first", "failed.to.fetch"):
            return "fetch models and make a selection first"

        # Ass-u-me the frontend validated the choices
        return True

    TITLE = "LLM Provider (API)"

    RETURN_TYPES = ("LLMMODEL",)
    RETURN_NAMES = ("llm_model",)

    FUNCTION = "execute"

    CATEGORY = "YALLM"

    def execute(self, provider, model):
        llm = LLMModel(PROVIDERS.BY_NAME[provider], model=model)

        return (llm,)


class LLMModelNode:
    @classmethod
    def INPUT_TYPES(cls):
        MODELS.refresh()
        return {
            "required": {
                "model": (MODELS.CHOICES,),
            },
        }

    @classmethod
    def IS_CHANGED(cls, model):
        MODELS.refresh()
        return MODELS._mtime

    TITLE = "LLM Model (API)"

    RETURN_TYPES = ("LLMMODEL",)
    RETURN_NAMES = ("llm_model",)

    FUNCTION = "execute"

    CATEGORY = "YALLM"

    def execute(self, model):
        llm = LLMModel(MODELS.BY_NAME[model])

        return (llm,)


class LLMChat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLMMODEL",),
                "seed": (
                    "INT",
                    {  # Thankfully, frontend has special handling for a widget of this name...
                        # min/max/default from the KSampler nodes
                        # llama.cpp is only 32-bit (with truncation, no doubt), OpenAI API spec doesn't specify a size
                        "min": 0,
                        "default": 0,
                        "max": 0xFFFFFFFF_FFFFFFFF,
                    },
                ),
                "user_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                    },
                ),
            },
            "optional": {
                "llm_sampler": ("LLMSAMPLER",),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                    },
                ),
                "image": ("IMAGE",),
            },
        }

    TITLE = "LLM Chat"

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("completion",)
    OUTPUT_IS_LIST = (True,)

    FUNCTION = "execute"

    CATEGORY = "YALLM"

    def execute(
        self,
        llm_model: LLMModel,
        seed,
        user_prompt,
        llm_sampler=None,
        system_prompt=None,
        image=None,
    ):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        result = []
        if image is None:
            text = llm_model.chat_completion(messages, samplers=llm_sampler, seed=seed)
            result.append(text)
        else:
            image = image.permute(0, 3, 1, 2)  # Convert to [B,C,H,W]
            for img in image:
                pil_image = F.to_pil_image(img)
                # Fortunately or unfortunately, I'm forced to have this conform
                # to the conventions of my LlamaVision node.
                # I think it would be cleaner to do the conversion here, but
                # I have no idea how that would work for a local
                # Transformers model.
                text = llm_model.chat_completion(
                    messages, image=pil_image, samplers=llm_sampler, seed=seed
                )
                result.append(text)

        return (result,)


@PromptServer.instance.routes.get("/llm_models")
async def llm_models(request: Request):
    name = request.rel_url.query.get("name", None)
    if name:
        PROVIDERS.refresh()
        if (prov_def := PROVIDERS.BY_NAME.get(name, None)) is not None:
            try:
                llm = LLMModel(prov_def)
                models = llm.get_models()
                return web.json_response(models)
            except Exception as e:
                print(f"Problem fetching models: {e}")
                # Just eat it and fall through.

    # Fail, but don't error out
    return web.json_response(["failed.to.fetch"])


NODE_CLASS_MAPPINGS = {
    "LLMTextLatch": LLMTextLatch,
    "LLMProvider": LLMProvider,
    "LLMModel": LLMModelNode,
    "LLMChat": LLMChat,
    "LLMTemperature": LLMTemperature,
    "LLMTopP": LLMTopP,
    "LLMTopK": LLMTopK,
    "LLMMinP": LLMMinP,
}
