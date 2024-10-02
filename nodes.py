# Copyright (c) 2024 Allan Saddi <allan@saddi.com>
import os
from typing import Any, Optional

import openai
from pydantic import BaseModel
import yaml


BASE_PATH = os.path.dirname(os.path.realpath(__file__))


class ModelDefinition(BaseModel):
    name: str
    base_url: str
    api_key: Optional[str]
    model: str


class Models:
    LIST: list[ModelDefinition] = []
    BY_NAME: dict[str,ModelDefinition] = {}
    CHOICES: list[str] = []

    def load(self) -> None:
        models_file = os.path.join(BASE_PATH, 'models.yaml')
        if not os.path.exists(models_file):
            # Just read from the example for now
            print(f'WARNING: {models_file} does not exist; using default')
            models_file = os.path.join(BASE_PATH, 'models.yaml.example')

        with open(models_file) as inp:
            d = yaml.load(inp, yaml.Loader)

        for value in d['models']:
            self.LIST.append(ModelDefinition.model_validate(value))
        if not self.LIST:
            raise RuntimeError('Need at least one model defined')
        self.BY_NAME = { d.name: d for d in self.LIST }
        self.CHOICES = [ d.name for d in self.LIST ]

    def get_base_url(self, name: str) -> str:
        base_url = self.BY_NAME[name].base_url
        return Models._get_env_or_value(base_url)

    def get_api_key(self, name: str) -> str | None:
        api_key = self.BY_NAME[name].api_key
        if api_key:
            api_key = Models._get_env_or_value(api_key)
        if not api_key or api_key == 'none':
            return None
        return api_key

    def get_model(self, name: str) -> str:
        return self.BY_NAME[name].model

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
        PREFIX = 'os.environ/'
        if value.startswith(PREFIX):
            env_var = value[len(PREFIX):]
            value = os.environ[env_var]
        return value


MODELS = Models()
MODELS.load()


class SamplerDefinition(BaseModel):
    name: str
    is_int: bool
    min: Optional[float|int] = None
    default: float|int
    max: Optional[float|int] = None

    def to_input_type(self):
        constraints = {}
        if self.min is not None:
            constraints['min'] = self.min
        if self.max is not None:
            constraints['max'] = self.max
        constraints['default'] = self.default
        return {
            self.name: ('INT' if self.is_int else 'FLOAT', constraints)
        }


class LLMTemperature:
    _DEF = SamplerDefinition(
        name='temperature',
        is_int=False,
        min=0.0,
        default=0.8,
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': cls._DEF.to_input_type(),
            'optional': {
                'previous': ('LLMSAMPLER',),
            },
        }

    TITLE = 'LLM Temperature'

    RETURN_TYPES = ('LLMSAMPLER',)
    RETURN_NAMES = ('llm_sampler',)

    FUNCTION = 'execute'

    CATEGORY = 'YALLM/samplers'

    def execute(self, previous=None, **args):
        value = args[self._DEF.name]

        if previous is None:
            previous = []

        previous.append((self._DEF.name, value))

        return (previous,)


class LLMTopP(LLMTemperature):
    _DEF = SamplerDefinition(
        name='top_p',
        is_int=False,
        min=0.0,
        default=0.95,
        max=1.0,
    )

    TITLE = 'LLM Top-P'


class LLMTopK(LLMTemperature):
    _DEF = SamplerDefinition(
        name='top_k',
        is_int=True,
        min=1,
        default=40,
    )

    TITLE = 'LLM Top-K'


class LLMMinP(LLMTemperature):
    _DEF = SamplerDefinition(
        name='min_p',
        is_int=False,
        min=0.0,
        default=0.05,
        max=1.0,
    )

    TITLE = 'LLM Min-P'


# TODO typical_p? tfs_z? Any others? mirostat?


# TODO Will we need a more complex definition? (e.g. multi-modal)
ChatMessage = dict[str,str]
ChatHistory = list[ChatMessage]

SamplerSetting = tuple[str,Any]


class LLMModel:
    def __init__(self, model: str):
        self._llm = openai.OpenAI(
            base_url=MODELS.get_base_url(model),
            api_key=(MODELS.get_api_key(model) or 'none'), # openai package requires something for API key
        )
        self._model = MODELS.get_model(model)

    def chat_completion(self, messages: ChatHistory, samplers: list[SamplerSetting]|None=None, seed: int|None=None) -> str:
        if samplers is None:
            samplers = []

        sampler_order = []
        extra_args = {}
        extra_body = {}
        for k,v in samplers:
            sampler_order.append(k)
            if k in ('temperature', 'top_p'): # The only ones supported directly by openai package
                extra_args[k] = v
            else:
                # The rest have to go into "extra_body"
                extra_body[k] = v
        extra_body['samplers'] = sampler_order # Hopefully won't cause an issue for non-llama.cpp endpoints

        if seed is not None:
            extra_args['seed'] = seed

        # print(f'extra_args = {repr(extra_args)}')
        # print(f'extra_body = {repr(extra_body)}')

        output = self._llm.chat.completions.create(
            messages=messages,
            model=self._model,
            extra_body=extra_body,
            **extra_args
        )

        # TODO Since we're not streaming, we'll need some sort of timeout. How to specify that?

        return output.choices[0].message.content


class LLMModelNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'model': (MODELS.CHOICES,),
            },
        }

    TITLE = 'LLM Model (API)'

    RETURN_TYPES = ('LLMMODEL',)
    RETURN_NAMES = ('llm_model',)

    FUNCTION = 'execute'

    CATEGORY = 'YALLM'

    def execute(self, model):
        llm = LLMModel(model)

        return (llm,)


class LLMChat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'llm_model': ('LLMMODEL',),
                'seed': ('INT', { # Thankfully, frontend has special handling for a widget of this name...
                    # min/max/default from the KSampler nodes
                    # llama.cpp is only 32-bit (with truncation, no doubt), OpenAI API spec doesn't specify a size
                    'min': 0,
                    'default': 0,
                    'max': 0xffffffff_ffffffff,
                }),
                'user_prompt': ('STRING', {
                    'multiline': True,
                }),
            },
            'optional': {
                'llm_sampler': ('LLMSAMPLER', ),
                'system_prompt': ('STRING', {
                    'multiline': True,
                }),
            },
        }

    TITLE = 'LLM Chat'

    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('completion',)

    FUNCTION = 'execute'

    CATEGORY = 'YALLM'

    def execute(self, llm_model: LLMModel, seed, user_prompt, llm_sampler=None, system_prompt=None):
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': user_prompt})

        result = llm_model.chat_completion(messages, samplers=llm_sampler, seed=seed)

        return (result,)


NODE_CLASS_MAPPINGS = {
    'LLMModel': LLMModelNode,
    'LLMChat': LLMChat,
    'LLMTemperature': LLMTemperature,
    'LLMTopP': LLMTopP,
    'LLMTopK': LLMTopK,
    'LLMMinP': LLMMinP,
}
