import os
from typing import Optional

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
        with open(os.path.join(BASE_PATH, 'models.yaml')) as inp:
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


class LLMChat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'model': (MODELS.CHOICES,),
                'temperature': ('FLOAT', {
                    'min': 0.0,
                    'default': 0.8,
                }),
                'seed': ('INT', { # Thankfully, frontend has special handling for a widget of this name...
                    'max': 0xffffffff_ffffffff_ffffffff_ffffffff, # TODO what's the actual acceptable range??
                }),
                'user_prompt': ('STRING', {
                    'multiline': True,
                }),
            },
            'optional': {
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

    def execute(self, model, temperature, seed, user_prompt, system_prompt=None):
        llm = openai.OpenAI(
            base_url=MODELS.get_base_url(model),
            api_key=(MODELS.get_api_key(model) or 'none'),
        )

        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': user_prompt})

        output = llm.chat.completions.create(
            messages=messages,
            model=MODELS.get_model(model),
            seed=seed,
            temperature=temperature,
        )

        # TODO Since we're not streaming, we'll need some sort of timeout. How to specify that?

        return (output.choices[0].message.content,)


NODE_CLASS_MAPPINGS = {
    'LLMChat': LLMChat,
}
