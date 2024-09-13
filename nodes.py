import os

import openai


class LLMConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'base_url': ('STRING', {
                    'multiline': False,
                    'default': 'http://localhost:8080/v1',
                }),
            },
            'optional': {
                'api_key_env_var': ('STRING', {
                    'multiline': False,
                    'default': '',
                }),
            }
        }

    TITLE = 'OpenAI-like LLM Config'

    RETURN_TYPES = ('LLMCONFIG',)

    FUNCTION = 'execute'

    CATEGORY = 'YALLM'

    def execute(self, base_url, api_key_env_var):
        api_key = "none" # The openai module needs *something*
        if api_key_env_var:
            api_key = os.environ[api_key_env_var] # TODO is there a way to fail gracefully or at least display error message?

        llm = openai.OpenAI(base_url=base_url, api_key=api_key)

        models = [m.id for m in llm.models.list()]

        llmconfig = {
            'llm': llm,
            'models': models,
        }

        return (llmconfig,)


class LLMChat:
    _MODEL_CACHE: dict[str,list[str]] = {}

    @classmethod
    def _get_models(cls) -> list[str]:
        # TODO Kludge for now, just merge everyone's list of models
        models = set([])
        for v in cls._MODEL_CACHE.values():
            models.update(v)

        models.add('default') # Needs to have at least 1 value so we can execute this node...

        result = list(models)
        result.sort()

        return result

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'llmconfig': ('LLMCONFIG', { 'forceInput': True }),
                'user_prompt': ('STRING', {
                    'multiline': True,
                }),
                'model': (cls._get_models(), {}),  # Hmm, we don't have access to our UNIQUE_ID...
                'temperature': ('FLOAT', {
                    'min': 0.0,
                    'default': 0.8,
                }),
                'seed': ('INT', { # Thankfully, frontend has special handling for a widget of this name...
                    'max': 0xffffffff_ffffffff_ffffffff_ffffffff, # TODO what's the actual acceptable range??
                })
            },
            'optional': {
                'system_prompt': ('STRING', {
                    'multiline': True,
                }),
            },
            'hidden': {
                'unique_id': 'UNIQUE_ID',
            }
        }

    TITLE = 'LLM Chat'

    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('completion',)

    FUNCTION = 'execute'

    CATEGORY = 'YALLM'

    def execute(self, llmconfig, user_prompt, model, temperature, seed, unique_id, system_prompt=None):
        llm = llmconfig['llm']
        # TODO need to figure out how to deal with this properly
        # Maybe create an API endpoint and then modify the frontend (ugh) to deal with the combo select?
        LLMChat._MODEL_CACHE[unique_id] = llmconfig['models']

        if model == 'default':
            model = llmconfig['models'][0]

        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': user_prompt})

        output = llm.chat.completions.create(
            messages=messages,
            model=model,
            seed=seed,
            temperature=temperature,
        )

        # TODO Since we're not streaming, we'll need some sort of timeout. How to specify that?

        return (output.choices[0].message.content,)


NODE_CLASS_MAPPINGS = {
    'LLMConfig': LLMConfig,
    'LLMChat': LLMChat,
}
