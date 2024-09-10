import os

import openai


class LLMPrompt:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'prompt': ('STRING', {
                    'multiline': True,
                })
            },
        }

    TITLE = 'LLM Prompt'

    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('prompt',)

    FUNCTION = 'execute'

    CATEGORY = 'YALLM'

    def execute(self, prompt):
        return (prompt,)


class OpenAILikeLLM:
    def __init__(self):
        # TODO If we have base_url (or maybe if a particular switch is true), attempt to fetch models from endpoint.
        # Is this where we would do it?
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'user_prompt': ('STRING', {
                    'forceInput': True,
                }),
                'base_url': ('STRING', {
                    'multiline': False,
                    'default': 'http://localhost:8080/v1',
                }),
                'model': (['model1', 'model2', 'model3'], {
                    'default': 'model1', # TODO
                }),
                'temperature': ('FLOAT', {
                    'min': 0.0,
                    'default': 0.8,
                }),
            },
            'optional': {
                'system_prompt': ('STRING', {
                    'forceInput': True,
                }),
                'api_key_env_var': ('STRING', {
                    'multiline': False,
                    'default': '',
                }),
            }
        }

    TITLE = 'OpenAI-like LLM'

    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('completion',)

    FUNCTION = 'execute'

    CATEGORY = 'YALLM'

    # def check_lazy_status(self, user_prompt, base_url, model, system_prompt, api_key_env_var):
    #     # TODO don't really have anything lazy yet
    #     return []

    def execute(self, user_prompt, base_url, model, temperature, system_prompt=None, api_key_env_var=None):
        api_key = "none" # The openai module needs *something*
        if api_key_env_var:
            api_key = os.environ[api_key_env_var] # TODO is there a way to fail gracefully or at least display error message?

        llm = openai.OpenAI(base_url=base_url, api_key=api_key)

        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': user_prompt})

        output = llm.chat.completions.create(
            messages=messages,
            model=model,
            temperature=temperature,
        )

        # TODO Since we're not streaming, we'll need some sort of timeout. How to specify that?

        return (output.choices[0].message.content,)


NODE_CLASS_MAPPINGS = {
    'LLMPrompt': LLMPrompt,
    'OpenAILikeLLM': OpenAILikeLLM,
}
