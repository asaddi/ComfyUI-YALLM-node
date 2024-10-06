# Yet Another LLM Node (for ComfyUI)

Yet another set of LLM nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI). This one only supports OpenAI-like APIs, but of course can be used with local LLM providers such as [llama.cpp](https://github.com/ggerganov/llama.cpp) (and, I believe, ollama and LMStudio, among others).

This project mainly served as an exercise in creating ComfyUI nodes.

![sample workflow](yallm-sample.png)

(the "Show Text" node is from the very useful https://github.com/pythongosssss/ComfyUI-Custom-Scripts)

## Features

* Connection profiles are defined externally in `models.yaml`, optionally pulling API URLs & API keys from environment variables.

   Since they're externally defined, there's no chance for your precious API keys to leak via workflow metadata.
* Only uses the chat completion API endpoint. No messing around with prompt templates.
* Can set a system prompt.
* The usual set of LLM samplers (temperature, top-k, top-p, min-p)

   For llama.cpp at least, the order you chain them together affects their order of application. (Other llama.cpp-derived providers might also respect the `samplers` option.)

   Also note that the OpenAI API spec only officially supports temperature and top-p, so what's actually supported will depend on your provider.
* Seed is exposed and configurable, allowing some degree of determinism.
* Can optionally pass in an image and query the LLM about it. Of course this means the LLM at your remote API needs to support images. Also be mindful that some VLMs like Llama 3.2 Vision don't support a system prompt when prompting with an image. (Some providers will just silently fail!)

Aside from adding more samplers (as requested/as I need them), I consider this project feature complete.

Though don't be surprised if I start adding RAG features or something in the future.

## Installation

Clone this project into your `custom_nodes` directory.

Then within the `ComfyUI-YALLM-node` directory:

    cp models.yaml.example models.yaml

and edit `models.yaml` if you would like to use more than just the `http://localhost:8080/v1` endpoint.

Finally, install the dependencies:

    pip install -r requirements.txt

or

    path/to/ComfyUI/python_embeded/python -m pip install -r requirements.txt

Just make sure `pip`/`python` is the same one used for your ComfyUI installation.

## License

Licensed under [BSD-2-Clause-Patent](https://opensource.org/license/bsdpluspatent).
