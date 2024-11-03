# Examples

Note: Nowadays I prefer the "LLM Provider" node over "LLM Model". Be sure to customize your `providers.yaml`, or if you prefer it, swap back to the "LLM Model" node.

## prompt-enhancer

Uses a system prompt posted by a now-deleted account [on reddit](https://www.reddit.com/r/LocalLLaMA/comments/1fi0jkj/comment/lnef616/). Though helpfully, the comment still remains.

## img2text2img

This just takes the example [Stable Diffusion 3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) workflow but replaces the prompt with LLM output.

The YALLM nodes query a (vision) LLM, asking for a description of the loaded image.
