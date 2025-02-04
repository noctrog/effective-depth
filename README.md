# Leveraging the True Depth of LLMs

## Installation

Fetch the environment specified in the `pyproject.toml`. If using `uv`, you can use `uv sync`.

## Generating outputs

The `src/effective-depth/main.py` will generate a string

`torchrun --standalone --nproc_per_node=2 src/effective-depth/main.py`

Arguments:
- `--model_key`: the huggingface-hub model key. Only Llama models are supported.
- `--seed`: specifies the seed for reproducibility.
- `--start_idx`: start index of the sequence of layers to be parallelized with Layer Parallelism.
- `--end_idx`: end index of the sequence of layers to be parallelized with Layer Parallelism.

## Layer Parallelism

The `TPParallelLlamaDecoderLayer` class in `src/effective-depth/layers.py` implements Layer Parallelism on Hugging Face's Llama implementation.
