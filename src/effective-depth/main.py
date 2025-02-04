import os
import argparse
import logging
import torch
import torch.distributed as dist

from layers import TPLlamaDecoderLayer, TPParallelLlamaDecoderLayer
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

# Initialize parser
parser = argparse.ArgumentParser(description="Set global settings for the application.")
# Model settings
parser.add_argument(
    "--model_key",
    type=str,
    default="meta-llama/Llama-3.2-3B",
    help="Key for the model to use.",
)
# Approximation settings
parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility.")
parser.add_argument(
    "--start_idx", type=int, default=21, help="Index of the first block to patch."
)
parser.add_argument(
    "--end_idx",
    type=int,
    default=25,
    help="Index of the last block to patch (exclusive).",
)


def get_model(model_key):
    assert "meta-llama" in model_key

    tokenizer = AutoTokenizer.from_pretrained(model_key, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_size = "right"
    tokenizer.add_eos_token = False

    model = AutoModelForCausalLM.from_pretrained(model_key, trust_remote_code=True)
    model.config.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def replace_layers(model, config, device: str, start_idx: int, end_idx: int):
    assert (end_idx - start_idx) % 2 == 0

    # Replace non-layer-parallel layers with normal tensor parallelism
    for i, layer in enumerate(model.model.layers):
        if i < start_idx or i >= end_idx:
            tp_layer = TPLlamaDecoderLayer(config=config, layer_idx=i).to(device)
            tp_layer.init_from_llama_decoder_layer(layer)
            layer = tp_layer

    # Replace the rest with Layer Parallelism, reducing the models effective depth
    new_layers = torch.nn.ModuleList()
    for i in range(start_idx, end_idx, 2):
        layers = (model.model.layers[i], model.model.layers[i + 1])
        tp_layer = TPParallelLlamaDecoderLayer(config=config, layers=layers).to(device)
        new_layers.append(tp_layer)
    model.model.layers = torch.nn.ModuleList(
        [
            *model.model.layers[:start_idx],
            *new_layers,
            *model.model.layers[end_idx:],
        ]
    )

    return model


def main(args):
    ddp = int(os.environ.get("RANK", -1)) != -1

    if ddp:
        assert torch.cuda.is_available()
        dist.init_process_group(backend="nccl")
        master_process = dist.get_rank() == 0
        device = f"cuda:{dist.get_rank()}"
        torch.cuda.set_device(device)
    else:
        raise RuntimeError(
            "Not using multiple GPUs! Have you run this script without torchrun?"
        )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if master_process:
        logging.basicConfig(level=logging.INFO)
        logging.info(f"Arguments:\n{vars(args)}\n")

    config = AutoConfig.from_pretrained(args.model_key)
    model, tokenizer = get_model(args.model_key)
    model = model.to(device)
    model = replace_layers(model, config, device, args.start_idx, args.end_idx)

    prompt = "Once upon a time"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(
        input_ids, max_length=1024, num_return_sequences=1, do_sample=True
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    if master_process:
        print(generated_text)

    dist.destroy_process_group()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
