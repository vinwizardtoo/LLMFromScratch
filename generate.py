import argparse
import json
import os
import tiktoken
import torch
from model import DecoderLM
from omegaconf import OmegaConf
from tqdm import trange
from utils import determine_device, enable_tf32


def softmax_with_temperature(
    logits: torch.FloatTensor, temperature: float
) -> torch.FloatTensor:
    """Turns logits into probabilities under softmax (with temperature)

    Args:
        logits: a 2d torch tensor of token ids (B x V)
        temperature: temperature of the softmax function

    Returns:
        a 2d torch tensor of token probabilities (B x V)
    """

    # to avoid division by 0
    temperature = max(temperature, 1e-5)
    
    # Applying the softmax with temperature
    temped_logits = logits / temperature
    max_logits = torch.max(temped_logits, dim=-1, keepdim=True)[0]
    exp_logits = torch.exp(temped_logits - max_logits)
    probs = exp_logits / exp_logits.sum(dim=-1, keepdim=True)

    return probs

@torch.inference_mode()
def generate(
    model: DecoderLM,
    device: str,
    tokenizer: tiktoken.Encoding,
    prefixes: list[str],
    batch_size: int,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
) -> list[str]:
    """Generates completions conditioned on prefixes"""
    generations = []
    for start_idx in trange(0, len(prefixes), batch_size, desc="Generating text"):
        batch = prefixes[start_idx:start_idx + batch_size]
        # Tokenize prefixes and pad sequences to equal length
        tokenized = [tokenizer.encode(prefix) for prefix in batch]
        max_len = max(len(tokens) for tokens in tokenized)
        input_ids = torch.full((len(batch), max_len), tokenizer.eot_token, dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.float, device=device)
        for idx, tokens in enumerate(tokenized):
            input_ids[idx, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            attention_mask[idx, :len(tokens)] = 1.0
       
        # Initialize variables for generation loop
        output_sequences = input_ids.clone()
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = model(input_ids, attention_mask=attention_mask)
           
            # Apply softmax with temperature
            probs = softmax_with_temperature(logits[:, -1, :], temperature)
           
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            output_sequences = torch.cat([output_sequences, next_token.unsqueeze(-1)], dim=-1)
           
            # Update input_ids for next iteration
            input_ids = output_sequences[:, -max_len:]
            attention_mask = torch.cat([attention_mask, torch.ones((len(batch), 1), dtype=torch.float, device=device)], dim=-1)
            attention_mask = attention_mask[:, -max_len:]
       
        # Convert generated token ids back to strings
        for tokens in output_sequences:
            generations.append(tokenizer.decode(tokens.tolist()))
   
    return generations

def main():
    enable_tf32()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=OmegaConf.load,
        required=True,
        help="the yaml config file used for model training",
    )
    parser.add_argument(
        "--prefixes",
        type=str,
        required=True,
        help="a json file with a list of strings as prefixes for generation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="temperature in sampling"
    )

    args = parser.parse_args()
    config = args.config
    with open(args.prefixes) as f:
        prefixes = [json.loads(line)["prefix"] for line in f]
    max_new_tokens = args.max_new_tokens
    temperature = args.temperature

    # initialize tokenizer and model
    model_path = os.path.join(config.output_dir, "model.pt")
    assert os.path.exists(model_path), f"no model checkpoint at {model_path}"
    tokenizer = tiktoken.get_encoding(config.tokenizer_encoding)
    device = determine_device() if config.device == "auto" else config.device
    model = DecoderLM(tokenizer.n_vocab, **config.model_config).to(device)
    model.load_state_dict(torch.load(model_path))

    # generate and save outputs
    model.eval()
    generations = generate(
        model,
        device,
        tokenizer,
        prefixes,
        config.batch_size,
        max_new_tokens,
        temperature,
    )

    generation_path = os.path.join(config.output_dir, "generation.jsonl")
    print(f"writing generations to {generation_path}")
    with open(generation_path, "w") as f:
        for prefix, generation in zip(prefixes, generations):
            json.dump({"prefix": prefix, "generation": generation}, f)
            f.write("\n")

    print("done!")


if __name__ == "__main__":
    main()