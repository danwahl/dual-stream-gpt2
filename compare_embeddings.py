"""
Compare input embeddings between two checkpoints.

Usage:
    python compare_embeddings.py checkpoint1.pt checkpoint2.pt
"""

import argparse
import torch

from config import DualStreamConfig


def find_embedding_key(state_dict):
    """Find the embedding weight key in the state dict."""
    for key in state_dict.keys():
        if "wte.weight" in key:
            return key
    raise KeyError(f"Could not find embedding weights. Keys: {list(state_dict.keys())[:10]}...")


def main():
    parser = argparse.ArgumentParser(description="Compare embeddings between checkpoints")
    parser.add_argument("checkpoint1", type=str, help="First checkpoint (e.g., initial)")
    parser.add_argument("checkpoint2", type=str, help="Second checkpoint (e.g., after training)")
    args = parser.parse_args()

    print(f"Loading {args.checkpoint1}...")
    ckpt1 = torch.load(args.checkpoint1, map_location="cpu", weights_only=False)

    print(f"Loading {args.checkpoint2}...")
    ckpt2 = torch.load(args.checkpoint2, map_location="cpu", weights_only=False)

    # Get config (from checkpoint or default)
    config = ckpt1.get("config", DualStreamConfig())

    # Find embedding key
    state_dict1 = ckpt1["model_state_dict"]
    state_dict2 = ckpt2["model_state_dict"]
    embed_key = find_embedding_key(state_dict1)
    print(f"Found embedding key: {embed_key}")

    # Extract embedding weights
    embed1 = state_dict1[embed_key]
    embed2 = state_dict2[embed_key]

    print()
    print("=" * 60)
    print("Embedding Comparison")
    print("=" * 60)
    print()
    print(f"Embedding shape: {embed1.shape}")
    print(f"Main vocab size: {config.main_vocab_size}")
    print(f"Pidgin vocab size: {config.pidgin_vocab_size}")
    print(f"Pidgin offset: {config.pidgin_offset}")
    print()

    # Overall difference
    diff = embed2 - embed1
    print("Overall embedding changes:")
    print(f"  Mean absolute diff: {diff.abs().mean():.6f}")
    print(f"  Max absolute diff:  {diff.abs().max():.6f}")
    print(f"  Std of diff:        {diff.std():.6f}")
    print()

    # Main region (0 to main_vocab_size)
    main_diff = diff[:config.main_vocab_size]
    print(f"Main region (tokens 0-{config.main_vocab_size - 1}):")
    print(f"  Mean absolute diff: {main_diff.abs().mean():.6f}")
    print(f"  Max absolute diff:  {main_diff.abs().max():.6f}")
    print(f"  Rows with any change: {(main_diff.abs().sum(dim=1) > 1e-8).sum().item()}")
    print()

    # Pidgin region (main_vocab_size to end)
    pidgin_diff = diff[config.pidgin_offset:]
    print(f"Pidgin region (tokens {config.pidgin_offset}-{embed1.shape[0] - 1}):")
    print(f"  Mean absolute diff: {pidgin_diff.abs().mean():.6f}")
    print(f"  Max absolute diff:  {pidgin_diff.abs().max():.6f}")
    print(f"  Rows with any change: {(pidgin_diff.abs().sum(dim=1) > 1e-8).sum().item()}")
    print()

    # Check EOS tokens specifically
    main_eos = config.main_eos_id
    pidgin_eos = config.pidgin_eos_id

    print("EOS token embeddings:")
    print(f"  Main EOS ({main_eos}) diff norm: {diff[main_eos].norm():.6f}")
    print(f"  Pidgin EOS ({pidgin_eos}) diff norm: {diff[pidgin_eos].norm():.6f}")
    print()

    # Show a few example rows with largest changes
    row_diffs = diff.abs().sum(dim=1)
    top_changed = row_diffs.topk(10)

    print("Top 10 most changed token embeddings:")
    for idx, val in zip(top_changed.indices.tolist(), top_changed.values.tolist()):
        region = "main" if idx < config.main_vocab_size else "pidgin"
        print(f"  Token {idx} ({region}): total diff = {val:.6f}")


if __name__ == "__main__":
    main()
