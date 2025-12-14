"""
Validation script for Dual-Stream GPT-2.

Loads a checkpoint and shows detailed analysis of model predictions on
validation samples, including input text, tokenization, logprobs, and
predicted tokens for both streams.

Usage:
    python validate.py checkpoints/best_model.pt --num_samples 5
    python validate.py checkpoints/best_model.pt --num_samples 5 --seed 42
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from config import DualStreamConfig, TrainingConfig
from model import DualStreamGPT2
from tokenizer import DualStreamTokenizer
from dataset import create_dataloader


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> DualStreamGPT2:
    """Load model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get config from checkpoint or use default
    config = checkpoint.get("config", DualStreamConfig())

    model = DualStreamGPT2(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    step = checkpoint.get("global_step", "unknown")
    print(f"Loaded model from step {step}")

    return model, config


def analyze_sample(
    model: DualStreamGPT2,
    tokenizer: DualStreamTokenizer,
    batch,
    sample_idx: int,
    config: DualStreamConfig,
    device: torch.device,
    max_display_tokens: int = 30,
) -> dict:
    """
    Analyze a single sample from a batch.

    Returns dict with detailed analysis including:
    - Input tokens and text
    - Model predictions (top-k tokens and probs)
    - Per-position loss
    """
    # Extract single sample
    input_ids_a = batch.input_ids_a[sample_idx:sample_idx+1].to(device)
    input_ids_b = batch.input_ids_b[sample_idx:sample_idx+1].to(device)
    attention_mask = batch.attention_mask[sample_idx:sample_idx+1].to(device)
    labels_a = batch.labels_a[sample_idx:sample_idx+1].to(device)
    labels_b = batch.labels_b[sample_idx:sample_idx+1].to(device)

    # Get valid length (before padding)
    valid_len = attention_mask[0].sum().item()
    valid_len = min(valid_len, max_display_tokens)

    with torch.no_grad():
        output = model(
            input_ids_a=input_ids_a,
            input_ids_b=input_ids_b,
            attention_mask=attention_mask,
            labels_a=labels_a,
            labels_b=labels_b,
        )

    # Get logits and compute probabilities
    logits_a = output.logits_a[0]  # (seq_len, main_vocab_size)
    logits_b = output.logits_b[0]  # (seq_len, pidgin_vocab_size)

    probs_a = F.softmax(logits_a, dim=-1)
    probs_b = F.softmax(logits_b, dim=-1)

    # Decode input text
    input_a_ids = input_ids_a[0, :valid_len].tolist()
    input_b_ids = input_ids_b[0, :valid_len].tolist()

    text_a = tokenizer.decode_main(input_a_ids)
    text_b = tokenizer.decode_pidgin(input_b_ids)

    # Analyze predictions position by position
    analysis = {
        "text_a": text_a,
        "text_b": text_b,
        "valid_len": valid_len,
        "loss": output.loss.item() if output.loss is not None else None,
        "loss_a": output.loss_a.item() if output.loss_a is not None else None,
        "loss_b": output.loss_b.item() if output.loss_b is not None else None,
        "positions": [],
    }

    # Get token strings for display
    _, tokens_a = tokenizer.main.encode_with_tokens(text_a)
    _, tokens_b = tokenizer.pidgin.encode_with_tokens(text_b)

    # Analyze each position (for next-token prediction)
    for pos in range(min(valid_len - 1, max_display_tokens - 1)):
        # Target is next token
        target_a = input_a_ids[pos + 1]
        target_b = input_b_ids[pos + 1]
        target_b_local = target_b - config.main_vocab_size  # local index

        # Get prediction probs for target
        prob_a = probs_a[pos, target_a].item()
        prob_b = probs_b[pos, target_b_local].item() if target_b_local >= 0 else 0.0

        # Get top-10 predictions
        top10_a = torch.topk(probs_a[pos], k=10)
        top10_b = torch.topk(probs_b[pos], k=10)

        # Decode top predictions - store token ID, decoded string, and probability
        top10_a_tokens = []
        for tid, p in zip(top10_a.indices.tolist(), top10_a.values.tolist()):
            try:
                tok_str = tokenizer.main.decode([tid])
            except:
                tok_str = f"<{tid}>"
            top10_a_tokens.append((tid, tok_str, p))

        top10_b_tokens = []
        for tid, p in zip(top10_b.indices.tolist(), top10_b.values.tolist()):
            try:
                tok_str = tokenizer.pidgin.decode([tid + config.main_vocab_size])
            except:
                tok_str = f"<{tid}>"
            top10_b_tokens.append((tid, tok_str, p))  # tid is local (0-999)

        # Decode target tokens using the same method as predictions for consistency
        try:
            target_a_str = tokenizer.main.decode([target_a])
        except:
            target_a_str = f"<{target_a}>"
        try:
            target_b_str = tokenizer.pidgin.decode([target_b])
        except:
            target_b_str = f"<{target_b}>"

        pos_info = {
            "pos": pos,
            "input_token_a": tokens_a[pos] if pos < len(tokens_a) else "?",
            "input_token_b": tokens_b[pos] if pos < len(tokens_b) else "?",
            "target_id_a": target_a,
            "target_id_b": target_b_local,  # local index for comparison
            "target_token_a": target_a_str,
            "target_token_b": target_b_str,
            "target_prob_a": prob_a,
            "target_prob_b": prob_b,
            "target_logprob_a": math.log(prob_a) if prob_a > 0 else float("-inf"),
            "target_logprob_b": math.log(prob_b) if prob_b > 0 else float("-inf"),
            "top10_a": top10_a_tokens,
            "top10_b": top10_b_tokens,
        }
        analysis["positions"].append(pos_info)

    return analysis


def print_analysis(analysis: dict, sample_num: int) -> None:
    """Print formatted analysis of a sample."""
    print("=" * 80)
    print(f"SAMPLE {sample_num}")
    print("=" * 80)
    print()

    print(f"Stream A text ({analysis['valid_len']} tokens):")
    print(f"  {analysis['text_a'][:200]}{'...' if len(analysis['text_a']) > 200 else ''}")
    print()

    print(f"Stream B text ({analysis['valid_len']} tokens):")
    print(f"  {analysis['text_b'][:200]}{'...' if len(analysis['text_b']) > 200 else ''}")
    print()

    if analysis['loss'] is not None:
        ppl_a = math.exp(min(analysis['loss_a'], 100))
        ppl_b = math.exp(min(analysis['loss_b'], 100))
        print(f"Loss: {analysis['loss']:.4f} | Loss_A: {analysis['loss_a']:.4f} (ppl={ppl_a:.2f}) | Loss_B: {analysis['loss_b']:.4f} (ppl={ppl_b:.2f})")
        print()

    # Show first few positions in detail with top 10 predictions
    print("Position-by-position analysis (first 5 positions):")

    for pos_info in analysis["positions"][:5]:
        print("-" * 80)
        print(f"Position {pos_info['pos']}")
        print("-" * 80)

        # Stream A - compare by token ID
        target_id_a = pos_info["target_id_a"]
        rank_a = next((i+1 for i, (tid, tok, p) in enumerate(pos_info["top10_a"]) if tid == target_id_a), ">10")
        print(f"  Stream A: '{pos_info['input_token_a']}' → '{pos_info['target_token_a']}' (P={pos_info['target_prob_a']:.4f}, rank={rank_a})")
        print(f"  Top 10 predictions:")
        for i, (tid, tok, prob) in enumerate(pos_info["top10_a"]):
            marker = "←" if tid == target_id_a else ""
            print(f"    {i+1:2}. '{tok}' ({prob:.4f}) {marker}")

        print()

        # Stream B - compare by token ID
        target_id_b = pos_info["target_id_b"]
        rank_b = next((i+1 for i, (tid, tok, p) in enumerate(pos_info["top10_b"]) if tid == target_id_b), ">10")
        print(f"  Stream B: '{pos_info['input_token_b']}' → '{pos_info['target_token_b']}' (P={pos_info['target_prob_b']:.4f}, rank={rank_b})")
        print(f"  Top 10 predictions:")
        for i, (tid, tok, prob) in enumerate(pos_info["top10_b"]):
            marker = "←" if tid == target_id_b else ""
            print(f"    {i+1:2}. '{tok}' ({prob:.4f}) {marker}")

        print()


def compute_accuracy(analysis: dict) -> tuple[float, float]:
    """Compute top-1 accuracy for both streams using token ID comparison."""
    correct_a = 0
    correct_b = 0
    total = len(analysis["positions"])

    for pos_info in analysis["positions"]:
        # Compare by token ID, not string
        if pos_info["top10_a"] and pos_info["target_id_a"] == pos_info["top10_a"][0][0]:
            correct_a += 1
        if pos_info["top10_b"] and pos_info["target_id_b"] == pos_info["top10_b"][0][0]:
            correct_b += 1

    acc_a = correct_a / total if total > 0 else 0
    acc_b = correct_b / total if total > 0 else 0
    return acc_a, acc_b


def main():
    parser = argparse.ArgumentParser(description="Validate Dual-Stream GPT-2")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to analyze")
    parser.add_argument("--max_tokens", type=int, default=30, help="Max tokens to display per sample")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible sampling")
    args = parser.parse_args()

    # Set random seeds if provided
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"Using seed: {args.seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model, config = load_model_from_checkpoint(Path(args.checkpoint), device)

    # Tokenizer
    print("Initializing tokenizer...")
    tokenizer = DualStreamTokenizer("words.txt", config.pidgin_vocab_size)

    # Load validation data - load more examples so we can sample from them
    print("Loading validation data...")
    training_config = TrainingConfig()
    val_dataloader = create_dataloader(
        tokenizer=tokenizer,
        config=config,
        training_config=training_config,
        split="validation",
        max_examples=max(100, args.num_samples * 10),  # Load enough to sample from
    )

    # Collect all batches to sample from
    print("Collecting samples...")
    all_batches = list(val_dataloader)

    # Create list of (batch_idx, sample_idx) for all samples
    all_samples = []
    for batch_idx, batch in enumerate(all_batches):
        for sample_idx in range(len(batch.input_ids_a)):
            all_samples.append((batch_idx, sample_idx))

    # Shuffle and select samples (seed already set earlier if provided)
    random.shuffle(all_samples)
    selected_samples = all_samples[:args.num_samples]

    print(f"Selected {len(selected_samples)} samples from {len(all_samples)} available")
    print()

    all_acc_a = []
    all_acc_b = []

    for i, (batch_idx, sample_idx) in enumerate(selected_samples):
        batch = all_batches[batch_idx]
        analysis = analyze_sample(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            sample_idx=sample_idx,
            config=config,
            device=device,
            max_display_tokens=args.max_tokens,
        )
        print_analysis(analysis, i + 1)

        acc_a, acc_b = compute_accuracy(analysis)
        all_acc_a.append(acc_a)
        all_acc_b.append(acc_b)

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Samples analyzed: {len(all_acc_a)}")
    print(f"Average top-1 accuracy Stream A: {sum(all_acc_a)/len(all_acc_a):.2%}")
    print(f"Average top-1 accuracy Stream B: {sum(all_acc_b)/len(all_acc_b):.2%}")


if __name__ == "__main__":
    main()
