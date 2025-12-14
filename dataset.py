"""
Dataset utilities for Dual-Stream GPT-2 training.

Loads WikiText-103 and creates paired examples where Stream A and Stream B
contain unrelated text chunks. This tests whether the model can learn to
predict both streams without interference.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from tokenizer import DualStreamTokenizer
from config import DualStreamConfig, TrainingConfig


__all__ = ["DualStreamDataset", "create_dataloader"]


# =============================================================================
# Dataset
# =============================================================================

@dataclass
class DualStreamBatch:
    """
    A batch of dual-stream examples.

    Attributes:
        input_ids_a: Stream A token IDs, shape (batch, seq_len).
        input_ids_b: Stream B token IDs, shape (batch, seq_len).
        labels_a: Stream A targets (same as input_ids_a for LM).
        labels_b: Stream B targets (same as input_ids_b for LM).
        attention_mask: Attention mask, shape (batch, seq_len).
    """
    input_ids_a: torch.Tensor
    input_ids_b: torch.Tensor
    labels_a: torch.Tensor
    labels_b: torch.Tensor
    attention_mask: torch.Tensor

    def to(self, device: torch.device) -> "DualStreamBatch":
        """Move all tensors to device."""
        return DualStreamBatch(
            input_ids_a=self.input_ids_a.to(device),
            input_ids_b=self.input_ids_b.to(device),
            labels_a=self.labels_a.to(device),
            labels_b=self.labels_b.to(device),
            attention_mask=self.attention_mask.to(device),
        )


class DualStreamDataset(Dataset):
    """
    Dataset that pairs unrelated text chunks for dual-stream training.

    Each example consists of:
    - Stream A: A chunk of text tokenized with MainTokenizer
    - Stream B: A different, unrelated chunk tokenized with PidginTokenizer

    The pairing is randomized so streams are independent.

    Args:
        tokenizer: DualStreamTokenizer instance.
        max_length: Maximum sequence length.
        split: Dataset split ("train", "validation", "test").
        max_examples: Optional limit on number of examples.
    """

    def __init__(
        self,
        tokenizer: DualStreamTokenizer,
        max_length: int = 256,
        split: str = "train",
        max_examples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split

        # Load WikiText-103
        print(f"Loading WikiText-103 ({split})...")
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)

        # Filter out empty lines and very short texts
        self.texts = [
            text for text in dataset["text"]
            if text.strip() and len(text.strip()) > 50
        ]

        if max_examples:
            self.texts = self.texts[:max_examples]

        print(f"Loaded {len(self.texts)} text chunks")

        # Create shuffled indices for Stream B pairing
        self.stream_b_indices = list(range(len(self.texts)))
        random.shuffle(self.stream_b_indices)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a paired example.

        Returns dict with:
            - input_ids_a: List[int] for Stream A
            - input_ids_b: List[int] for Stream B
        """
        # Stream A: text at idx
        text_a = self.texts[idx]

        # Stream B: text at shuffled index (unrelated)
        text_b = self.texts[self.stream_b_indices[idx]]

        # Tokenize
        ids_a = self.tokenizer.encode_main(text_a)
        ids_b = self.tokenizer.encode_pidgin(text_b)

        # Truncate to max_length
        ids_a = ids_a[: self.max_length]
        ids_b = ids_b[: self.max_length]

        return {
            "input_ids_a": ids_a,
            "input_ids_b": ids_b,
        }


# =============================================================================
# Collation and DataLoader
# =============================================================================

def collate_fn(
    batch: list[dict],
    pad_token_a: int,
    pad_token_b: int,
    max_length: int,
) -> DualStreamBatch:
    """
    Collate examples into a batch with padding.

    Args:
        batch: List of examples from DualStreamDataset.
        pad_token_a: Padding token ID for Stream A.
        pad_token_b: Padding token ID for Stream B.
        max_length: Maximum sequence length.

    Returns:
        DualStreamBatch with padded tensors.
    """
    batch_size = len(batch)

    # Find max length in this batch
    max_len_a = max(len(ex["input_ids_a"]) for ex in batch)
    max_len_b = max(len(ex["input_ids_b"]) for ex in batch)
    seq_len = min(max(max_len_a, max_len_b), max_length)

    # Initialize tensors
    input_ids_a = torch.full((batch_size, seq_len), pad_token_a, dtype=torch.long)
    input_ids_b = torch.full((batch_size, seq_len), pad_token_b, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.long)

    for i, ex in enumerate(batch):
        ids_a = ex["input_ids_a"][:seq_len]
        ids_b = ex["input_ids_b"][:seq_len]

        # Use the shorter of the two to determine valid positions
        # Both streams should have same length for proper alignment
        valid_len = min(len(ids_a), len(ids_b))

        input_ids_a[i, :len(ids_a)] = torch.tensor(ids_a)
        input_ids_b[i, :len(ids_b)] = torch.tensor(ids_b)
        attention_mask[i, :valid_len] = 1

    # Labels are same as inputs for language modeling
    # Use -100 for padded positions (ignored in loss)
    labels_a = input_ids_a.clone()
    labels_b = input_ids_b.clone()
    labels_a[attention_mask == 0] = -100
    labels_b[attention_mask == 0] = -100

    return DualStreamBatch(
        input_ids_a=input_ids_a,
        input_ids_b=input_ids_b,
        labels_a=labels_a,
        labels_b=labels_b,
        attention_mask=attention_mask,
    )


def create_dataloader(
    tokenizer: DualStreamTokenizer,
    config: DualStreamConfig,
    training_config: TrainingConfig,
    split: str = "train",
    max_examples: Optional[int] = None,
) -> DataLoader:
    """
    Create a DataLoader for dual-stream training.

    Args:
        tokenizer: DualStreamTokenizer instance.
        config: DualStreamConfig with vocab sizes.
        training_config: TrainingConfig with batch size, etc.
        split: Dataset split.
        max_examples: Optional limit on examples.

    Returns:
        DataLoader yielding DualStreamBatch objects.
    """
    dataset = DualStreamDataset(
        tokenizer=tokenizer,
        max_length=training_config.max_seq_length,
        split=split,
        max_examples=max_examples,
    )

    # Padding tokens (actual value doesn't matter since we use attention mask)
    # Both streams use byte-level BPE, no special pad token needed
    pad_token_a = 0
    pad_token_b = config.pidgin_offset  # First token in pidgin range

    def collate(batch):
        return collate_fn(
            batch,
            pad_token_a=pad_token_a,
            pad_token_b=pad_token_b,
            max_length=training_config.max_seq_length,
        )

    return DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=(split == "train"),
        collate_fn=collate,
        num_workers=0,  # Keep simple for now
        pin_memory=True,
    )


# =============================================================================
# Verification
# =============================================================================

def verify_dataset(dataloader: DataLoader, config: DualStreamConfig) -> None:
    """Print statistics about a dataloader."""
    print("=" * 60)
    print("Dataset Verification")
    print("=" * 60)
    print()

    batch = next(iter(dataloader))
    print(f"Batch shapes:")
    print(f"  input_ids_a: {batch.input_ids_a.shape}")
    print(f"  input_ids_b: {batch.input_ids_b.shape}")
    print(f"  attention_mask: {batch.attention_mask.shape}")
    print()

    # Check token ranges
    print("Token ID ranges:")
    print(f"  Stream A: [{batch.input_ids_a.min()}, {batch.input_ids_a.max()}]")
    print(f"    Expected: [0, {config.main_vocab_size - 1}]")
    print(f"  Stream B: [{batch.input_ids_b.min()}, {batch.input_ids_b.max()}]")
    print(f"    Expected: [{config.pidgin_offset}, {config.pidgin_offset + config.pidgin_vocab_size - 1}]")
    print()

    # Verify ranges
    assert batch.input_ids_a.max() < config.main_vocab_size, "Stream A token out of range!"
    assert batch.input_ids_b.min() >= config.pidgin_offset, "Stream B token out of range!"
    assert batch.input_ids_b.max() < config.pidgin_offset + config.pidgin_vocab_size, "Stream B token out of range!"

    print("Sample from first example:")
    print(f"  Stream A (first 10 tokens): {batch.input_ids_a[0, :10].tolist()}")
    print(f"  Stream B (first 10 tokens): {batch.input_ids_b[0, :10].tolist()}")
    print(f"  Attention mask (first 10):  {batch.attention_mask[0, :10].tolist()}")
    print()
    print("Verification passed!")


def main() -> None:
    """CLI entry point for dataset verification."""
    from config import DualStreamConfig, TrainingConfig

    config = DualStreamConfig()
    training_config = TrainingConfig()

    print("Initializing tokenizer...")
    tokenizer = DualStreamTokenizer(config.pidgin_vocab_size)

    print()
    dataloader = create_dataloader(
        tokenizer=tokenizer,
        config=config,
        training_config=training_config,
        split="train",
        max_examples=1000,  # Limit for quick test
    )

    print()
    verify_dataset(dataloader, config)


if __name__ == "__main__":
    main()
