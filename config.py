"""
Configuration for the Dual-Stream GPT-2 experiment.

This module centralizes all hyperparameters and constants for easy tuning.
"""

from dataclasses import dataclass


# =============================================================================
# Vocabulary Layout
# =============================================================================

GPT2_VOCAB_SIZE = 50257
"""Total vocabulary size of GPT-2."""

DEFAULT_PIDGIN_VOCAB_SIZE = 10000
"""Default number of tokens for Stream B (pidgin vocabulary)."""

# Derived constants (for default config with 10K pidgin)
# Main vocab: 0 to 40,256 (40,257 tokens)
# Pidgin vocab: 40,257 to 50,256 (10,000 tokens)


# =============================================================================
# Model Configuration
# =============================================================================

@dataclass
class DualStreamConfig:
    """
    Configuration for DualStreamGPT2 model.

    Attributes:
        pidgin_vocab_size: Number of tokens for Stream B.
        main_vocab_size: Derived as GPT2_VOCAB_SIZE - pidgin_vocab_size.
        embed_init_std: Std for reinitializing pidgin embeddings.
        loss_weight_a: Weight for Stream A loss.
        loss_weight_b: Weight for Stream B loss.
        gpt2_model_name: HuggingFace model identifier.
    """

    # Vocabulary
    pidgin_vocab_size: int = DEFAULT_PIDGIN_VOCAB_SIZE

    # Initialization
    embed_init_std: float = 0.02  # matches GPT-2's original init

    # Loss weighting
    loss_weight_a: float = 1.0
    loss_weight_b: float = 1.0

    # Base model
    gpt2_model_name: str = "gpt2"  # GPT-2 Small (124M params)

    def __post_init__(self):
        """Compute derived values."""
        self.main_vocab_size = GPT2_VOCAB_SIZE - self.pidgin_vocab_size

        # Special token IDs
        # Stream A: EOS relocated from 50256 to last position in main vocab
        self.main_eos_id = self.main_vocab_size - 1  # 49256 by default

        # Stream B special tokens (with offset)
        self.pidgin_offset = self.main_vocab_size
        self.pidgin_pad_id = self.pidgin_offset + 0   # 49257
        self.pidgin_unk_id = self.pidgin_offset + 1   # 49258
        self.pidgin_eos_id = self.pidgin_offset + 2   # 49259

        # Original GPT-2 EOS location (for copying during init)
        self.original_eos_id = 50256

    def __repr__(self) -> str:
        return (
            f"DualStreamConfig(\n"
            f"  main_vocab_size={self.main_vocab_size},\n"
            f"  pidgin_vocab_size={self.pidgin_vocab_size},\n"
            f"  main_eos_id={self.main_eos_id},\n"
            f"  pidgin_pad_id={self.pidgin_pad_id},\n"
            f"  pidgin_eos_id={self.pidgin_eos_id},\n"
            f"  loss_weight_a={self.loss_weight_a},\n"
            f"  loss_weight_b={self.loss_weight_b},\n"
            f")"
        )


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """
    Training hyperparameters (for Phase 4).

    Optimized for GTX 1070 (8GB VRAM).
    """

    # Batch size
    batch_size: int = 4
    gradient_accumulation_steps: int = 4  # effective batch = 16

    # Sequence length
    max_seq_length: int = 256

    # Optimizer
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100

    # Training duration
    num_epochs: int = 3
    max_steps: int = -1  # -1 means use num_epochs

    # Mixed precision
    fp16: bool = True

    # Logging
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000
