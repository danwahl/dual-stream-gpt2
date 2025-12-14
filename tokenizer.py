"""
Dual-stream tokenizers for the Dual-Stream GPT-2 experiment.

Both streams use truncated GPT-2 BPE tokenizers with different token ranges:
- Stream A (Main): tokens 0 to (GPT2_VOCAB - pidgin_size - 1)
- Stream B (Pidgin): tokens (GPT2_VOCAB - pidgin_size) to (GPT2_VOCAB - 1)

The vocab sizes are parameterized so the relative sizes can be adjusted for testing.
Default: 10,000 pidgin tokens, leaving 40,257 main tokens.

Usage:
    from tokenizer import DualStreamTokenizer

    tokenizer = DualStreamTokenizer(pidgin_vocab_size=10000)
    main_ids = tokenizer.encode_main("Hello world")
    pidgin_ids = tokenizer.encode_pidgin("the big thing")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from transformers import GPT2Tokenizer
from tokenizers import Tokenizer, models, pre_tokenizers, decoders

if TYPE_CHECKING:
    pass

__all__ = [
    "GPT2_VOCAB_SIZE",
    "DEFAULT_PIDGIN_VOCAB_SIZE",
    "MainTokenizer",
    "PidginTokenizer",
    "DualStreamTokenizer",
]

# =============================================================================
# Constants
# =============================================================================

GPT2_VOCAB_SIZE = 50257
"""Total vocabulary size of GPT-2 (256 bytes + 50,001 merges)."""

DEFAULT_PIDGIN_VOCAB_SIZE = 10000
"""Default number of tokens reserved for the pidgin (Stream B) vocabulary."""


# =============================================================================
# Internal helpers
# =============================================================================

def _create_truncated_tokenizer(
    start_token: int,
    end_token: int,
) -> Tokenizer:
    """
    Create a truncated BPE tokenizer using a subset of GPT-2's vocab.

    Uses tokens from start_token to end_token-1 (inclusive). The tokenizer
    will only produce tokens in this range by using a truncated merge list.

    For Stream A (start=0), rare patterns that would produce higher token IDs
    naturally fall back to their component tokens.

    For Stream B (start>0), we remap the vocab so that encoding produces
    local IDs [0, vocab_size), and the caller applies the offset.

    Args:
        start_token: First token ID to include.
        end_token: One past the last token ID to include.

    Returns:
        A configured BPE tokenizer with local IDs [0, end_token - start_token).
    """
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    vocab_size = end_token - start_token

    if start_token == 0:
        # Stream A: Use first N tokens directly
        vocab = {
            token: token_id
            for token, token_id in gpt2_tokenizer.encoder.items()
            if token_id < end_token
        }

        # Truncate merge list: need (vocab_size - 256) merges
        num_merges = vocab_size - 256
        sorted_merges = sorted(gpt2_tokenizer.bpe_ranks.items(), key=lambda x: x[1])
        truncated_merges = [pair for pair, _rank in sorted_merges[:num_merges]]
    else:
        # Stream B: Use tokens from upper portion, remapped to local IDs [0, vocab_size)
        # We take the LAST vocab_size tokens from GPT-2's vocabulary
        vocab = {}
        for token, token_id in gpt2_tokenizer.encoder.items():
            if start_token <= token_id < end_token:
                # Remap to local index
                local_id = token_id - start_token
                vocab[token] = local_id

        # For Stream B, we use all merges but they'll only produce tokens in our vocab
        # Merges that would produce tokens outside our range won't apply
        # We need merges that produce tokens in [start_token, end_token)
        # The merge list is ordered by frequency, and later merges produce higher token IDs
        # We want merges that produce token IDs >= start_token
        sorted_merges = sorted(gpt2_tokenizer.bpe_ranks.items(), key=lambda x: x[1])

        # Find which merges produce tokens in our range
        # Merge i produces token (256 + i), so we need merges where 256 + i >= start_token
        # and 256 + i < end_token
        truncated_merges = []
        for pair, rank in sorted_merges:
            produced_token = 256 + rank
            if start_token <= produced_token < end_token:
                truncated_merges.append(pair)

    # Build tokenizer
    tokenizer = Tokenizer(
        models.BPE(vocab=vocab, merges=truncated_merges, fuse_unk=False)
    )
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    return tokenizer


# =============================================================================
# Tokenizer classes
# =============================================================================

class MainTokenizer:
    """
    Truncated BPE tokenizer for Stream A (main vocabulary).

    Uses GPT-2's vocabulary but only the first N tokens, where
    N = GPT2_VOCAB_SIZE - pidgin_vocab_size. Rare patterns that would
    require higher token IDs decompose into component tokens.

    Attributes:
        vocab_size: Number of tokens in this vocabulary.
        tokenizer: The underlying HuggingFace tokenizer.
    """

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.tokenizer = _create_truncated_tokenizer(0, vocab_size)

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs (all guaranteed < vocab_size)."""
        return self.tokenizer.encode(text).ids

    def encode_with_tokens(self, text: str) -> tuple[list[int], list[str]]:
        """Encode text, returning both token IDs and token strings."""
        encoding = self.tokenizer.encode(text)
        return encoding.ids, encoding.tokens

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text."""
        return self.tokenizer.decode(ids)


class PidginTokenizer:
    """
    Truncated BPE tokenizer for Stream B (pidgin vocabulary).

    Uses the upper portion of GPT-2's vocabulary, remapped to local IDs.
    Token IDs are offset by main_vocab_size so they occupy the range
    [offset, offset + vocab_size).

    Attributes:
        vocab_size: Number of tokens in this vocabulary.
        offset: Value added to raw token IDs (= main_vocab_size).
        pad_token_id: Token ID for padding (with offset).
        unk_token_id: Token ID for unknown (with offset).
        eos_token_id: Token ID for end-of-sequence (with offset).
    """

    def __init__(self, vocab_size: int, offset: int) -> None:
        self.vocab_size = vocab_size
        self.offset = offset

        # Create tokenizer using upper portion of GPT-2 vocab
        self.tokenizer = _create_truncated_tokenizer(offset, offset + vocab_size)

        # Special token IDs (with offset applied)
        # These are the first few local tokens, offset to global range
        self.pad_token_id = offset + 0
        self.unk_token_id = offset + 1
        self.eos_token_id = offset + 2

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs (offset applied)."""
        return [id + self.offset for id in self.tokenizer.encode(text).ids]

    def encode_with_tokens(self, text: str) -> tuple[list[int], list[str]]:
        """Encode text, returning both token IDs and token strings."""
        encoding = self.tokenizer.encode(text)
        ids = [id + self.offset for id in encoding.ids]
        return ids, encoding.tokens

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text (offset removed internally)."""
        local_ids = [id - self.offset for id in ids]
        return self.tokenizer.decode(local_ids)

    def get_vocab(self) -> dict[str, int]:
        """Get vocabulary mapping token strings to IDs (with offset)."""
        return {
            token: id + self.offset
            for token, id in self.tokenizer.get_vocab().items()
        }


class DualStreamTokenizer:
    """
    Combined tokenizer for both streams.

    Partitions GPT-2's embedding space between main and pidgin vocabularies:
    - Stream A (main): tokens 0 to main_vocab_size - 1
    - Stream B (pidgin): tokens main_vocab_size to GPT2_VOCAB_SIZE - 1

    Both streams use truncated GPT-2 BPE, ensuring well-trained subword merges.

    Args:
        pidgin_vocab_size: Number of tokens for Stream B (default: 10000).

    Attributes:
        main_vocab_size: Derived as GPT2_VOCAB_SIZE - pidgin_vocab_size.
        pidgin_vocab_size: Size of Stream B vocabulary.
        main: The MainTokenizer instance.
        pidgin: The PidginTokenizer instance.
    """

    def __init__(
        self,
        pidgin_vocab_size: int = DEFAULT_PIDGIN_VOCAB_SIZE,
    ) -> None:
        self.main_vocab_size = GPT2_VOCAB_SIZE - pidgin_vocab_size
        self.pidgin_vocab_size = pidgin_vocab_size

        self.main = MainTokenizer(self.main_vocab_size)
        self.pidgin = PidginTokenizer(
            pidgin_vocab_size,
            offset=self.main_vocab_size,
        )

    def encode_main(self, text: str) -> list[int]:
        """Encode text for Stream A."""
        return self.main.encode(text)

    def encode_pidgin(self, text: str) -> list[int]:
        """Encode text for Stream B."""
        return self.pidgin.encode(text)

    def decode_main(self, ids: list[int]) -> str:
        """Decode Stream A token IDs to text."""
        return self.main.decode(ids)

    def decode_pidgin(self, ids: list[int]) -> str:
        """Decode Stream B token IDs to text."""
        return self.pidgin.decode(ids)


# =============================================================================
# Verification / CLI
# =============================================================================

def verify_tokenizers(dual_tokenizer: DualStreamTokenizer) -> bool:
    """
    Verify tokenizers work correctly and print diagnostic information.

    Args:
        dual_tokenizer: The DualStreamTokenizer to verify.

    Returns:
        True if all checks pass.

    Raises:
        AssertionError: If token ID ranges overlap or other invariants fail.
    """
    main_size = dual_tokenizer.main_vocab_size
    pidgin_size = dual_tokenizer.pidgin_vocab_size
    pidgin_offset = main_size

    print("=" * 60)
    print("Dual-Stream Tokenizer Verification")
    print("=" * 60)
    print()
    print(f"GPT-2 total vocab: {GPT2_VOCAB_SIZE:,}")
    print(f"Stream A (main):   {main_size:,} tokens (IDs 0-{main_size - 1})")
    print(f"Stream B (pidgin): {pidgin_size:,} tokens (IDs {pidgin_offset}-{pidgin_offset + pidgin_size - 1})")
    print()

    # Sample pidgin vocabulary (first 15 tokens)
    print("Sample pidgin vocabulary (first 15 tokens):")
    pidgin_vocab = dual_tokenizer.pidgin.get_vocab()
    for token, token_id in sorted(pidgin_vocab.items(), key=lambda x: x[1])[:15]:
        print(f"  {token_id}: {token!r}")
    print(f"  ... ({len(pidgin_vocab)} total)")
    print()

    # Stream A examples
    print("Stream A (Main) examples:")
    for text in ["Hello world", "The quick brown fox", "Machine learning"]:
        ids, tokens = dual_tokenizer.main.encode_with_tokens(text)
        print(f"  '{text}'")
        print(f"    IDs:    {ids}")
        print(f"    Tokens: {tokens}")
    print()

    # Truncation test for Stream A: show that rare words get decomposed
    print("Stream A truncation test (rare patterns decompose into subwords):")
    rare_patterns = [
        "cryptocurrency",  # technical jargon
        "biodegradable",   # compound word
        "quinoa",          # uncommon word
    ]
    for pattern in rare_patterns:
        ids, tokens = dual_tokenizer.main.encode_with_tokens(pattern)
        print(f"  '{pattern}' -> {tokens} ({len(tokens)} tokens)")
    print()

    # Stream B examples
    print("Stream B (Pidgin) examples:")
    for text in ["The Big Red Thing", "Understanding the World", "Hello there"]:
        ids, tokens = dual_tokenizer.pidgin.encode_with_tokens(text)
        print(f"  '{text}'")
        print(f"    IDs:    {ids}")
        print(f"    Tokens: {tokens}")
    print()

    # Verify non-overlapping ranges
    print("Verifying token ID ranges...")
    main_ids = dual_tokenizer.encode_main("The quick brown fox")
    pidgin_ids = dual_tokenizer.encode_pidgin("the big thing")

    assert all(id < main_size for id in main_ids), f"Main IDs must be < {main_size}"
    assert all(id >= pidgin_offset for id in pidgin_ids), f"Pidgin IDs must be >= {pidgin_offset}"

    print(f"  Main IDs:   {main_ids} (all < {main_size})")
    print(f"  Pidgin IDs: {pidgin_ids} (all >= {pidgin_offset})")
    print()
    print("Verification passed!")
    return True


def main() -> None:
    """CLI entry point."""
    import sys

    pidgin_size = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PIDGIN_VOCAB_SIZE

    print(f"Pidgin vocab size: {pidgin_size}")
    print()

    tokenizer = DualStreamTokenizer(pidgin_size)
    verify_tokenizers(tokenizer)


if __name__ == "__main__":
    main()
