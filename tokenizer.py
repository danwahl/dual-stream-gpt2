"""
Dual-stream tokenizers for the Dual-Stream GPT-2 experiment.

Stream A (Main): Truncated GPT-2 BPE tokenizer, tokens 0 to (GPT2_VOCAB - pidgin_size - 1)
Stream B (Pidgin): BPE tokenizer trained on words.txt, tokens offset by main vocab size

The vocab sizes are parameterized so the relative sizes can be adjusted for testing.
Default: 1000 pidgin tokens, leaving 49,257 main tokens.

Usage:
    from tokenizer import DualStreamTokenizer

    tokenizer = DualStreamTokenizer("words.txt", pidgin_vocab_size=1000)
    main_ids = tokenizer.encode_main("Hello world")
    pidgin_ids = tokenizer.encode_pidgin("the big thing")
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from transformers import GPT2Tokenizer
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

if TYPE_CHECKING:
    from tokenizers import Encoding

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

DEFAULT_PIDGIN_VOCAB_SIZE = 1000
"""Default number of tokens reserved for the pidgin (Stream B) vocabulary."""


# =============================================================================
# Internal helpers
# =============================================================================

def _create_truncated_main_tokenizer(main_vocab_size: int) -> Tokenizer:
    """
    Create a truncated BPE tokenizer for Stream A using GPT-2's vocab.

    Uses only the first `main_vocab_size` tokens by truncating the merge list.
    Rare patterns that would produce higher token IDs naturally fall back
    to their component tokens.

    Args:
        main_vocab_size: Number of tokens to include (256 bytes + N merges).

    Returns:
        A configured BPE tokenizer.
    """
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Extract first main_vocab_size tokens from vocabulary
    vocab = {
        token: token_id
        for token, token_id in gpt2_tokenizer.encoder.items()
        if token_id < main_vocab_size
    }

    # Truncate merge list: need (main_vocab_size - 256) merges
    num_merges = main_vocab_size - 256
    sorted_merges = sorted(gpt2_tokenizer.bpe_ranks.items(), key=lambda x: x[1])
    truncated_merges = [pair for pair, _rank in sorted_merges[:num_merges]]

    # Build tokenizer
    tokenizer = Tokenizer(
        models.BPE(vocab=vocab, merges=truncated_merges, fuse_unk=False)
    )
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    return tokenizer


def _train_pidgin_tokenizer(corpus_path: Path, vocab_size: int) -> Tokenizer:
    """
    Train a BPE tokenizer on the pidgin corpus.

    Args:
        corpus_path: Path to the training corpus (e.g., words.txt).
        vocab_size: Target vocabulary size.

    Returns:
        A trained BPE tokenizer with token IDs 0 to vocab_size-1.
    """
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=1,
        special_tokens=["<PAD>", "<UNK>", "<EOS>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train([str(corpus_path)], trainer)

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
        self.tokenizer = _create_truncated_main_tokenizer(vocab_size)

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
    BPE tokenizer for Stream B (pidgin vocabulary).

    Trained on a custom corpus (e.g., Thing Explainer words). Token IDs are
    offset by main_vocab_size so they occupy the range [offset, offset + vocab_size).

    Attributes:
        vocab_size: Number of tokens in this vocabulary.
        offset: Value added to raw token IDs (= main_vocab_size).
        pad_token_id: Token ID for <PAD> (with offset).
        unk_token_id: Token ID for <UNK> (with offset).
        eos_token_id: Token ID for <EOS> (with offset).
    """

    def __init__(self, corpus_path: str | Path, vocab_size: int, offset: int) -> None:
        self.corpus_path = Path(corpus_path)
        self.vocab_size = vocab_size
        self.offset = offset

        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {self.corpus_path}")

        self.tokenizer = _train_pidgin_tokenizer(self.corpus_path, vocab_size)

        # Special token IDs (with offset applied)
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

    Args:
        pidgin_corpus_path: Path to the pidgin training corpus.
        pidgin_vocab_size: Number of tokens for Stream B (default: 1000).

    Attributes:
        main_vocab_size: Derived as GPT2_VOCAB_SIZE - pidgin_vocab_size.
        pidgin_vocab_size: Size of Stream B vocabulary.
        main: The MainTokenizer instance.
        pidgin: The PidginTokenizer instance.
    """

    def __init__(
        self,
        pidgin_corpus_path: str | Path = "words.txt",
        pidgin_vocab_size: int = DEFAULT_PIDGIN_VOCAB_SIZE,
    ) -> None:
        self.main_vocab_size = GPT2_VOCAB_SIZE - pidgin_vocab_size
        self.pidgin_vocab_size = pidgin_vocab_size

        self.main = MainTokenizer(self.main_vocab_size)
        self.pidgin = PidginTokenizer(
            pidgin_corpus_path,
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
    print(f"Stream A (main):   {main_size:,} tokens (IDs 0–{main_size - 1})")
    print(f"Stream B (pidgin): {pidgin_size:,} tokens (IDs {pidgin_offset}–{pidgin_offset + pidgin_size - 1})")
    print()

    # Sample pidgin vocabulary
    print("Sample pidgin vocabulary:")
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

    # Truncation test
    print("Truncation test (tokens that decompose):")
    gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2")
    rare = [
        (tok, tid) for tok, tid in gpt2_tok.encoder.items()
        if tid >= main_size and tid < GPT2_VOCAB_SIZE
        and tok.isprintable() and len(tok) > 2
    ][:5]
    for token, orig_id in rare:
        ids, tokens = dual_tokenizer.main.encode_with_tokens(token)
        print(f"  '{token}' (GPT-2 ID {orig_id}) -> {tokens}")
    print()

    # Stream B examples
    print("Stream B (Pidgin) examples:")
    for text in ["the big red thing", "understanding the world", "hello unknown123"]:
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

    print(f"  Main IDs:   {main_ids} (all < {main_size}) ✓")
    print(f"  Pidgin IDs: {pidgin_ids} (all >= {pidgin_offset}) ✓")
    print()
    print("Verification passed!")
    return True


def main() -> None:
    """CLI entry point."""
    import sys

    corpus_path = sys.argv[1] if len(sys.argv) > 1 else "words.txt"
    pidgin_size = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_PIDGIN_VOCAB_SIZE

    print(f"Corpus: {corpus_path}")
    print(f"Pidgin vocab size: {pidgin_size}")
    print()

    try:
        tokenizer = DualStreamTokenizer(corpus_path, pidgin_size)
        verify_tokenizers(tokenizer)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Usage: python tokenizer.py [corpus_path] [pidgin_vocab_size]")
        sys.exit(1)


if __name__ == "__main__":
    main()
