"""
Dual-stream tokenizers for the Dual-Stream GPT-2 experiment.

Stream A (Main): Truncated GPT-2 BPE tokenizer, tokens 0 to (GPT2_VOCAB - pidgin_size - 1)
Stream B (Pidgin): BPE tokenizer trained on words.txt, tokens offset by main vocab size

The vocab sizes are parameterized so the relative sizes can be adjusted for testing.
Default: 1000 pidgin tokens, leaving 49,257 main tokens.
"""

from pathlib import Path
from transformers import GPT2Tokenizer
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


# GPT-2's full vocabulary size
GPT2_VOCAB_SIZE = 50257

# Default pidgin vocab size (can be overridden)
DEFAULT_PIDGIN_VOCAB_SIZE = 1000


def _create_truncated_main_tokenizer(main_vocab_size: int) -> Tokenizer:
    """
    Create a truncated BPE tokenizer for Stream A using GPT-2's vocab.

    Uses only the first main_vocab_size tokens by truncating the merge list.
    Rare patterns that would produce higher token IDs naturally fall back
    to their component tokens.
    """
    # Load GPT-2 tokenizer to extract vocab and merges
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Get first main_vocab_size tokens from vocab
    vocab = {}
    for token, token_id in gpt2_tokenizer.encoder.items():
        if token_id < main_vocab_size:
            vocab[token] = token_id

    # Get merges in order, truncated to produce only tokens < main_vocab_size
    # We need (main_vocab_size - 256) merges since first 256 are byte tokens
    num_merges = main_vocab_size - 256
    sorted_merges = sorted(gpt2_tokenizer.bpe_ranks.items(), key=lambda x: x[1])
    truncated_merges = [pair for pair, rank in sorted_merges[:num_merges]]

    # Create tokenizer with truncated vocab and merges
    tokenizer = Tokenizer(
        models.BPE(vocab=vocab, merges=truncated_merges, fuse_unk=False)
    )

    # Match GPT-2's pre-tokenization
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    return tokenizer


def _train_pidgin_tokenizer(corpus_path: Path, vocab_size: int) -> Tokenizer:
    """
    Train a BPE tokenizer on the pidgin corpus (words.txt).

    Args:
        corpus_path: Path to the training corpus (words.txt)
        vocab_size: Target vocabulary size for pidgin tokenizer

    Returns:
        A trained BPE tokenizer (token IDs 0 to vocab_size-1, not yet offset)
    """
    # Create blank BPE tokenizer
    tokenizer = Tokenizer(models.BPE())

    # Use byte-level pre-tokenization like GPT-2
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # Train on the corpus
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=1,  # Include all patterns from the limited corpus
        special_tokens=["<PAD>", "<UNK>", "<EOS>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    tokenizer.train([str(corpus_path)], trainer)

    return tokenizer


class MainTokenizer:
    """
    Truncated BPE tokenizer for Stream A.

    Uses GPT-2's vocabulary but only the first N tokens (where N = GPT2_VOCAB - pidgin_size).
    Rare patterns that would use higher token IDs naturally decompose
    into component tokens via the truncated merge list.
    """

    def __init__(self, vocab_size: int):
        self.tokenizer = _create_truncated_main_tokenizer(vocab_size)
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs (all guaranteed to be < vocab_size)."""
        encoding = self.tokenizer.encode(text)
        return encoding.ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text."""
        return self.tokenizer.decode(ids)


class PidginTokenizer:
    """
    BPE tokenizer for Stream B, trained on a pidgin corpus (words.txt).

    Token IDs are offset by main_vocab_size so they don't overlap with Stream A.
    For example, with main_vocab_size=49257 and pidgin_vocab_size=1000:
    - Pidgin token 0 becomes 49257
    - Pidgin token 999 becomes 50256
    """

    def __init__(self, corpus_path: str | Path, vocab_size: int, offset: int):
        self.corpus_path = Path(corpus_path)
        self.vocab_size = vocab_size
        self.offset = offset

        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {self.corpus_path}")

        # Train BPE on the corpus
        self.tokenizer = _train_pidgin_tokenizer(self.corpus_path, vocab_size)

        # Store special token IDs (with offset)
        self.pad_token_id = self.offset + 0  # <PAD> is token 0
        self.unk_token_id = self.offset + 1  # <UNK> is token 1
        self.eos_token_id = self.offset + 2  # <EOS> is token 2

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs (offset by main vocab size)."""
        encoding = self.tokenizer.encode(text)
        # Apply offset to all token IDs
        return [id + self.offset for id in encoding.ids]

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text (removing offset first)."""
        # Remove offset before decoding
        local_ids = [id - self.offset for id in ids]
        return self.tokenizer.decode(local_ids)

    def get_vocab(self) -> dict[str, int]:
        """Get vocabulary with offset applied to IDs."""
        return {token: id + self.offset
                for token, id in self.tokenizer.get_vocab().items()}


class DualStreamTokenizer:
    """
    Combined tokenizer for both streams.

    Vocab sizes are parameterized:
    - pidgin_vocab_size: Size of Stream B vocabulary (default: 1000)
    - main_vocab_size: Derived as GPT2_VOCAB_SIZE - pidgin_vocab_size

    This ensures the two vocabularies partition the full GPT-2 embedding space.
    """

    def __init__(
        self,
        pidgin_corpus_path: str | Path = "words.txt",
        pidgin_vocab_size: int = DEFAULT_PIDGIN_VOCAB_SIZE
    ):
        # Derive main vocab size from GPT-2 total minus pidgin allocation
        self.main_vocab_size = GPT2_VOCAB_SIZE - pidgin_vocab_size
        self.pidgin_vocab_size = pidgin_vocab_size

        # Create tokenizers
        self.main = MainTokenizer(self.main_vocab_size)
        self.pidgin = PidginTokenizer(
            pidgin_corpus_path,
            pidgin_vocab_size,
            offset=self.main_vocab_size
        )

    def encode_main(self, text: str) -> list[int]:
        """Encode text for Stream A (main)."""
        return self.main.encode(text)

    def encode_pidgin(self, text: str) -> list[int]:
        """Encode text for Stream B (pidgin)."""
        return self.pidgin.encode(text)

    def decode_main(self, ids: list[int]) -> str:
        """Decode Stream A token IDs."""
        return self.main.decode(ids)

    def decode_pidgin(self, ids: list[int]) -> str:
        """Decode Stream B token IDs."""
        return self.pidgin.decode(ids)


def verify_tokenizers(dual_tokenizer: DualStreamTokenizer) -> bool:
    """
    Verify the tokenizers work correctly.
    """
    main_vocab_size = dual_tokenizer.main_vocab_size
    pidgin_vocab_size = dual_tokenizer.pidgin_vocab_size
    pidgin_offset = main_vocab_size

    print("=" * 60)
    print("Dual-Stream Tokenizer Verification")
    print("=" * 60)
    print()

    print(f"GPT-2 total vocab: {GPT2_VOCAB_SIZE:,}")
    print(f"Main vocab size (Stream A): {main_vocab_size:,} (tokens 0-{main_vocab_size-1})")
    print(f"Pidgin vocab size (Stream B): {pidgin_vocab_size:,} (tokens {pidgin_offset}-{pidgin_offset + pidgin_vocab_size - 1})")
    print()

    # Show sample tokens from pidgin vocabulary
    print("Sample pidgin vocabulary tokens:")
    pidgin_vocab = dual_tokenizer.pidgin.get_vocab()
    # Sort by token ID and show first 20 (skipping special tokens)
    sorted_tokens = sorted(pidgin_vocab.items(), key=lambda x: x[1])
    shown = 0
    for token, token_id in sorted_tokens:
        if shown >= 20:
            break
        print(f"  {token!r} -> {token_id}")
        shown += 1
    print(f"  ... ({len(pidgin_vocab)} total tokens)")
    print()

    # Test main tokenizer
    print("Stream A (Main) examples:")
    main_tests = [
        "Hello world",
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is fascinating",
    ]
    for text in main_tests:
        ids = dual_tokenizer.encode_main(text)
        decoded = dual_tokenizer.decode_main(ids)
        print(f"  '{text}'")
        print(f"    -> {len(ids)} tokens: {ids[:10]}{'...' if len(ids) > 10 else ''}")
        print(f"    -> decoded: '{decoded}'")
    print()

    # Test truncated tokens: these would have been >= main_vocab_size in original GPT-2
    print("Truncated token test (rare patterns that decompose):")
    gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2")
    rare_tokens = []
    for token, token_id in gpt2_tok.encoder.items():
        if token_id >= main_vocab_size and token_id < GPT2_VOCAB_SIZE:
            if token.isprintable() and len(token) > 2:
                rare_tokens.append((token, token_id))
        if len(rare_tokens) >= 5:
            break

    for token, orig_id in rare_tokens:
        our_ids = dual_tokenizer.encode_main(token)
        print(f"  '{token}' (original GPT-2 ID: {orig_id})")
        print(f"    -> our tokenizer: {our_ids} (decomposed into {len(our_ids)} tokens)")
    print()

    # Test pidgin tokenizer
    print("Stream B (Pidgin) examples:")
    pidgin_tests = [
        "the big red thing",
        "understanding the world",
        "hello xyzunknownpattern123",
    ]
    for text in pidgin_tests:
        ids = dual_tokenizer.encode_pidgin(text)
        decoded = dual_tokenizer.decode_pidgin(ids)
        print(f"  '{text}'")
        print(f"    -> {len(ids)} tokens: {ids}")
        print(f"    -> decoded: '{decoded}'")
    print()

    # Verify token ID ranges don't overlap
    print("Verifying token ID ranges...")
    main_ids = dual_tokenizer.encode_main("The quick brown fox")
    pidgin_ids = dual_tokenizer.encode_pidgin("the big thing")

    assert all(id < main_vocab_size for id in main_ids), \
        f"Main tokens should be < {main_vocab_size}"
    assert all(id >= pidgin_offset for id in pidgin_ids), \
        f"Pidgin tokens should be >= {pidgin_offset}"

    print(f"  Main token IDs: {main_ids} (all < {main_vocab_size})")
    print(f"  Pidgin token IDs: {pidgin_ids} (all >= {pidgin_offset})")
    print()

    print("Tokenizer verification passed!")
    return True


if __name__ == "__main__":
    import sys

    # Parse arguments
    corpus_path = "words.txt"
    pidgin_size = DEFAULT_PIDGIN_VOCAB_SIZE

    if len(sys.argv) > 1:
        corpus_path = sys.argv[1]
    if len(sys.argv) > 2:
        pidgin_size = int(sys.argv[2])

    print(f"Using corpus: {corpus_path}")
    print(f"Pidgin vocab size: {pidgin_size}")
    print()

    try:
        dual_tok = DualStreamTokenizer(corpus_path, pidgin_size)
        verify_tokenizers(dual_tok)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Usage: python tokenizer.py [words.txt] [pidgin_vocab_size]")
        sys.exit(1)
