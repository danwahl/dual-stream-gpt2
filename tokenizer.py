"""
Dual-stream tokenizer creation using truncated BPE from GPT-2.

Creates two tokenizers:
- Main tokenizer: 49,000 tokens (256 bytes + 48,744 merges)
- Pidgin tokenizer: 1,000 tokens (256 bytes + 744 merges)

Both share the same base vocabulary and early merges, ensuring token IDs
0-999 mean the same thing in both tokenizers.
"""

from transformers import GPT2Tokenizer
from tokenizers import Tokenizer, models, pre_tokenizers, decoders


def extract_merges(model_name: str = "gpt2") -> list[tuple[str, str]]:
    """
    Extract the ordered merge list from a pretrained GPT-2 tokenizer.

    Args:
        model_name: HuggingFace model name (default: 'gpt2')

    Returns:
        List of merge tuples ordered by priority (most common first).
        Example: [('Ġ', 't'), ('Ġ', 'a'), ('h', 'e'), ...]
    """
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # bpe_ranks maps (token1, token2) -> merge_priority
    # Lower number = earlier merge = more common pattern
    merges = sorted(tokenizer.bpe_ranks.items(), key=lambda x: x[1])

    # Return just the merge pairs, in order
    return [merge_pair for merge_pair, rank in merges]


def create_truncated_bpe_tokenizer(
    merges: list[tuple[str, str]], vocab_size: int
) -> Tokenizer:
    """
    Create a BPE tokenizer using only the first N merges.

    Args:
        merges: Full list of merge tuples from GPT-2
        vocab_size: Target vocabulary size (256 + num_merges_to_use)

    Returns:
        A tokenizers.Tokenizer object
    """
    num_merges = vocab_size - 256  # Account for base byte vocabulary
    truncated_merges = merges[:num_merges]

    # Build vocabulary: bytes 0-255, then merged tokens
    vocab = {}
    for i in range(256):
        # GPT-2 uses a specific byte encoding scheme
        byte_char = bytes([i]).decode("latin-1")
        vocab[byte_char] = i

    # Add merged tokens
    for idx, (tok1, tok2) in enumerate(truncated_merges):
        merged = tok1 + tok2
        vocab[merged] = 256 + idx

    # Create tokenizer with truncated merges
    tokenizer = Tokenizer(
        models.BPE(vocab=vocab, merges=truncated_merges, fuse_unk=False)
    )

    # Match GPT-2's pre-tokenization (split on spaces, punctuation)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    return tokenizer


def create_dual_tokenizers(
    main_vocab_size: int = 49000, pidgin_vocab_size: int = 1000
) -> tuple[Tokenizer, Tokenizer]:
    """
    Create main and pidgin tokenizers from GPT-2's merge rules.

    Both tokenizers share the same base vocabulary (256 bytes) and
    early merges, but pidgin uses fewer merges = coarser chunking.

    Args:
        main_vocab_size: Vocabulary size for main stream (default: 49000)
        pidgin_vocab_size: Vocabulary size for pidgin stream (default: 1000)

    Returns:
        Tuple of (main_tokenizer, pidgin_tokenizer)
    """
    # Extract all merges from pretrained GPT-2
    all_merges = extract_merges("gpt2")

    main_tokenizer = create_truncated_bpe_tokenizer(all_merges, main_vocab_size)
    pidgin_tokenizer = create_truncated_bpe_tokenizer(all_merges, pidgin_vocab_size)

    return main_tokenizer, pidgin_tokenizer


def verify_tokenizers(
    main_tokenizer: Tokenizer, pidgin_tokenizer: Tokenizer
) -> bool:
    """
    Verify the tokenizers work correctly.

    Checks:
    - Encoding/decoding round-trips work
    - Pidgin produces >= as many tokens as main (coarser tokenization)
    - Shared tokens have the same IDs in both tokenizers

    Returns:
        True if all checks pass, raises AssertionError otherwise
    """
    test_strings = [
        "Hello world",
        "The quick brown fox",
        "Machine learning is fascinating",
        "12345",  # Numbers
        "cafe naive",  # Simple ASCII (avoiding accents for now)
        "The thermostat temperature is lower",
    ]

    print("Verifying tokenizers...")
    print(f"Main vocab size: {main_tokenizer.get_vocab_size()}")
    print(f"Pidgin vocab size: {pidgin_tokenizer.get_vocab_size()}")
    print()

    for s in test_strings:
        main_enc = main_tokenizer.encode(s)
        pidgin_enc = pidgin_tokenizer.encode(s)

        print(f"Text: {s!r}")
        print(f"  Main:   {len(main_enc.ids):3d} tokens - {main_enc.tokens}")
        print(f"  Pidgin: {len(pidgin_enc.ids):3d} tokens - {pidgin_enc.tokens}")

        # Verify: pidgin should produce >= as many tokens (coarser = more tokens)
        assert len(pidgin_enc.ids) >= len(main_enc.ids), (
            f"Pidgin should produce >= tokens than main, "
            f"got {len(pidgin_enc.ids)} < {len(main_enc.ids)}"
        )

        # Verify round-trip decoding works
        main_decoded = main_tokenizer.decode(main_enc.ids)
        pidgin_decoded = pidgin_tokenizer.decode(pidgin_enc.ids)
        assert main_decoded == s, f"Main round-trip failed: {main_decoded!r} != {s!r}"
        assert pidgin_decoded == s, f"Pidgin round-trip failed: {pidgin_decoded!r} != {s!r}"

        print()

    # Verify shared tokens have same IDs
    print("Verifying shared token IDs...")
    main_vocab = main_tokenizer.get_vocab()
    pidgin_vocab = pidgin_tokenizer.get_vocab()

    for token, pidgin_id in pidgin_vocab.items():
        if token in main_vocab:
            main_id = main_vocab[token]
            assert main_id == pidgin_id, (
                f"Token {token!r} has different IDs: "
                f"main={main_id}, pidgin={pidgin_id}"
            )

    print("All shared tokens have matching IDs!")
    print()
    print("Tokenizer verification passed!")
    return True


if __name__ == "__main__":
    # Create and verify tokenizers
    main_tok, pidgin_tok = create_dual_tokenizers()
    verify_tokenizers(main_tok, pidgin_tok)
