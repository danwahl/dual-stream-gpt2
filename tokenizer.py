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


def create_truncated_bpe_tokenizer(
    gpt2_tokenizer: GPT2Tokenizer, vocab_size: int
) -> Tokenizer:
    """
    Create a BPE tokenizer by truncating GPT-2's vocab and merges.

    Args:
        gpt2_tokenizer: The pretrained GPT-2 tokenizer to extract from
        vocab_size: Target vocabulary size (256 base + N merges)

    Returns:
        A tokenizers.Tokenizer object with truncated vocab
    """
    # GPT-2's vocab is already ordered: 0-255 are byte tokens, 256+ are merges
    # Just take the first vocab_size tokens
    vocab = {}
    for token, token_id in gpt2_tokenizer.encoder.items():
        if token_id < vocab_size:
            vocab[token] = token_id

    # Get merges in order (bpe_ranks maps tuple -> priority)
    # We need (vocab_size - 256) merges
    num_merges = vocab_size - 256
    sorted_merges = sorted(gpt2_tokenizer.bpe_ranks.items(), key=lambda x: x[1])
    truncated_merges = [pair for pair, rank in sorted_merges[:num_merges]]

    # Create tokenizer with truncated vocab and merges
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
    Create main and pidgin tokenizers from GPT-2's vocab and merge rules.

    Both tokenizers share the same base vocabulary (256 bytes) and
    early merges, but pidgin uses fewer merges = coarser chunking.

    Args:
        main_vocab_size: Vocabulary size for main stream (default: 49000)
        pidgin_vocab_size: Vocabulary size for pidgin stream (default: 1000)

    Returns:
        Tuple of (main_tokenizer, pidgin_tokenizer)
    """
    # Load pretrained GPT-2 tokenizer once
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    main_tokenizer = create_truncated_bpe_tokenizer(gpt2_tokenizer, main_vocab_size)
    pidgin_tokenizer = create_truncated_bpe_tokenizer(gpt2_tokenizer, pidgin_vocab_size)

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
