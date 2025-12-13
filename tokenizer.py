"""
Dual-stream tokenizers for the Dual-Stream GPT-2 experiment.

Stream A (Main): Standard GPT-2 tokenizer, tokens 0-49,256
Stream B (Pidgin): Word-level "Thing Explainer" tokenizer, tokens 49,257-50,256

The pidgin vocabulary is loaded from a words.txt file containing ~1000 common
English words. Words are normalized (lowercase, stripped punctuation) before lookup.
"""

import re
from pathlib import Path
from transformers import GPT2Tokenizer
from tokenizers import Tokenizer, models, pre_tokenizers, decoders


# Token ID boundaries
MAIN_VOCAB_SIZE = 49257  # Main vocab is 0 to 49,256 (inclusive)
PIDGIN_OFFSET = 49257    # Pidgin tokens start at 49,257
PIDGIN_VOCAB_SIZE = 1000 # 1000 pidgin tokens (49,257 to 50,256)

# Special token indices within pidgin vocab (before offset)
PIDGIN_PAD_IDX = 0
PIDGIN_UNK_IDX = 1
PIDGIN_EOS_IDX = 2
PIDGIN_RESERVED = 3  # First 3 slots reserved, words start at index 3

# Actual token IDs (with offset applied)
PIDGIN_PAD_ID = PIDGIN_OFFSET + PIDGIN_PAD_IDX  # 49,257
PIDGIN_UNK_ID = PIDGIN_OFFSET + PIDGIN_UNK_IDX  # 49,258
PIDGIN_EOS_ID = PIDGIN_OFFSET + PIDGIN_EOS_IDX  # 49,259


def _create_truncated_main_tokenizer() -> Tokenizer:
    """
    Create a truncated BPE tokenizer for Stream A using GPT-2's vocab.

    Uses only the first 49,257 tokens (0-49,256) by truncating the merge list.
    Rare patterns that would produce higher token IDs naturally fall back
    to their component tokens.
    """
    # Load GPT-2 tokenizer to extract vocab and merges
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Get first MAIN_VOCAB_SIZE tokens from vocab
    vocab = {}
    for token, token_id in gpt2_tokenizer.encoder.items():
        if token_id < MAIN_VOCAB_SIZE:
            vocab[token] = token_id

    # Get merges in order, truncated to produce only tokens < MAIN_VOCAB_SIZE
    # We need (MAIN_VOCAB_SIZE - 256) merges since first 256 are byte tokens
    num_merges = MAIN_VOCAB_SIZE - 256
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


class MainTokenizer:
    """
    Truncated BPE tokenizer for Stream A.

    Uses GPT-2's vocabulary but only the first 49,257 tokens (0-49,256).
    Rare patterns that would use higher token IDs naturally decompose
    into component tokens via the truncated merge list.
    """

    def __init__(self):
        self.tokenizer = _create_truncated_main_tokenizer()
        self.vocab_size = MAIN_VOCAB_SIZE

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs (all guaranteed to be < 49,257)."""
        encoding = self.tokenizer.encode(text)
        return encoding.ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text."""
        return self.tokenizer.decode(ids)

    @property
    def eos_token_id(self) -> int:
        # GPT-2's EOS is 50256, which is outside our range
        # We don't use a special EOS for main stream in this design
        return None


class PidginTokenizer:
    """
    Word-level tokenizer for Stream B (Thing Explainer vocabulary).

    Loads vocabulary from a words.txt file. Words are normalized
    (lowercase, punctuation stripped) before lookup. Unknown words
    map to <UNK>.

    Token IDs are offset by 49,257 so they occupy the range 49,257-50,256.
    """

    def __init__(self, vocab_path: str | Path):
        self.vocab_path = Path(vocab_path)
        self.word_to_id: dict[str, int] = {}
        self.id_to_word: dict[int, str] = {}

        # Reserve special tokens
        self._add_token("<PAD>", PIDGIN_PAD_IDX)
        self._add_token("<UNK>", PIDGIN_UNK_IDX)
        self._add_token("<EOS>", PIDGIN_EOS_IDX)

        # Load vocabulary from file
        self._load_vocab()

    def _add_token(self, word: str, local_idx: int):
        """Add a token with a local index (before offset)."""
        token_id = PIDGIN_OFFSET + local_idx
        self.word_to_id[word] = token_id
        self.id_to_word[token_id] = word

    def _load_vocab(self):
        """Load vocabulary from words.txt file."""
        if not self.vocab_path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {self.vocab_path}")

        with open(self.vocab_path) as f:
            words = [line.strip().lower() for line in f if line.strip()]

        # Add words starting after reserved tokens
        max_words = PIDGIN_VOCAB_SIZE - PIDGIN_RESERVED  # 997 words max
        local_idx = PIDGIN_RESERVED
        for word in words:
            if local_idx >= PIDGIN_VOCAB_SIZE:
                break  # Vocab full
            if word and word not in self.word_to_id:  # Skip empty/duplicates
                self._add_token(word, local_idx)
                local_idx += 1

        self.vocab_size = len(self.word_to_id)

    @staticmethod
    def normalize(text: str) -> list[str]:
        """Normalize text: lowercase and split into words."""
        text = text.lower()
        # Split on whitespace and punctuation, keep only alphanumeric
        words = re.findall(r'[a-z0-9]+', text)
        return words

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        words = self.normalize(text)
        ids = []
        for word in words:
            if word in self.word_to_id:
                ids.append(self.word_to_id[word])
            else:
                ids.append(PIDGIN_UNK_ID)
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text."""
        words = []
        for id in ids:
            if id in self.id_to_word:
                words.append(self.id_to_word[id])
            else:
                words.append("<UNK>")
        return " ".join(words)

    @property
    def pad_token_id(self) -> int:
        return PIDGIN_PAD_ID

    @property
    def unk_token_id(self) -> int:
        return PIDGIN_UNK_ID

    @property
    def eos_token_id(self) -> int:
        return PIDGIN_EOS_ID


class DualStreamTokenizer:
    """
    Combined tokenizer for both streams.

    Provides a unified interface for encoding/decoding both streams.
    """

    def __init__(self, pidgin_vocab_path: str | Path = "words.txt"):
        self.main = MainTokenizer()
        self.pidgin = PidginTokenizer(pidgin_vocab_path)

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
    print("=" * 60)
    print("Dual-Stream Tokenizer Verification")
    print("=" * 60)
    print()

    print(f"Main tokenizer vocab size: {dual_tokenizer.main.vocab_size:,}")
    print(f"Pidgin tokenizer vocab size: {dual_tokenizer.pidgin.vocab_size}")
    print(f"Pidgin token ID range: {PIDGIN_OFFSET} - {PIDGIN_OFFSET + PIDGIN_VOCAB_SIZE - 1}")
    print()

    # Show sample words from pidgin vocabulary
    print("Sample pidgin vocabulary words:")
    pidgin_words = [w for w in dual_tokenizer.pidgin.word_to_id.keys()
                    if not w.startswith("<")]  # Skip special tokens
    for word in pidgin_words[:20]:
        token_id = dual_tokenizer.pidgin.word_to_id[word]
        print(f"  '{word}' -> {token_id}")
    print(f"  ... ({len(pidgin_words)} total words)")
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

    # Test truncated tokens: these would have been >=49,257 in original GPT-2
    # With truncated merges, they should decompose into component tokens
    print("Truncated token test (rare patterns that decompose):")
    gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2")
    # Find some tokens that are >= 49,257 in original GPT-2
    rare_tokens = []
    for token, token_id in gpt2_tok.encoder.items():
        if token_id >= MAIN_VOCAB_SIZE and token_id < 50000:
            # Skip byte sequences, get readable tokens
            if token.isprintable() and len(token) > 2:
                rare_tokens.append((token, token_id))
        if len(rare_tokens) >= 5:
            break

    for token, orig_id in rare_tokens:
        # Encode with our truncated tokenizer
        our_ids = dual_tokenizer.encode_main(token)
        print(f"  '{token}' (original GPT-2 ID: {orig_id})")
        print(f"    -> our tokenizer: {our_ids} (decomposed into {len(our_ids)} tokens)")
    print()

    # Test pidgin tokenizer using actual vocabulary words
    print("Stream B (Pidgin) examples:")
    # Use first few actual words from vocabulary
    test_words = pidgin_words[:5] if pidgin_words else ["test"]
    test_text = " ".join(test_words)
    ids = dual_tokenizer.encode_pidgin(test_text)
    decoded = dual_tokenizer.decode_pidgin(ids)
    print(f"  '{test_text}'")
    print(f"    -> {len(ids)} tokens: {ids}")
    print(f"    -> decoded: '{decoded}'")

    # Test with unknown word
    unknown_test = f"{test_words[0] if test_words else 'test'} xyzunknownword123"
    ids = dual_tokenizer.encode_pidgin(unknown_test)
    decoded = dual_tokenizer.decode_pidgin(ids)
    print(f"  '{unknown_test}'")
    print(f"    -> {len(ids)} tokens: {ids}")
    print(f"    -> decoded: '{decoded}'")
    print()

    # Verify token ID ranges don't overlap
    print("Verifying token ID ranges...")
    main_ids = dual_tokenizer.encode_main("The quick brown fox")
    pidgin_ids = dual_tokenizer.encode_pidgin(test_text)

    assert all(id < MAIN_VOCAB_SIZE for id in main_ids), "Main tokens should be < 49,257"
    assert all(id >= PIDGIN_OFFSET for id in pidgin_ids), "Pidgin tokens should be >= 49,257"

    print(f"  Main token IDs: {main_ids} (all < {MAIN_VOCAB_SIZE})")
    print(f"  Pidgin token IDs: {pidgin_ids} (all >= {PIDGIN_OFFSET})")
    print()

    print("Tokenizer verification passed!")
    return True


if __name__ == "__main__":
    import sys

    # Default vocab path, can be overridden via command line
    vocab_path = sys.argv[1] if len(sys.argv) > 1 else "words.txt"

    try:
        dual_tok = DualStreamTokenizer(vocab_path)
        verify_tokenizers(dual_tok)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please provide a words.txt file with the Thing Explainer vocabulary.")
        sys.exit(1)
