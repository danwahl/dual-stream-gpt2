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


# Token ID boundaries
MAIN_VOCAB_END = 49257  # Main vocab is 0 to 49,256 (inclusive)
PIDGIN_OFFSET = 49257   # Pidgin tokens start at 49,257
PIDGIN_VOCAB_SIZE = 1000  # 1000 pidgin tokens (49,257 to 50,256)

# Special token indices within pidgin vocab (before offset)
PIDGIN_PAD_IDX = 0
PIDGIN_UNK_IDX = 1
PIDGIN_EOS_IDX = 2
PIDGIN_RESERVED = 3  # First 3 slots reserved, words start at index 3

# Actual token IDs (with offset applied)
PIDGIN_PAD_ID = PIDGIN_OFFSET + PIDGIN_PAD_IDX  # 49,257
PIDGIN_UNK_ID = PIDGIN_OFFSET + PIDGIN_UNK_IDX  # 49,258
PIDGIN_EOS_ID = PIDGIN_OFFSET + PIDGIN_EOS_IDX  # 49,259


class MainTokenizer:
    """
    Wrapper around GPT-2 tokenizer for Stream A.

    Tokens 0-49,256 are valid. If GPT-2 produces tokens >= 49,257,
    they are mapped to a fallback (though this is rare in practice).
    """

    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.vocab_size = MAIN_VOCAB_END  # 49,257 tokens (0 to 49,256)

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs, clamping any out-of-range tokens."""
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        # Clamp any tokens >= MAIN_VOCAB_END (rare, but possible)
        return [min(id, MAIN_VOCAB_END - 1) for id in ids]

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text."""
        return self.tokenizer.decode(ids)

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id  # 50256, but we may not use it


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
        for i, word in enumerate(words[:max_words]):
            if word not in self.word_to_id:  # Skip duplicates
                local_idx = PIDGIN_RESERVED + i
                self._add_token(word, local_idx)

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

    # Test pidgin tokenizer
    print("Stream B (Pidgin) examples:")
    pidgin_tests = [
        "the big red thing",
        "water fire earth air",
        "hello world unknown_word_xyz",
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

    assert all(id < MAIN_VOCAB_END for id in main_ids), "Main tokens should be < 49,257"
    assert all(id >= PIDGIN_OFFSET for id in pidgin_ids), "Pidgin tokens should be >= 49,257"

    print(f"  Main token IDs: {main_ids} (all < {MAIN_VOCAB_END})")
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
