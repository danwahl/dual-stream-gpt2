# Dual-Stream GPT-2 Experiment

## Project Goal

Train a GPT-2 model to process two text streams simultaneously:
- **Stream A (Main)**: Normal text, tokens 0–49,256
- **Stream B (Pidgin)**: Secondary text, "Thing Explainer" vocabulary, tokens 49,257–50,256

The two streams are summed at the embedding level. The model learns to predict both next tokens through a single output head, with logits split before softmax so each stream has an independent probability distribution.

---

## Architecture Overview

```
Input:  token_A ∈ [0, 49256]         (main vocabulary)
        token_B ∈ [49257, 50256]     (pidgin vocabulary, 1000 tokens)
                    ↓
Embed:  combined = embed[token_A] + embed[token_B]
        (single 50,257 × 768 matrix; pidgin region re-initialized)
                    ↓
Transformer:  standard GPT-2 layers [trainable]
                    ↓
Output: logits = lm_head(hidden)     (single head, 50,257 logits)
                    ↓
Split:  logits_A = logits[..., :49257]
        logits_B = logits[..., 49257:]
                    ↓
Loss:   CE(logits_A, target_A) + CE(logits_B, target_B - 49257)
        (separate softmax per stream via CrossEntropyLoss)
```

---

## Key Design Decisions

### Single Embedding Matrix, Different Regions

The standard GPT-2 embedding matrix (50,257 × 768) is partitioned:
- Rows 0–49,256: Main vocabulary (initialized from pretrained GPT-2)
- Rows 49,257–50,256: Pidgin vocabulary (re-initialized, will learn new meanings)

At each position, we look up both token embeddings and sum them:
```
combined = embed[token_A] + embed[token_B]
```

The pidgin region is re-initialized because its original tokens (rare Unicode, long merges) have meanings we want to overwrite with our Thing Explainer vocabulary.

### Why Re-initialize Pidgin Embeddings?

The last 1000 tokens in GPT-2's vocabulary encode rare patterns. If we reuse their embeddings, the model starts with arbitrary (and possibly interfering) representations. Re-initializing gives the model a clean slate to learn pidgin meanings.

### Split Logits Before Softmax

A single softmax over 50,257 tokens forces probabilities to sum to 1 across *both* streams. This creates competition: high confidence in stream A leaves little probability mass for stream B.

Instead, we split logits and compute loss separately:
```python
logits_A = logits[..., :49257]
logits_B = logits[..., 49257:]

loss_A = CrossEntropyLoss(logits_A, target_A)
loss_B = CrossEntropyLoss(logits_B, target_B - 49257)  # shift to [0, 999]
```

CrossEntropyLoss applies log_softmax internally, so each stream gets an independent probability distribution. The model can be 95% confident in both predictions simultaneously.

### Why "Thing Explainer" Vocabulary?

The pidgin stream uses a curated vocabulary of ~1000 common English words rather than GPT-2's rare tokens. This provides:
- Semantically meaningful tokens (actual words, not subword fragments)
- Case-insensitivity (less redundancy)
- Interpretable intermediate reasoning
- Natural fit for the "internal thought" use case

The vocabulary should include:
- Top ~750 common English words
- Concept words: "more", "less", "same", "if", "then", "because"
- Numbers: "zero" through "ten", "hundred"
- Reserved: `<PAD>`, `<UNK>`, `<EOS>`

---

## Tokenizer Implementation

### Stream A (Main)

Use standard GPT-2 tokenizer, but only tokens 0–49,256. Tokens ≥49,257 are reserved for pidgin.

In practice, GPT-2's tokens 49,257–50,256 are rare enough that most text won't use them. If they do appear, either:
- Map to `<UNK>`
- Fall back to component tokens (BPE naturally handles this)

### Stream B (Pidgin / Thing Explainer)

Build a word-level tokenizer:
1. Curate vocabulary: top 1000 common English words
2. Normalize input: lowercase, strip punctuation
3. Map words to token IDs 49,257–50,256
4. Unknown words → `<UNK>` token

The mapping from word to token ID is simply:
```
token_id = word_index + 49257
```

---

## Model Implementation

### Initialization

1. Load pretrained GPT-2 (embeddings, transformer, lm_head)
2. Re-initialize embedding rows 49,257–50,256 (pidgin region)
3. Optionally re-initialize corresponding lm_head weights for pidgin region
4. All parameters trainable

### Forward Pass

1. Look up `embed[token_A]` and `embed[token_B]` from same matrix
2. Sum: `combined = embed_A + embed_B`
3. Add position embeddings
4. Pass through transformer blocks
5. Project via lm_head → 50,257 logits
6. Split logits at index 49,257
7. Compute CE loss for each stream separately, sum

### Special Tokens

- `<PAD>`: Use `ignore_index` in CrossEntropyLoss
- `<EOS>`: Include in both vocabularies (different token IDs)
- Attention mask: Mask pad positions

---

## Training

### Hardware Constraints (GTX 1070, 8GB)
- GPT-2 Small (124M params)
- Sequence length: 256
- Batch size: 4 with gradient accumulation
- Mixed precision (fp16)

### Data
- Corpus: WikiText-103, OpenWebText, or similar
- Each example: two unrelated text chunks
- Later experiments: related streams (summary, reasoning steps)

### Loss
```
loss = weight_A * CE(logits_A, target_A) + weight_B * CE(logits_B, target_B - 49257)
```

Start with equal weights; adjust if one stream dominates.

---

## Evaluation

### Primary Metrics

1. **Stream separation**: Both streams predict accurately; cross-prediction should fail
2. **Main stream quality**: Perplexity compared to baseline GPT-2
3. **Pidgin stream quality**: Perplexity and qualitative coherence

### Baselines
- Original GPT-2 (no fine-tuning)
- GPT-2 fine-tuned single-stream on same data
- GPT-2 with random noise added to embeddings (ablation)

### Scenarios
1. Dual-stream: measure both accuracies
2. Main only (zero pidgin embeddings): does removing B hurt A?
3. Cross-prediction: verify streams are separated

---

## Experimental Progression

1. **Sanity check**: Two unrelated streams. Goal: both streams beat random chance.
2. **Interference measurement**: Compare main perplexity with/without pidgin.
3. **Related streams**: Stream B is simplified version of A. Does it help?
4. **Reasoning task**: Stream A is problem + answer, Stream B is reasoning steps.

---

## Success Criteria

| Level | Criteria |
|-------|----------|
| Minimum | Both streams > random accuracy; main perplexity < 2x baseline |
| Good | Both streams > 60% accuracy; main perplexity < 1.2x baseline; clear separation |
| Exciting | Related/reasoning stream improves main stream performance |

---

## References

- Coconut (continuous latent reasoning): arxiv.org/abs/2412.06769
- Quiet-STaR (internal rationales): arxiv.org/abs/2403.09629
- Multi-token prediction: arxiv.org/abs/2404.19737
