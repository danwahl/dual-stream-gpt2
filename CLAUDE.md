# Dual-Stream GPT-2 Experiment

## Project Goal

Train a GPT-2 model to process two text streams simultaneously:
- **Stream A (Main)**: Normal text, full vocabulary (~49,000 tokens)
- **Stream B (Pidgin)**: Secondary text, constrained vocabulary (~1,000 tokens)

The two streams are summed at the embedding level. The model learns to predict both next tokens through separate output heads. This tests whether transformers can learn to "unmix" superposed signals and maintain parallel reasoning threads.

---

## Architecture Overview

```
Input:  Stream A tokens [a1, a2, a3, ...]    (vocab 0-48999)
        Stream B tokens [b1, b2, b3, ...]    (vocab 0-999)
                    ↓
Embed:  embed_A = main_embeddings(tokens_A)      [frozen]
        embed_B = pidgin_embeddings(tokens_B)    [trainable]
                    ↓
Sum:    combined = embed_A + embed_B
                    ↓
Transformer:  GPT-2 layers (trainable, or LoRA)
                    ↓
Output: head_A → logits over 49,000 tokens       [frozen or light tune]
        head_B → logits over 1,000 tokens        [trainable]
                    ↓
Loss:   loss = CE(logits_A, target_A) + CE(logits_B, target_B)
```

---

## Phase 1: Environment Setup

### Hardware Target
- GTX 1070 8GB VRAM
- GPT-2 Small (124M params) fits comfortably
- Batch size will be limited (~4-8 sequences)
- Use gradient accumulation for effective larger batches

### Dependencies
- Python 3.8+

```
torch>=2.0
transformers>=4.30
tokenizers>=0.13
datasets
wandb (optional, for logging)
numpy
tqdm
```

### Directory Structure
```
dual-stream-gpt2/
├── CLAUDE.md                 # This file
├── config.py                 # Hyperparameters and settings
├── tokenizer.py              # Dual tokenizer creation (truncated BPE)
├── model.py                  # DualStreamGPT2 model class
├── data.py                   # Dataset and data loading
├── train.py                  # Training loop
├── evaluate.py               # Evaluation and comparison
├── utils.py                  # Helper functions
└── experiments/
    └── run_001/              # Checkpoints and logs
```

---

## Phase 2: Model Implementation

### File: `model.py`

**Key Components:**

1. **DualStreamGPT2 class**
   - Wraps HuggingFace GPT2LMHeadModel
   - Adds separate pidgin embedding layer (nn.Embedding, 1000 × 768)
   - Adds pidgin prediction head (nn.Linear, 768 → 1000)
   - Forward pass sums embeddings, returns both logit sets

2. **Initialization**
   - Load pretrained GPT-2 small
   - Freeze main embeddings: `model.transformer.wte.weight.requires_grad = False`
   - Initialize pidgin embeddings: normal distribution, std=0.02
   - Initialize pidgin head: standard linear init

3. **Forward signature**
   ```python
   def forward(self, tokens_A, tokens_B, labels_A=None, labels_B=None):
       # Returns: logits_A, logits_B, loss (if labels provided)
   ```

**Design Decisions:**
- Keep main token embeddings frozen (preserve language capability)
- Train: pidgin embeddings, pidgin head, transformer layers
- Option to use LoRA instead of full transformer fine-tuning (saves memory)

---

## Phase 3: Data Pipeline

### File: `data.py`

### Understanding BPE Tokenization (Background)

GPT-2 uses Byte Pair Encoding (BPE), which works as follows:

1. **Base vocabulary**: All 256 possible bytes (token IDs 0-255)
2. **Merges**: Iteratively learned pairs that get combined into new tokens
3. **Vocabulary size**: 256 (bytes) + num_merges = total tokens

GPT-2 has ~50,000 merges, giving ~50,257 total tokens. The merges are stored in order of frequency—earlier merges represent more common patterns.

**Tokenization algorithm:**
```
Input: "lower"
Start: ['l', 'o', 'w', 'e', 'r']  (raw bytes)

Apply merges in order:
  Merge #47 (e+r → er):     ['l', 'o', 'w', 'er']
  Merge #102 (l+o → lo):    ['lo', 'w', 'er']  
  Merge #315 (lo+w → low):  ['low', 'er']
  Merge #2847 (low+er → lower): ['lower']

Output: [token_id_for_'lower']
```

The key insight: **by truncating the merge list, we get a coarser tokenizer** that compresses less. Text becomes more tokens, but uses a smaller vocabulary.

### Tokenizer Setup

**Vocabulary Split:**
```
Main tokenizer:   256 bytes + 48,744 merges = 49,000 tokens (IDs 0-48,999)
Pidgin tokenizer: 256 bytes + 744 merges    = 1,000 tokens  (IDs 0-999)
```

The pidgin tokenizer uses only the first 744 merges—the most common patterns like `'th'`, `'er'`, `'Ġthe'`, etc. Rare words get chunked more finely.

### File: `tokenizer.py`

**Step 1: Extract GPT-2's merge rules**
```python
from transformers import GPT2Tokenizer
import json

def extract_merges(model_name='gpt2'):
    """Extract the ordered merge list from a pretrained tokenizer."""
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    # bpe_ranks maps (token1, token2) -> merge_priority
    # Lower number = earlier merge = more common pattern
    merges = sorted(tokenizer.bpe_ranks.items(), key=lambda x: x[1])
    
    # Returns list of tuples: [('e', 'r'), ('Ġ', 't'), ('th', 'e'), ...]
    return [merge_pair for merge_pair, rank in merges]
```

**Step 2: Create truncated tokenizers**
```python
from tokenizers import Tokenizer, models, pre_tokenizers, decoders

def create_truncated_bpe_tokenizer(merges, vocab_size):
    """
    Create a BPE tokenizer using only the first N merges.
    
    Args:
        merges: Full list of merge tuples from GPT-2
        vocab_size: Target vocabulary size (256 + num_merges_to_use)
    
    Returns:
        A tokenizer object
    """
    num_merges = vocab_size - 256  # Account for base byte vocabulary
    truncated_merges = merges[:num_merges]
    
    # Build vocabulary: bytes 0-255, then merged tokens
    vocab = {bytes([i]).decode('latin-1'): i for i in range(256)}
    
    for idx, (tok1, tok2) in enumerate(truncated_merges):
        merged = tok1 + tok2
        vocab[merged] = 256 + idx
    
    # Create tokenizer with truncated merges
    tokenizer = Tokenizer(models.BPE(
        vocab=vocab,
        merges=truncated_merges,
    ))
    
    # Match GPT-2's pre-tokenization (split on spaces, punctuation)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    return tokenizer

# Create both tokenizers
all_merges = extract_merges('gpt2')

main_tokenizer = create_truncated_bpe_tokenizer(all_merges, vocab_size=49000)
pidgin_tokenizer = create_truncated_bpe_tokenizer(all_merges, vocab_size=1000)
```

**Step 3: Tokenization behavior comparison**
```python
text = "The thermostat temperature is lower"

main_tokens = main_tokenizer.encode(text)
# Might produce: ['The', 'Ġthermo', 'stat', 'Ġtemperature', 'Ġis', 'Ġlower']
# → 6 tokens

pidgin_tokens = pidgin_tokenizer.encode(text)  
# Might produce: ['The', 'Ġth', 'er', 'mo', 'st', 'at', 'Ġtem', 'per', 'at', 'ure', 'Ġis', 'Ġlow', 'er']
# → 13 tokens (coarser merging = more tokens)
```

**Key property**: Both tokenizers use the SAME token IDs for shared vocabulary. Token ID 262 means the same thing ('Ġthe') in both. The pidgin tokenizer just can't access token IDs >= 1000.

### Handling Sequence Length Mismatch

Since pidgin tokenization produces more tokens for the same text, we need alignment:

**Option A: Fixed token count, variable text coverage**
```python
def tokenize_pair_fixed_tokens(text_A, text_B, seq_length=256):
    """
    Tokenize both streams to exactly seq_length tokens.
    Stream B may cover less text due to coarser tokenization.
    """
    tokens_A = main_tokenizer.encode(text_A)[:seq_length]
    tokens_B = pidgin_tokenizer.encode(text_B)[:seq_length]
    
    # Pad if needed
    tokens_A = pad_to_length(tokens_A, seq_length, pad_id=0)
    tokens_B = pad_to_length(tokens_B, seq_length, pad_id=0)
    
    return tokens_A, tokens_B
```

**Option B: Fixed text, variable token count (then pad)**
```python
def tokenize_pair_fixed_text(text_A, text_B, max_length=256):
    """
    Tokenize same-length texts, pad to the longer sequence.
    """
    tokens_A = main_tokenizer.encode(text_A)
    tokens_B = pidgin_tokenizer.encode(text_B)
    
    # Pad both to max of the two lengths (up to max_length)
    target_len = min(max(len(tokens_A), len(tokens_B)), max_length)
    
    tokens_A = pad_to_length(tokens_A[:max_length], target_len, pad_id=0)
    tokens_B = pad_to_length(tokens_B[:max_length], target_len, pad_id=0)
    
    return tokens_A, tokens_B
```

**Recommendation**: Start with Option A (fixed token count). It's simpler and ensures consistent tensor shapes. The text coverage mismatch is fine—we're testing whether streams can be unmixed, not whether they cover identical content.

### Verifying Tokenizer Correctness

Before training, verify the tokenizers work as expected:

```python
def verify_tokenizers():
    test_strings = [
        "Hello world",
        "The quick brown fox",
        "Machine learning is fascinating",
        "12345",  # Numbers
        "café naïve",  # Accented characters
    ]
    
    for s in test_strings:
        main_toks = main_tokenizer.encode(s)
        pidgin_toks = pidgin_tokenizer.encode(s)
        
        print(f"Text: {s}")
        print(f"  Main:   {len(main_toks.ids)} tokens - {main_toks.tokens}")
        print(f"  Pidgin: {len(pidgin_toks.ids)} tokens - {pidgin_toks.tokens}")
        
        # Verify: pidgin should produce >= as many tokens
        assert len(pidgin_toks.ids) >= len(main_toks.ids)
        
        # Verify: shared tokens have same IDs
        # (tokens that exist in both vocabs should have identical IDs)
        
    print("Tokenizer verification passed!")
```

### Dataset:
- Use a standard corpus (WikiText-103, OpenWebText, or similar)
- Each training example: two unrelated text chunks
- Stream A: chunk from position i
- Stream B: chunk from position j (different document or far offset)

**Alignment:**
- Pad/truncate both streams to same length (e.g., 256 tokens)
- Handle length mismatch between main and pidgin tokenization

**DataLoader Output:**
```python
{
    'tokens_A': tensor([batch, seq_len]),      # main stream token IDs
    'tokens_B': tensor([batch, seq_len]),      # pidgin stream token IDs  
    'labels_A': tensor([batch, seq_len]),      # shifted targets for A
    'labels_B': tensor([batch, seq_len]),      # shifted targets for B
}
```

---

## Phase 4: Training Loop

### File: `train.py`

**Training Configuration:**
```python
config = {
    'model_name': 'gpt2',                # GPT-2 small
    'pidgin_vocab_size': 1000,
    'seq_length': 256,                   # context length
    'batch_size': 4,                     # limited by VRAM
    'gradient_accumulation_steps': 8,   # effective batch = 32
    'learning_rate': 5e-5,
    'num_epochs': 3,
    'warmup_steps': 500,
    'eval_every': 500,                   # steps
    'save_every': 2000,
    'loss_weight_A': 1.0,                # weight for main stream loss
    'loss_weight_B': 1.0,                # weight for pidgin stream loss
}
```

**Training Loop Pseudocode:**
```python
for epoch in range(num_epochs):
    for batch in dataloader:
        tokens_A, tokens_B = batch['tokens_A'], batch['tokens_B']
        labels_A, labels_B = batch['labels_A'], batch['labels_B']
        
        logits_A, logits_B = model(tokens_A, tokens_B)
        
        loss_A = cross_entropy(logits_A, labels_A)
        loss_B = cross_entropy(logits_B, labels_B)
        loss = config['loss_weight_A'] * loss_A + config['loss_weight_B'] * loss_B
        
        loss.backward()
        
        if step % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # Logging
        log({'loss_A': loss_A, 'loss_B': loss_B, 'loss_total': loss})
```

**Memory Optimizations:**
- Mixed precision (fp16) via torch.cuda.amp
- Gradient checkpointing if needed
- Small batch + gradient accumulation

---

## Phase 5: Evaluation

### File: `evaluate.py`

**Metrics to Track:**

1. **Stream Separation (Primary Goal)**
   - Accuracy of head_A predicting stream A tokens
   - Accuracy of head_B predicting stream B tokens
   - Cross-prediction accuracy (head_A on stream B, head_B on stream A)
   - Goal: high same-stream accuracy, low cross-stream accuracy

2. **Main Stream Quality (Regression Check)**
   - Perplexity on held-out text (stream A only, no stream B)
   - Compare to: original GPT-2, GPT-2 fine-tuned on same data without dual-stream
   - Goal: not significantly worse than baseline

3. **Pidgin Stream Quality**
   - Perplexity on stream B predictions
   - Qualitative: sample from head_B, inspect if coherent

**Evaluation Scenarios:**

```python
# Scenario 1: Dual-stream (normal operation)
logits_A, logits_B = model(tokens_A, tokens_B)
# Measure both accuracies

# Scenario 2: Main stream only (zero out pidgin)
logits_A, _ = model(tokens_A, zeros_like(tokens_B))
# Measure stream A accuracy — does removing B hurt?

# Scenario 3: Cross-prediction (sanity check)
# Use head_A to predict tokens_B, head_B to predict tokens_A
# Should be near random chance if streams are separated
```

**Baselines for Comparison:**
- Original GPT-2 small (no fine-tuning)
- GPT-2 fine-tuned on same data, single stream
- GPT-2 with noise added to embeddings (ablation: is stream B just noise?)

---

## Phase 6: Experimental Progression

### Experiment 1: Sanity Check
- Two completely unrelated text streams
- Goal: verify the model can learn to unmix at all
- Success: both heads achieve >50% top-1 accuracy (better than random)

### Experiment 2: Measure Interference  
- Compare main stream perplexity with/without pidgin stream
- Goal: understand the cost of dual-stream processing
- Metric: perplexity increase over single-stream baseline

### Experiment 3: Related Streams
- Stream A: text, Stream B: simplified/summarized version of same text
- Goal: see if related pidgin stream helps main prediction
- This is closest to the "internal reasoning" use case

### Experiment 4: Reasoning Task
- Stream A: math problem + answer, Stream B: step-by-step reasoning (pidginized)
- Goal: test if explicit reasoning stream improves answer accuracy
- Compare to: CoT fine-tuning, no-CoT baseline

---

## Implementation Order

### Week 1: Core Infrastructure
1. [ ] Set up project structure and dependencies
2. [ ] Implement `tokenizer.py` with truncated BPE tokenizers
3. [ ] Verify tokenizers: test encoding/decoding, compare token counts
4. [ ] Implement `model.py` with DualStreamGPT2 class
5. [ ] Verify forward pass works, shapes are correct

### Week 2: Training Pipeline
6. [ ] Implement `data.py` with dual-stream dataset loader
7. [ ] Implement `train.py` with basic loop
8. [ ] Add mixed precision, gradient accumulation
9. [ ] Add checkpointing and logging
10. [ ] Run first training on small data subset

### Week 3: Evaluation & Iteration
11. [ ] Implement `evaluate.py` with all metrics
12. [ ] Run Experiment 1 (sanity check)
13. [ ] Analyze results, debug if needed
14. [ ] Run Experiment 2 (interference measurement)

### Week 4: Extensions
15. [ ] Run Experiments 3-4 (related streams, reasoning)
16. [ ] Document findings
17. [ ] Consider scaling to GPT-2 medium if results promising

---

## Key Code Snippets

### Loading and Modifying GPT-2
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.nn as nn

class DualStreamGPT2(nn.Module):
    def __init__(self, pidgin_vocab_size=1000):
        super().__init__()
        
        # Load pretrained GPT-2
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
        hidden_size = self.gpt2.config.n_embd  # 768 for GPT-2 small
        
        # Freeze main embeddings
        self.gpt2.transformer.wte.weight.requires_grad = False
        
        # New pidgin components
        self.pidgin_embeddings = nn.Embedding(pidgin_vocab_size, hidden_size)
        self.pidgin_head = nn.Linear(hidden_size, pidgin_vocab_size)
        
        # Initialize
        self.pidgin_embeddings.weight.data.normal_(mean=0, std=0.02)
        self.pidgin_head.weight.data.normal_(mean=0, std=0.02)
```

### Forward Pass
```python
def forward(self, tokens_A, tokens_B, labels_A=None, labels_B=None):
    # Get embeddings
    embed_A = self.gpt2.transformer.wte(tokens_A)
    embed_B = self.pidgin_embeddings(tokens_B)
    
    # Sum embeddings
    combined = embed_A + embed_B
    
    # Add position embeddings
    position_ids = torch.arange(tokens_A.size(1), device=tokens_A.device)
    position_embeds = self.gpt2.transformer.wpe(position_ids)
    hidden = combined + position_embeds
    
    # Pass through transformer blocks
    for block in self.gpt2.transformer.h:
        hidden = block(hidden)[0]
    
    hidden = self.gpt2.transformer.ln_f(hidden)
    
    # Predictions
    logits_A = self.gpt2.lm_head(hidden)
    logits_B = self.pidgin_head(hidden)
    
    # Compute loss if labels provided
    loss = None
    if labels_A is not None and labels_B is not None:
        loss_fn = nn.CrossEntropyLoss()
        loss_A = loss_fn(logits_A.view(-1, logits_A.size(-1)), labels_A.view(-1))
        loss_B = loss_fn(logits_B.view(-1, logits_B.size(-1)), labels_B.view(-1))
        loss = loss_A + loss_B
    
    return logits_A, logits_B, loss
```

### Truncated Tokenizer Creation
```python
from transformers import GPT2Tokenizer
from tokenizers import Tokenizer, models, pre_tokenizers, decoders

def create_dual_tokenizers(main_vocab_size=49000, pidgin_vocab_size=1000):
    """
    Create main and pidgin tokenizers from GPT-2's merge rules.
    
    Both tokenizers share the same base vocabulary (256 bytes) and
    early merges, but pidgin uses fewer merges = coarser chunking.
    """
    # Extract merges from pretrained GPT-2
    gpt2_tok = GPT2Tokenizer.from_pretrained('gpt2')
    all_merges = sorted(gpt2_tok.bpe_ranks.items(), key=lambda x: x[1])
    all_merges = [pair for pair, rank in all_merges]
    
    def build_tokenizer(num_merges):
        # Base vocab: all 256 bytes
        vocab = {bytes([i]).decode('latin-1'): i for i in range(256)}
        
        # Add merged tokens
        truncated = all_merges[:num_merges]
        for idx, (t1, t2) in enumerate(truncated):
            vocab[t1 + t2] = 256 + idx
        
        # Build tokenizer
        tok = Tokenizer(models.BPE(vocab=vocab, merges=truncated))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        return tok
    
    main_tokenizer = build_tokenizer(main_vocab_size - 256)    # 48,744 merges
    pidgin_tokenizer = build_tokenizer(pidgin_vocab_size - 256) # 744 merges
    
    return main_tokenizer, pidgin_tokenizer
```

---

## Potential Issues and Mitigations

| Issue | Mitigation |
|-------|-----------|
| Model ignores stream B (treats as noise) | Increase loss_weight_B, verify gradients flow |
| Stream A quality degrades badly | Freeze more of transformer, use LoRA |
| Pidgin predictions are random | Check pidgin embedding gradients, try larger pidgin vocab |
| OOM on 8GB GPU | Reduce seq_length, use gradient checkpointing |
| Streams interfere destructively | Try orthogonal initialization for pidgin embeds |
| Tokenizer mismatch (HF vs tokenizers lib) | Verify byte-level encoding matches, test edge cases |
| Pidgin sequences much longer than main | Use fixed token count (Option A), or increase pidgin vocab size |
| Special tokens (<PAD>, <EOS>) handling | Ensure both tokenizers use same special token IDs |

---

## Success Criteria

**Minimum Viable Result:**
- Both heads predict their respective streams better than random
- Main stream perplexity within 2x of baseline

**Good Result:**
- Both heads achieve >60% top-1 accuracy
- Main stream perplexity within 20% of baseline
- Clear evidence of separation (low cross-prediction accuracy)

**Exciting Result:**
- Related/reasoning stream B improves stream A performance
- Model demonstrates ability to "think" in pidgin while "speaking" normally

---

## References

- Coconut paper: arxiv.org/abs/2412.06769
- Quiet-STaR: arxiv.org/abs/2403.09629  
- Multi-token prediction: arxiv.org/abs/2404.19737
- GPT-2: huggingface.co/gpt2
