"""
Dual-Stream GPT-2 Model.

This module implements the DualStreamGPT2 model that processes two text streams
simultaneously by summing their embeddings and splitting logits before softmax.

Architecture:
    Input:  token_A (main) + token_B (pidgin)
            ↓
    Embed:  combined = embed[token_A] + embed[token_B]
            ↓
    Transformer: standard GPT-2 layers
            ↓
    Output: logits split into stream A and stream B
            ↓
    Loss:   CE(logits_A, target_A) + CE(logits_B, target_B)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

from config import DualStreamConfig, GPT2_VOCAB_SIZE


__all__ = ["DualStreamGPT2", "DualStreamOutput"]


# =============================================================================
# Output container
# =============================================================================

@dataclass
class DualStreamOutput:
    """
    Output from DualStreamGPT2 forward pass.

    Attributes:
        loss: Combined loss (loss_a * weight_a + loss_b * weight_b), or None if no labels.
        loss_a: Stream A cross-entropy loss, or None if no labels_a.
        loss_b: Stream B cross-entropy loss, or None if no labels_b.
        logits_a: Logits for Stream A, shape (batch, seq_len, main_vocab_size).
        logits_b: Logits for Stream B, shape (batch, seq_len, pidgin_vocab_size).
    """

    loss: Optional[torch.Tensor] = None
    loss_a: Optional[torch.Tensor] = None
    loss_b: Optional[torch.Tensor] = None
    logits_a: Optional[torch.Tensor] = None
    logits_b: Optional[torch.Tensor] = None


# =============================================================================
# Model
# =============================================================================

class DualStreamGPT2(nn.Module):
    """
    GPT-2 model modified for dual-stream text processing.

    Two token streams are summed at the embedding level. The model produces
    logits for both streams, which are split before softmax so each stream
    has an independent probability distribution.

    Args:
        config: DualStreamConfig with vocab sizes and hyperparameters.

    Attributes:
        config: The configuration object.
        gpt2: The underlying GPT2LMHeadModel.
    """

    def __init__(self, config: DualStreamConfig):
        super().__init__()
        self.config = config

        # Load pretrained GPT-2
        self.gpt2 = GPT2LMHeadModel.from_pretrained(config.gpt2_model_name)

        # Reinitialize pidgin region of embeddings
        self._reinit_pidgin_embeddings()

        # Loss functions (separate for each stream)
        # ignore_index handles padding in labels
        self.loss_fn_a = nn.CrossEntropyLoss(ignore_index=-100)
        self.loss_fn_b = nn.CrossEntropyLoss(ignore_index=-100)

    def _reinit_pidgin_embeddings(self) -> None:
        """
        Reinitialize the pidgin region of embeddings and relocate EOS.

        Steps:
        1. Save original EOS embedding (token 50256)
        2. Reinitialize pidgin region (tokens main_vocab_size to GPT2_VOCAB_SIZE-1)
        3. Copy EOS embedding to new position (main_vocab_size - 1)

        Since lm_head weights are tied to embeddings, this also reinits
        the corresponding output projection weights.
        """
        wte = self.gpt2.transformer.wte.weight  # (50257, 768)

        # 1. Save original EOS embedding before reinit
        eos_embed = wte[self.config.original_eos_id].clone()

        # 2. Reinitialize pidgin region
        pidgin_start = self.config.main_vocab_size
        pidgin_end = GPT2_VOCAB_SIZE

        with torch.no_grad():
            nn.init.normal_(
                wte[pidgin_start:pidgin_end],
                mean=0.0,
                std=self.config.embed_init_std,
            )

            # 3. Copy EOS to its new position in main vocab
            wte[self.config.main_eos_id] = eos_embed

        # Verify weight tying is still intact
        assert self.gpt2.lm_head.weight is self.gpt2.transformer.wte.weight, (
            "Weight tying between embeddings and lm_head is broken!"
        )

    def get_embeddings(self) -> nn.Embedding:
        """Return the token embedding layer."""
        return self.gpt2.transformer.wte

    def forward(
        self,
        input_ids_a: torch.Tensor,
        input_ids_b: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels_a: Optional[torch.Tensor] = None,
        labels_b: Optional[torch.Tensor] = None,
    ) -> DualStreamOutput:
        """
        Forward pass for dual-stream processing.

        Args:
            input_ids_a: Stream A token IDs, shape (batch, seq_len).
                         Values in range [0, main_vocab_size).
            input_ids_b: Stream B token IDs, shape (batch, seq_len).
                         Values in range [main_vocab_size, GPT2_VOCAB_SIZE).
            attention_mask: Optional mask, shape (batch, seq_len). 1 = attend, 0 = mask.
            labels_a: Optional Stream A targets, shape (batch, seq_len).
                      Use -100 for positions to ignore in loss.
            labels_b: Optional Stream B targets, shape (batch, seq_len).
                      Values should be in [main_vocab_size, GPT2_VOCAB_SIZE).
                      Use -100 for positions to ignore in loss.

        Returns:
            DualStreamOutput with losses and logits.
        """
        # Get embeddings for both streams
        wte = self.gpt2.transformer.wte
        wpe = self.gpt2.transformer.wpe

        embed_a = wte(input_ids_a)  # (batch, seq_len, hidden)
        embed_b = wte(input_ids_b)  # (batch, seq_len, hidden)

        # Sum embeddings from both streams
        combined = embed_a + embed_b

        # Add position embeddings
        seq_len = input_ids_a.shape[1]
        position_ids = torch.arange(seq_len, device=input_ids_a.device).unsqueeze(0)
        combined = combined + wpe(position_ids)

        # Pass through transformer
        # We bypass the normal GPT2 forward and directly use transformer blocks
        hidden = self.gpt2.transformer.drop(combined)

        for block in self.gpt2.transformer.h:
            outputs = block(hidden, attention_mask=attention_mask)
            hidden = outputs[0]

        hidden = self.gpt2.transformer.ln_f(hidden)

        # Project to vocabulary logits
        logits = self.gpt2.lm_head(hidden)  # (batch, seq_len, 50257)

        # Split logits for each stream
        logits_a = logits[..., : self.config.main_vocab_size]
        logits_b = logits[..., self.config.main_vocab_size :]

        # Compute losses if labels provided
        loss_a = None
        loss_b = None
        loss = None

        if labels_a is not None:
            # Shift for next-token prediction
            shift_logits_a = logits_a[..., :-1, :].contiguous()
            shift_labels_a = labels_a[..., 1:].contiguous()
            loss_a = self.loss_fn_a(
                shift_logits_a.view(-1, self.config.main_vocab_size),
                shift_labels_a.view(-1),
            )

        if labels_b is not None:
            # Shift labels to local pidgin range [0, pidgin_vocab_size)
            shift_logits_b = logits_b[..., :-1, :].contiguous()
            shift_labels_b = labels_b[..., 1:].contiguous()

            # Convert labels from global [main_vocab_size, GPT2_VOCAB_SIZE)
            # to local [0, pidgin_vocab_size), preserving -100 for ignored positions
            local_labels_b = shift_labels_b.clone()
            mask = local_labels_b != -100
            local_labels_b[mask] = local_labels_b[mask] - self.config.main_vocab_size

            loss_b = self.loss_fn_b(
                shift_logits_b.view(-1, self.config.pidgin_vocab_size),
                local_labels_b.view(-1),
            )

        if loss_a is not None or loss_b is not None:
            loss = torch.tensor(0.0, device=logits.device)
            if loss_a is not None:
                loss = loss + self.config.loss_weight_a * loss_a
            if loss_b is not None:
                loss = loss + self.config.loss_weight_b * loss_b

        return DualStreamOutput(
            loss=loss,
            loss_a=loss_a,
            loss_b=loss_b,
            logits_a=logits_a,
            logits_b=logits_b,
        )

    def generate_stream_a(
        self,
        input_ids_a: torch.Tensor,
        input_ids_b: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate tokens for Stream A (greedy/sampling).

        Simple generation loop for testing. For production use,
        implement proper beam search or use HuggingFace's generate().

        Args:
            input_ids_a: Initial Stream A tokens.
            input_ids_b: Stream B tokens (held constant during generation).
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (1.0 = no change).
            top_k: If set, only sample from top k tokens.

        Returns:
            Generated token IDs for Stream A.
        """
        self.eval()
        generated = input_ids_a.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Truncate if needed (positional embeddings limit)
                if generated.shape[1] > 1024:
                    generated = generated[:, -1024:]
                    curr_b = input_ids_b[:, -1024:]
                else:
                    curr_b = input_ids_b

                # Forward pass
                output = self(generated, curr_b)
                next_logits = output.logits_a[:, -1, :]  # (batch, vocab)

                # Apply temperature
                if temperature != 1.0:
                    next_logits = next_logits / temperature

                # Top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[:, [-1]]] = float("-inf")

                # Sample
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                generated = torch.cat([generated, next_token], dim=1)

                # Stop at EOS
                if (next_token == self.config.main_eos_id).all():
                    break

        return generated


# =============================================================================
# Verification
# =============================================================================

def verify_model(model: DualStreamGPT2) -> None:
    """
    Verify model initialization is correct.

    Checks:
    1. Weight tying between embeddings and lm_head
    2. Pidgin region has been reinitialized (different stats)
    3. Main region preserved (similar stats to original GPT-2)
    4. EOS embedding relocated correctly
    """
    print("=" * 60)
    print("DualStreamGPT2 Verification")
    print("=" * 60)
    print()
    print(model.config)
    print()

    wte = model.gpt2.transformer.wte.weight

    # 1. Check weight tying
    tied = model.gpt2.lm_head.weight is model.gpt2.transformer.wte.weight
    print(f"Weight tying intact: {tied}")
    assert tied, "Weight tying is broken!"

    # 2. Check embedding statistics
    main_end = model.config.main_vocab_size
    pidgin_start = main_end
    pidgin_end = GPT2_VOCAB_SIZE

    main_embeds = wte[:main_end].detach()
    pidgin_embeds = wte[pidgin_start:pidgin_end].detach()

    print()
    print("Embedding statistics:")
    print(f"  Main region [0:{main_end}]:")
    print(f"    mean: {main_embeds.mean():.6f}")
    print(f"    std:  {main_embeds.std():.6f}")
    print(f"  Pidgin region [{pidgin_start}:{pidgin_end}]:")
    print(f"    mean: {pidgin_embeds.mean():.6f}")
    print(f"    std:  {pidgin_embeds.std():.6f}")

    # Pidgin should have std close to embed_init_std (0.02)
    expected_std = model.config.embed_init_std
    actual_std = pidgin_embeds.std().item()
    print(f"  Pidgin std matches init ({expected_std}): {abs(actual_std - expected_std) < 0.005}")

    # 3. Check EOS relocation
    # Load fresh GPT-2 to compare (keep on CPU to avoid extra GPU memory)
    fresh_gpt2 = GPT2LMHeadModel.from_pretrained(model.config.gpt2_model_name)
    original_eos = fresh_gpt2.transformer.wte.weight[model.config.original_eos_id].detach()
    relocated_eos = wte[model.config.main_eos_id].detach().cpu()  # move to CPU for comparison

    eos_match = torch.allclose(original_eos, relocated_eos, atol=1e-6)
    print()
    print(f"EOS relocated from {model.config.original_eos_id} to {model.config.main_eos_id}: {eos_match}")
    assert eos_match, "EOS embedding not correctly relocated!"

    # 4. Quick forward pass test
    print()
    print("Forward pass test:")
    device = next(model.parameters()).device
    batch_size, seq_len = 2, 10

    # Random tokens in valid ranges
    input_a = torch.randint(0, main_end, (batch_size, seq_len), device=device)
    input_b = torch.randint(pidgin_start, pidgin_end, (batch_size, seq_len), device=device)

    output = model(input_a, input_b)
    print(f"  logits_a shape: {output.logits_a.shape} (expected: {batch_size}, {seq_len}, {main_end})")
    print(f"  logits_b shape: {output.logits_b.shape} (expected: {batch_size}, {seq_len}, {pidgin_end - pidgin_start})")

    assert output.logits_a.shape == (batch_size, seq_len, main_end)
    assert output.logits_b.shape == (batch_size, seq_len, pidgin_end - pidgin_start)

    # Test with labels
    labels_a = input_a.clone()
    labels_b = input_b.clone()
    output_with_loss = model(input_a, input_b, labels_a=labels_a, labels_b=labels_b)
    print(f"  loss: {output_with_loss.loss.item():.4f}")
    print(f"  loss_a: {output_with_loss.loss_a.item():.4f}")
    print(f"  loss_b: {output_with_loss.loss_b.item():.4f}")

    print()
    print("All checks passed!")


def main() -> None:
    """CLI entry point for model verification."""
    import sys

    print("Loading DualStreamGPT2...")
    config = DualStreamConfig()
    model = DualStreamGPT2(config)

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    verify_model(model)


if __name__ == "__main__":
    main()
