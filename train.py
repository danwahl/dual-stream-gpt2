"""
Training script for Dual-Stream GPT-2.

Trains the model on WikiText-103 with two independent text streams.
Optimized for GTX 1070 (8GB VRAM) with mixed precision and gradient accumulation.

Usage:
    python train.py [--max_steps N] [--eval_steps N]
"""

from __future__ import annotations

import argparse
import math
import time
import warnings
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.amp import GradScaler, autocast

# Suppress false positive warning when resuming from checkpoint
# (we do call optimizer.step() before scheduler.step())
warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step\\(\\)` before")

from config import DualStreamConfig, TrainingConfig
from model import DualStreamGPT2
from tokenizer import DualStreamTokenizer
from dataset import create_dataloader, DualStreamBatch


# =============================================================================
# Training utilities
# =============================================================================

def get_lr_scheduler(
    optimizer: AdamW,
    num_warmup_steps: int,
    num_training_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Create linear warmup + cosine decay scheduler.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from loss."""
    return math.exp(min(loss, 100))  # Clip to avoid overflow


# =============================================================================
# Training loop
# =============================================================================

def train(
    model: DualStreamGPT2,
    train_dataloader,
    val_dataloader,
    config: DualStreamConfig,
    training_config: TrainingConfig,
    output_dir: Path,
    max_steps: int = -1,
    resume_from: dict = None,
) -> dict:
    """
    Main training loop.

    Args:
        model: DualStreamGPT2 model.
        train_dataloader: Training data.
        val_dataloader: Validation data (can be None).
        config: Model config.
        training_config: Training hyperparameters.
        output_dir: Directory for checkpoints.
        max_steps: Override max steps (-1 to use config).
        resume_from: Optional checkpoint dict to resume from.

    Returns:
        Dict with training statistics.
    """
    device = next(model.parameters()).device
    output_dir.mkdir(parents=True, exist_ok=True)

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    # Calculate total steps
    steps_per_epoch = len(train_dataloader) // training_config.gradient_accumulation_steps
    if max_steps > 0:
        total_steps = max_steps
    elif training_config.max_steps > 0:
        total_steps = training_config.max_steps
    else:
        total_steps = steps_per_epoch * training_config.num_epochs

    # Scheduler
    scheduler = get_lr_scheduler(
        optimizer,
        num_warmup_steps=training_config.warmup_steps,
        num_training_steps=total_steps,
    )

    # Mixed precision
    scaler = GradScaler('cuda') if training_config.fp16 else None

    # Training state
    global_step = 0
    start_step = 0  # Track where this session started

    # Resume from checkpoint if provided
    if resume_from is not None:
        global_step = resume_from.get("global_step", 0)
        start_step = global_step  # Remember starting point for this session
        if "optimizer_state_dict" in resume_from:
            optimizer.load_state_dict(resume_from["optimizer_state_dict"])
        if "scheduler_state_dict" in resume_from:
            scheduler.load_state_dict(resume_from["scheduler_state_dict"])
        print(f"Resumed training from step {global_step}")
    total_loss = 0.0
    total_loss_a = 0.0
    total_loss_b = 0.0
    logging_loss = 0.0
    best_val_loss = float("inf")

    print("=" * 60)
    print("Training Configuration")
    print("=" * 60)
    print(f"  Total steps: {total_steps}")
    print(f"  Batch size: {training_config.batch_size}")
    print(f"  Gradient accumulation: {training_config.gradient_accumulation_steps}")
    print(f"  Effective batch size: {training_config.batch_size * training_config.gradient_accumulation_steps}")
    print(f"  Learning rate: {training_config.learning_rate}")
    print(f"  Mixed precision (fp16): {training_config.fp16}")
    print(f"  Output dir: {output_dir}")
    print("=" * 60)
    print()

    model.train()
    optimizer.zero_grad()
    start_time = time.time()

    epoch = 0
    while global_step < total_steps:
        epoch += 1
        print(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            batch = batch.to(device)

            # Forward pass with mixed precision
            if training_config.fp16:
                with autocast('cuda'):
                    output = model(
                        input_ids_a=batch.input_ids_a,
                        input_ids_b=batch.input_ids_b,
                        attention_mask=batch.attention_mask,
                        labels_a=batch.labels_a,
                        labels_b=batch.labels_b,
                    )
                    loss = output.loss / training_config.gradient_accumulation_steps

                scaler.scale(loss).backward()
            else:
                output = model(
                    input_ids_a=batch.input_ids_a,
                    input_ids_b=batch.input_ids_b,
                    attention_mask=batch.attention_mask,
                    labels_a=batch.labels_a,
                    labels_b=batch.labels_b,
                )
                loss = output.loss / training_config.gradient_accumulation_steps
                loss.backward()

            total_loss += loss.item()
            if output.loss_a is not None:
                total_loss_a += output.loss_a.item() / training_config.gradient_accumulation_steps
            if output.loss_b is not None:
                total_loss_b += output.loss_b.item() / training_config.gradient_accumulation_steps

            # Gradient accumulation
            if (step + 1) % training_config.gradient_accumulation_steps == 0:
                if training_config.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % training_config.logging_steps == 0:
                    elapsed = time.time() - start_time
                    session_steps = global_step - start_step
                    avg_loss = (total_loss - logging_loss) / training_config.logging_steps
                    avg_loss_a = total_loss_a / session_steps
                    avg_loss_b = total_loss_b / session_steps

                    print(
                        f"  Step {global_step}/{total_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Loss_A: {avg_loss_a:.4f} (ppl: {compute_perplexity(avg_loss_a):.2f}) | "
                        f"Loss_B: {avg_loss_b:.4f} (ppl: {compute_perplexity(avg_loss_b):.2f}) | "
                        f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                        f"Time: {elapsed:.1f}s"
                    )
                    logging_loss = total_loss

                # Evaluation
                if val_dataloader and global_step % training_config.eval_steps == 0:
                    val_loss, val_loss_a, val_loss_b = evaluate(model, val_dataloader, device, training_config)
                    print(
                        f"  [Eval] Loss: {val_loss:.4f} | "
                        f"Loss_A: {val_loss_a:.4f} (ppl: {compute_perplexity(val_loss_a):.2f}) | "
                        f"Loss_B: {val_loss_b:.4f} (ppl: {compute_perplexity(val_loss_b):.2f})"
                    )

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(model, optimizer, scheduler, global_step, output_dir / "best_model.pt")
                        print(f"  [Eval] New best model saved!")

                    model.train()

                # Checkpointing
                if global_step % training_config.save_steps == 0:
                    save_checkpoint(
                        model, optimizer, scheduler, global_step,
                        output_dir / f"checkpoint_{global_step}.pt"
                    )

                if global_step >= total_steps:
                    break

        if global_step >= total_steps:
            break

    # Final save
    save_checkpoint(model, optimizer, scheduler, global_step, output_dir / "final_model.pt")

    total_time = time.time() - start_time
    session_steps = global_step - start_step
    print()
    print("=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"  Total steps: {global_step} (session: {session_steps})")
    print(f"  Total time: {total_time:.1f}s ({total_time / 60:.1f}m)")
    print(f"  Final loss: {total_loss / session_steps:.4f}")
    print(f"  Best val loss: {best_val_loss:.4f}")

    return {
        "global_step": global_step,
        "session_steps": session_steps,
        "total_loss": total_loss / session_steps,
        "best_val_loss": best_val_loss,
        "total_time": total_time,
    }


def evaluate(
    model: DualStreamGPT2,
    dataloader,
    device: torch.device,
    training_config: TrainingConfig,
) -> tuple[float, float, float]:
    """
    Evaluate model on a dataset.

    Returns:
        Tuple of (total_loss, loss_a, loss_b).
    """
    model.eval()
    total_loss = 0.0
    total_loss_a = 0.0
    total_loss_b = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)

            if training_config.fp16:
                with autocast('cuda'):
                    output = model(
                        input_ids_a=batch.input_ids_a,
                        input_ids_b=batch.input_ids_b,
                        attention_mask=batch.attention_mask,
                        labels_a=batch.labels_a,
                        labels_b=batch.labels_b,
                    )
            else:
                output = model(
                    input_ids_a=batch.input_ids_a,
                    input_ids_b=batch.input_ids_b,
                    attention_mask=batch.attention_mask,
                    labels_a=batch.labels_a,
                    labels_b=batch.labels_b,
                )

            total_loss += output.loss.item()
            if output.loss_a is not None:
                total_loss_a += output.loss_a.item()
            if output.loss_b is not None:
                total_loss_b += output.loss_b.item()
            num_batches += 1

    return (
        total_loss / num_batches,
        total_loss_a / num_batches,
        total_loss_b / num_batches,
    )


def save_checkpoint(
    model: DualStreamGPT2,
    optimizer: AdamW,
    scheduler,
    global_step: int,
    path: Path,
) -> None:
    """Save a training checkpoint."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "global_step": global_step,
            "config": model.config,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: DualStreamGPT2,
    optimizer: AdamW = None,
    scheduler = None,
) -> int:
    """Load a training checkpoint. Returns global_step."""
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["global_step"]


# =============================================================================
# Main
# =============================================================================

def find_latest_checkpoint(output_dir: Path) -> Path | None:
    """Find the most recent checkpoint in output_dir."""
    if not output_dir.exists():
        return None

    checkpoints = list(output_dir.glob("checkpoint_*.pt"))
    if not checkpoints:
        # Check for final_model.pt
        final = output_dir / "final_model.pt"
        if final.exists():
            return final
        return None

    # Sort by step number
    def get_step(p):
        try:
            return int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            return 0

    checkpoints.sort(key=get_step, reverse=True)
    return checkpoints[0]


def main():
    parser = argparse.ArgumentParser(description="Train Dual-Stream GPT-2")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum training steps")
    parser.add_argument("--eval_steps", type=int, default=200, help="Evaluate every N steps")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every N steps")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--max_train_examples", type=int, default=None, help="Limit training examples")
    parser.add_argument("--max_val_examples", type=int, default=1000, help="Limit validation examples")
    parser.add_argument("--resume", type=str, nargs="?", const="auto", default=None,
                        help="Resume from checkpoint. Use --resume for latest, or --resume PATH for specific checkpoint")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Configs
    config = DualStreamConfig()
    training_config = TrainingConfig()
    training_config.eval_steps = args.eval_steps
    training_config.logging_steps = args.logging_steps

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Tokenizer
    print("Initializing tokenizer...")
    tokenizer = DualStreamTokenizer(config.pidgin_vocab_size)

    # Model
    print("Loading model...")
    model = DualStreamGPT2(config)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Handle checkpoint resumption
    resume_step = 0
    optimizer = None
    scheduler = None

    if args.resume:
        if args.resume == "auto":
            checkpoint_path = find_latest_checkpoint(output_dir)
            if checkpoint_path is None:
                print("No checkpoint found to resume from. Starting fresh.")
            else:
                print(f"Found latest checkpoint: {checkpoint_path}")
        else:
            checkpoint_path = Path(args.resume)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        if checkpoint_path is not None:
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            resume_step = checkpoint.get("global_step", 0)
            print(f"Resuming from step {resume_step}")

            # We'll restore optimizer/scheduler state after creating them in train()
            # Store the checkpoint for later
            args._checkpoint = checkpoint

    # Data
    print("Loading datasets...")
    train_dataloader = create_dataloader(
        tokenizer=tokenizer,
        config=config,
        training_config=training_config,
        split="train",
        max_examples=args.max_train_examples,
    )

    val_dataloader = create_dataloader(
        tokenizer=tokenizer,
        config=config,
        training_config=training_config,
        split="validation",
        max_examples=args.max_val_examples,
    )

    # Train
    print()
    train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config,
        training_config=training_config,
        output_dir=output_dir,
        max_steps=args.max_steps,
        resume_from=getattr(args, "_checkpoint", None),
    )


if __name__ == "__main__":
    main()
