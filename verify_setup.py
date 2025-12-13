#!/usr/bin/env python3
"""
Verify environment setup for Dual-Stream GPT-2 experiment.
Run this script to check that all dependencies are installed and GPU is accessible.

Usage:
    python verify_setup.py
"""

import sys


def check_python_version():
    """Verify Python 3.8+"""
    print("=" * 50)
    print("1. Python Version")
    print("=" * 50)

    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("   [FAIL] Requires Python 3.8+")
        return False

    print("   [OK]")
    return True


def check_imports():
    """Verify all required packages can be imported."""
    print("\n" + "=" * 50)
    print("2. Package Imports")
    print("=" * 50)

    packages = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("tokenizers", "tokenizers"),
        ("datasets", "datasets"),
        ("numpy", "numpy"),
        ("tqdm", "tqdm"),
    ]

    all_ok = True
    for name, import_name in packages:
        try:
            module = __import__(import_name)
            version = getattr(module, "__version__", "unknown")
            print(f"   {name}: {version} [OK]")
        except ImportError as e:
            print(f"   {name}: [FAIL] {e}")
            all_ok = False

    return all_ok


def check_cuda():
    """Check CUDA availability and GPU memory."""
    print("\n" + "=" * 50)
    print("3. CUDA / GPU")
    print("=" * 50)

    import torch

    if not torch.cuda.is_available():
        print("   CUDA: Not available")
        print("   [WARN] Training will be slow on CPU")
        return True  # Not a hard failure

    print(f"   CUDA: Available (version {torch.version.cuda})")
    print(f"   Device count: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024 ** 3)
        print(f"   GPU {i}: {props.name}")
        print(f"          Memory: {memory_gb:.1f} GB")
        print(f"          Compute capability: {props.major}.{props.minor}")

    # Check if we have enough memory (target: 8GB for GTX 1070)
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if total_memory < 6:
        print("   [WARN] Less than 6GB VRAM - may need smaller batch sizes")
    else:
        print("   [OK] Sufficient VRAM for GPT-2 small")

    return True


def check_gpt2_loading():
    """Verify GPT-2 model loads correctly."""
    print("\n" + "=" * 50)
    print("4. GPT-2 Model Loading")
    print("=" * 50)

    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        print("   Loading GPT-2 tokenizer...")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        print(f"   Tokenizer vocab size: {tokenizer.vocab_size}")

        print("   Loading GPT-2 model...")
        model = GPT2LMHeadModel.from_pretrained("gpt2")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Model parameters: {total_params / 1e6:.1f}M")

        # Check embedding dimensions
        embed_dim = model.config.n_embd
        n_layer = model.config.n_layer
        n_head = model.config.n_head
        print(f"   Architecture: {n_layer} layers, {n_head} heads, {embed_dim} dim")

        print("   [OK]")
        return True

    except Exception as e:
        print(f"   [FAIL] {e}")
        return False


def check_tokenizer_merges():
    """Verify we can access and truncate BPE merges."""
    print("\n" + "=" * 50)
    print("5. BPE Merge Extraction")
    print("=" * 50)

    try:
        from transformers import GPT2Tokenizer

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        # Access bpe_ranks (merge priority)
        if not hasattr(tokenizer, "bpe_ranks"):
            print("   [FAIL] Cannot access bpe_ranks")
            return False

        merges = sorted(tokenizer.bpe_ranks.items(), key=lambda x: x[1])
        print(f"   Total merges available: {len(merges)}")

        # Show first few merges (most common patterns)
        print("   First 5 merges (most common):")
        for (t1, t2), rank in merges[:5]:
            print(f"      {rank}: '{t1}' + '{t2}' -> '{t1}{t2}'")

        # Verify we have enough merges for our vocab sizes
        main_merges_needed = 49000 - 256  # 48,744
        pidgin_merges_needed = 1000 - 256  # 744

        if len(merges) >= main_merges_needed:
            print(f"   [OK] Enough merges for main vocab (need {main_merges_needed})")
        else:
            print(f"   [FAIL] Not enough merges (have {len(merges)}, need {main_merges_needed})")
            return False

        return True

    except Exception as e:
        print(f"   [FAIL] {e}")
        return False


def check_forward_pass():
    """Test a simple forward pass to verify GPU execution."""
    print("\n" + "=" * 50)
    print("6. Forward Pass Test")
    print("=" * 50)

    try:
        import torch
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Using device: {device}")

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

        # Test input
        text = "The quick brown fox"
        inputs = tokenizer(text, return_tensors="pt").to(device)

        print(f"   Input: '{text}'")
        print(f"   Token IDs: {inputs['input_ids'].tolist()}")

        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        print(f"   Output shape: {logits.shape}")

        # Check we get valid probabilities
        probs = torch.softmax(logits[0, -1, :], dim=-1)
        top_prob, top_idx = probs.max(dim=-1)
        top_token = tokenizer.decode([top_idx.item()])
        print(f"   Next token prediction: '{top_token}' (prob: {top_prob.item():.3f})")

        # Memory usage
        if device == "cuda":
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            print(f"   GPU memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

        print("   [OK]")
        return True

    except Exception as e:
        print(f"   [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\nDual-Stream GPT-2: Environment Verification")
    print("=" * 50)

    results = []

    results.append(("Python version", check_python_version()))
    results.append(("Package imports", check_imports()))
    results.append(("CUDA/GPU", check_cuda()))
    results.append(("GPT-2 loading", check_gpt2_loading()))
    results.append(("BPE merges", check_tokenizer_merges()))
    results.append(("Forward pass", check_forward_pass()))

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    all_passed = True
    for name, passed in results:
        status = "[OK]" if passed else "[FAIL]"
        print(f"   {name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All checks passed! Ready for Phase 2.")
    else:
        print("Some checks failed. Please fix the issues above.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
