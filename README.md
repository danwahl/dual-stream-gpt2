# Dual-Stream GPT-2

Train a GPT-2 model to process two text streams simultaneously by summing embeddings from a main vocabulary (~49k tokens) and a constrained "pidgin" vocabulary (~1k tokens). The model learns to predict both next tokens through separate output heads.

See [CLAUDE.md](CLAUDE.md) for detailed project documentation.

## Setup

Requires Python 3.8+ and a CUDA-capable GPU (tested on GTX 1070 8GB).

```bash
# Create virtual environment
python -m venv env
source env/bin/activate  # Linux/Mac
# env\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### GPU Compatibility Note

The pinned `torch==2.7.0` version supports older NVIDIA GPUs (Pascal/sm_61 and newer). If you have a newer GPU and want the latest PyTorch, you can modify the version constraint in `requirements.txt`.

## Project Structure

```
dual-stream-gpt2/
├── CLAUDE.md       # Detailed project documentation
├── README.md       # This file
├── requirements.txt
├── config.py       # Hyperparameters (TODO)
├── tokenizer.py    # Dual tokenizer creation (TODO)
├── model.py        # DualStreamGPT2 model (TODO)
├── data.py         # Dataset loading (TODO)
├── train.py        # Training loop (TODO)
└── evaluate.py     # Evaluation metrics (TODO)
```
