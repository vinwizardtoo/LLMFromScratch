# Decoder-Only Transformer (PyTorch) — Training & Inference

This project implements a **decoder-only Transformer language model from scratch** in PyTorch. It covers the full workflow: model architecture, training loop, inference (text generation), and a downstream classification task.

The implementation is fully custom and does not rely on high-level Transformer libraries.

---

## Features

### 1. Full Transformer Model Implementation
Implemented in `model.py`:
- Multi-Head Self-Attention
- Causal attention masking
- Feed-Forward network (MLP)
- Decoder blocks
- Positional embeddings
- Token embeddings with weight tying
- Final logits projection
- Fully vectorized PyTorch operations

### 2. Training Pipeline
Implemented in `train.py`:
- Random batch sampler
- Sequential batch sampler (validation)
- Cosine annealing LR schedule with warmup
- Gradient accumulation
- Config-driven hyperparameters (`config.yaml`)
- FLOPs-bounded experiments
- Language modeling loss (cross-entropy)
- Training + validation loops with perplexity reporting

### 3. Text Generation
Implemented in `generate.py`:
- Autoregressive decoding
- Temperature-based sampling
- Softmax with adjustable temperature
- Prompt-based generation

### 4. Downstream Classification (Sentiment)
Implemented in `classify.py`:
- Logit scoring for next-token prediction
- Template-based sentiment classification
- Simple LM-based binary classifier

---

## File Structure

model.py # Transformer model components
train.py # Training loop + samplers + LR scheduler
generate.py # Text generation utilities
classify.py # Sentiment classification via LM scoring
utils.py # Helper utilities
config.yaml # Training configuration

## Pretrained Model Download

A pretrained `model.pt` is stored on Google Drive. Download it from [this link](https://drive.google.com/file/d/1nsPh-3on64wuNQJ6Z2nMNaEvjB4NQFdV/view?usp=sharing) and place it in the project root (next to `generate.py`, `classify.py`, etc.). Paths below assume the weights are saved as `model.pt`.

## How to Train

```bash
python train.py --config config.yaml
```
This will:

load the dataset (subset of C4),

initialize the model,

run training & validation,

log loss and perplexity,

save model.pt.

## How to Generate Text
```bash
python generate.py --model model.pt --prompts prefixs.json
```
Outputs generated text using temperature sampling.

## How to Run Sentiment Classification
```bash
python classify.py --model model.pt --input yelp_test.txt
```

Predicts positive/negative sentiment using LM scoring.
