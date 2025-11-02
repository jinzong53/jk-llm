# 🚀 JK-LLM: Building a Simple Large Language Model from Scratch

<p align="center">
[Chinese Version](README.md) | [English Version](README_en.md)
</p>

<p align="center">
Implementing a Decoder-Only Transformer (GPT-like) model using PyTorch, understanding the internal structure and training process of LLMs from scratch
</p>

<p align="center">
<a><img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white"></a>
<a><img src="https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white"></a>
<a><img src="https://img.shields.io/badge/License-MIT-green.svg"></a>
<a><img src="https://img.shields.io/badge/Model-GPT%20from%20scratch-blue"></a>
</p>

---

## 📖 Table of Contents

* [Project Introduction](#project-introduction)
* [Features](#features)
* [Project Structure](#project-structure)
* [Installation and Environment Setup](#installation-and-environment-setup)
* [Data Preparation](#data-preparation)
* [Model Architecture](#model-architecture)
* [Training and Inference](#training-and-inference)
* [Experiment Metrics](#experiment-metrics)
* [Future Plans](#future-plans)
* [Acknowledgements and References](#acknowledgements-and-references)
* [Contributions and License](#contributions-and-license)

---

## 🎯 Project Introduction

**JK-LLM** is an **educational** LLM codebase designed to help you:

* Understand how LLMs work
* Master the Transformer Decoder structure
* Learn the training, evaluation, and inference processes
* Gain the ability to build GPT-like models

> **Goal:** Build a "runnable, learnable, and extensible" LLM example in the clearest way possible.

---

## ✨ Features

| Module       | Content                                      |
| ------------ | -------------------------------------------- |
| 🔥 Model     | Decoder-Only Transformer (GPT-style)         |
| 🧠 Tokenizer | SentencePiece BPE                            |
| 📦 Data      | Sliding window text training set construction |
| ⚙️ Training  | AdamW + AMP + Grad Accum + Warmup + Cosine LR |
| 📊 Evaluation| Cross entropy & perplexity                   |
| ✍️ Inference | Greedy decoding text generation              |

---

## 📂 Project Structure

```
jk-llm/
├── src/
│   ├── data/          # Data processing
│   ├── models/        # Transformer implementation
│   ├── tokenizer/     # SentencePiece tokenizer
│   ├── train.py       # Training script
│   ├── evaluate.py    # Validation/perplexity
│   └── infer.py       # Text generation inference
├── configs/           # YAML configurations
├── artifacts/         # Corpus & generated files (.gitignore)
└── checkpoints/       # Model weights (.gitignore)
```

---

## 🛠️ Installation and Environment Setup

### ✅ Environment Requirements

* Python ≥ 3.8
* PyTorch ≥ 2.0
* GPU (optional, but recommended)

### ⚙️ Installation Steps

**Step 1: Install PyTorch (based on your system)**

> Recommended to use the official command: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

For example (CPU):

```bash
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For example (CUDA 11.8):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**Step 2: Install other dependencies**

```bash
pip install -r requirements.txt
```

---

## 📚 Data Preparation

### 0️⃣ Download dataset (optional)

You can download the news dataset from [Baidu AI Studio](https://aistudio.baidu.com/datasetdetail/83697). After downloading, the dataset is in .dat format, you need to convert it to .txt format and place it in the `artifacts/corpus/` directory as `train.txt`, `val.txt`, and `test.txt` files.

### 1️⃣ Prepare raw corpus (add your own)

```
artifacts/corpus/train.txt
artifacts/corpus/val.txt
artifacts/corpus/test.txt
```

### 2️⃣ Train tokenizer

```bash
python src/tokenizer/train_tokenizer.py
```

### 3️⃣ Build training datasets

```bash
python src/data/build_datasets.py
```

---

## 🧠 Model Architecture

> Decoder-Only Transformer (GPT-like)

Core network:

```python
class DecoderOnlyTransformer(nn.Module):
    ...
```

Includes:

* Token + Position Embedding
* Multi-head self-attention + Causal Mask
* FFN + GELU
* LayerNorm + Residual
* Cross-entropy training objective

---

## 🚀 Training and Inference

### 🏋️ Start training

```bash
python src/train.py --config configs/train_small.yaml
```

### ✅ Evaluate perplexity

```bash
python src/evaluate.py
```

### ✨ Text generation

```bash
python src/infer.py --prompt "Hello world" --max-new-tokens 200 --temperature 0.8 --top-k 40
```

---

## 📈 Experiment Metrics

| Metric      | Description     |
| ----------- | --------------- |
| Loss        | Cross entropy   |
| Perplexity  | exp(loss)       |
| Speed       | tokens/sec      |
| Memory      | GPU memory usage|

---

## 🔭 Future Plans

* [ ] Add Flash-Attention
* [ ] Add RoPE / ALiBi positional encoding
* [ ] Add RLHF / LoRA finetune demo
* [ ] Provide Chinese training examples
* [ ] Release Colab Notebook

---

## 🙏 Acknowledgements and References

This project uses **MIT License**.