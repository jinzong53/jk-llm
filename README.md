
<h1 align="center">🚀 JK-LLM：从零构建简易大语言模型</h1>

<!-- <p align="center">
[English Version](README_en.md) | [中文版本](README.md)
</p> -->

<p align="center">
使用 PyTorch 实现 Decoder-Only Transformer（GPT-like）模型，从零开始理解 LLM 的内部结构与训练过程
</p>

<p align="center">
<a><img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white"></a>
<a><img src="https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white"></a>
<a><img src="https://img.shields.io/badge/License-MIT-green.svg"></a>
<a><img src="https://img.shields.io/badge/Model-GPT%20from%20scratch-blue"></a>
</p>

---

## 📖 目录

* [项目简介](#项目简介)
* [功能特点](#功能特点)
* [项目结构](#项目结构)
* [安装和环境配置](#安装和环境配置)
* [数据准备](#数据准备)
* [模型架构](#模型架构)
* [训练与推理](#训练与推理)
* [实验指标](#实验指标)
* [未来计划](#未来计划)

---

## 🎯 项目简介

**JK-LLM** 是一个**简易向**的 LLM 代码仓库，旨在帮助你：

* 理解最基础的 LLM 的工作原理
* 掌握 Transformer Decoder 结构
* 学会训练、评估和推理流程
* 模拟 GPT-like 模型对话


---

## ✨ 功能特点

| 功能模块         | 内容                                            |
| ------------ | --------------------------------------------- |
| 🔥 模型        | Decoder-Only Transformer (GPT-style)          |
| 🧠 Tokenizer | SentencePiece BPE                             |
| 📦 数据        | 滑动窗口文本训练集构建                                   |
| ⚙️ 训练        | AdamW + AMP + Grad Accum + Warmup + Cosine LR |
| 📊 评估        | Cross entropy & perplexity                    |
| ✍️ 推理        | Greedy decoding 文本生成                          |

---

## 📂 项目结构

```
jk-llm/
├── src/
│   ├── data/          # 数据处理
│   ├── models/        # Transformer实现
│   ├── tokenizer/     # SentencePiece tokenizer
│   ├── train.py       # 训练脚本
│   ├── evaluate.py    # 验证/困惑度
│   └── infer.py       # 文本生成推理
├── configs/           # YAML配置
├── artifacts/         # 语料 & 生成文件(.gitignore)
└── checkpoints/       # 模型权重(.gitignore)
```

---

## 🛠️ 安装和环境配置

### ✅ 环境要求

* Python ≥ 3.8
* PyTorch ≥ 2.0
* GPU（可选，但推荐）

### ⚙️ 安装步骤

**Step 1：安装 PyTorch（根据系统）**

> 推荐使用官网命令：[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

例如（CPU）：

```bash
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

例如（CUDA 11.8）：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**Step 2：安装其他依赖**

```bash
pip install -r requirements.txt
```

---

## 📚 数据准备

### 0️⃣ 下载数据集（可选）

你可以从 [百度AI Studio](https://aistudio.baidu.com/datasetdetail/83697) 下载新闻数据集。下载后，数据集为.dat格式，你需要将其转换为.txt格式并放入 `artifacts/corpus/` 目录下的 `train.txt`、`val.txt` 和 `test.txt` 文件中。

### 1️⃣ 准备原始语料（自行放入）

```
artifacts/corpus/train.txt
artifacts/corpus/val.txt
artifacts/corpus/test.txt
```

### 2️⃣ 训练分词器

```bash
python src/tokenizer/train_tokenizer.py
```

### 3️⃣ 构建训练数据集

```bash
python src/data/build_datasets.py
```

---

## 🧠 模型架构

> Decoder-Only Transformer（类 GPT）

核心网络：

```python
class DecoderOnlyTransformer(nn.Module):
    ...
```

包含：

* Token + Position Embedding
* 多头自注意力 + 因果 Mask
* FFN + GELU
* LayerNorm + 残差
* 交叉熵训练目标

---

## 🚀 训练与推理

### 🏋️ 启动训练

```bash
python src/train.py --config configs/train_small.yaml
```

### ✅ 评估困惑度

```bash
python src/evaluate.py
```

### ✨ 文本生成

```bash
python src/infer.py --prompt "Hello world"
```

---

## 📈 实验指标

| 指标         | 说明            |
| ---------- | ------------- |
| Loss       | Cross entropy |
| Perplexity | exp(loss)     |
| Speed      | tokens/sec    |
| Memory     | GPU 显存占用      |

---

## 🔭 未来计划

* [ ] 加入 Flash-Attention
* [ ] 加入 RoPE / ALiBi 位置编码
* [ ] 加入 RLHF / LoRA finetune demo
* [ ] 提供中文训练样例
* [ ] 发布 Colab Notebook

---


本项目使用 **MIT License**。
