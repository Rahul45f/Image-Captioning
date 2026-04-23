# Show, Attend and Tell — A PyTorch Tutorial to Image Captioning

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Stars](https://img.shields.io/github/stars/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)
[![Forks](https://img.shields.io/github/forks/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)

A step-by-step tutorial on implementing the **"Show, Attend and Tell"** image captioning model using PyTorch. This project reproduces the neural image caption generation model with visual attention, as proposed by Xu et al. (ICML 2015)[web:19].

> 🔹 Part of the **[Deep Tutorials for PyTorch](https://github.com/sgrvinod/Deep-Tutorials-for-PyTorch)** series.
> Also check out the companion tutorials: [Super-Resolution](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution) and [Machine Translation](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Machine-Translation).

---

## 📋 Table of Contents

- [Objective](#objective)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference (Generate Captions)](#inference)
- [Results](#results)
- [How It Works](#how-it-works)
- [Frequently Asked Questions](#frequently-asked-questions)
- [References](#references)

---

## 🎯 Objective

Build a deep learning model that can generate a natural language caption for any input image. The model learns **where to look** — as it generates each word, you can visualize its "gaze" shifting across different regions of the image, thanks to its **attention mechanism**.

### Example Captions

| Image | Generated Caption |
|-------|------------------|
| Airplane | *"A plane sitting on top of an airport runway."* |
| Boats | *"Two boats are sitting in the water."* |

*(Run `caption.py` with your own images to see the model in action!)*

---

## 🏗️ Model Architecture

This implementation follows the **encoder-decoder with attention** paradigm:

| Component | Description |
|-----------|-------------|
| **Encoder** | Pretrained **ResNet-101** (ImageNet) with Adaptive Average Pooling, outputting a fixed-size `(2048, 14, 14)` feature map. |
| **Attention Network** | Computes soft attention weights (α) over spatial regions of the encoded image at each decoding timestep. |
| **Decoder** | Single-layer **LSTM** that generates captions word-by-word, conditioned on the attention-weighted image features and previous hidden state. |
| **Beam Search** | Greedy decoding with beam search (k=5) during inference for higher-quality outputs. |

**Overall flow:**
Input Image → ResNet-101 Encoder → Attention-weighted Features → LSTM Decoder → Caption


---

## 📊 Dataset

The model is trained on the **MS COCO 2014** dataset using the **Karpathy splits**[web:25]:

| Split | Images |
|-------|--------|
| Training | 113,287 |
| Validation | 5,000 |
| Test | 5,000 |

Each image has **5 ground-truth captions**. The dataset supports three variants:
- `coco` (default)
- `flickr8k`
- `flickr30k`

---

## ⚙️ Prerequisites

- Python 3.6+
- PyTorch 0.4+
- CUDA-compatible GPU (recommended for training)
- NLTK, h5py, scikit-image, Pillow, tqdm

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning.git
cd a-PyTorch-Tutorial-to-Image-Captioning

# Install dependencies
pip install torch torchvision nltk h5py scikit-image pillow tqdm scipy
```

---

## 📁 Project Structure
a-PyTorch-Tutorial-to-Image-Captioning/
├── README.md # This file
├── LICENSE # MIT License
├── img/ # Sample output images
├── create_input_files.py # Preprocess dataset into HDF5 + JSON
├── datasets.py # PyTorch Dataset & DataLoader
├── models.py # Encoder, Attention, Decoder classes
├── train.py # Training loop with validation
├── eval.py # Evaluate model on test set (BLEU scores)
├── caption.py # Generate captions for custom images
└── utils.py # Utility functions & data preprocessing


---

## 🚀 Quick Start

### Step 1: Prepare the Dataset

```bash
python create_input_files.py
```

This downloads the COCO dataset (if not present), applies the Karpathy splits, resizes images, and saves them as HDF5 files along with a word map JSON.

### Step 2: Train the Model

```bash
python train.py
```

Training runs for 120 epochs with early stopping based on validation BLEU-4. Checkpoints are saved automatically.

### Step 3: Evaluate on Test Set

```bash
python eval.py
```

Computes BLEU-1 through BLEU-4 scores on the test set.

### Step 4: Generate Captions for Custom Images

```bash
python caption.py \
  --img='path/to/your/image.jpg' \
  --model='path/to/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar' \
  --word_map='path/to/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json' \
  --beam_size=5
```

This outputs the generated caption and a visualization of attention heatmaps.

---

## 🧠 Training

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Embedding dimension | 512 |
| Attention dimension | 512 |
| Decoder hidden dim | 512 |
| Dropout | 0.5 |
| Batch size | 32 |
| Epochs | 120 |
| Encoder learning rate | 1e-4 |
| Decoder learning rate | 4e-4 |
| Gradient clip | 5.0 |
| Attention regularization (α_c) | 1.0 |

The training loop uses:
- **CrossEntropyLoss** for caption generation
- **Doubly stochastic regularization** on attention weights to encourage coverage
- **Teacher Forcing** (ground-truth captions as decoder input during training)
- **Learning rate decay** when validation BLEU-4 plateaus
- **Early stopping** based on validation performance

---

## 📈 Results

The trained model achieves competitive BLEU scores on the COCO test set:

| Metric | Score |
|--------|-------|
| BLEU-1 | ~68 |
| BLEU-2 | ~51 |
| BLEU-3 | ~39 |
| **BLEU-4** | **~33.29** |

*(Scores may vary slightly depending on hardware and random seed)*

A **pretrained checkpoint** is available in the repository.

---

## ⚙️ How It Works

### Encoder
The encoder uses a pretrained **ResNet-101** (trained on ImageNet) with the final fully connected and pooling layers removed. An adaptive average pooling layer resizes feature maps to `(14, 14)`, producing a tensor of shape `(batch, 2048, 14, 14)`.

### Attention Mechanism
At each decoding timestep, the attention network:
1. Projects encoder features and the decoder's hidden state into a shared attention space
2. Computes alignment scores via a linear layer
3. Applies softmax to get attention weights (α) across spatial positions
4. Produces a weighted sum of encoder features — the **context vector**

### Decoder
The decoder is an LSTM that:
1. Takes the context vector and previous word embedding as input
2. Updates its hidden state
3. Predicts the next word from the vocabulary
4. Repeats until it generates the `<end>` token

### Beam Search
During inference, instead of greedily picking the highest-probability word at each step, beam search maintains the top-k partial sequences, resulting in more coherent and fluent captions.

---

## ❓ Frequently Asked Questions

### What is the difference between Soft and Hard Attention?
**Soft attention** computes a weighted average over all spatial positions (differentiable, trained via backprop). **Hard attention** samples a single position stochastically (requires variational inference / REINFORCE). This implementation uses **soft attention**.

### What is Teacher Forcing?
Teacher Forcing feeds the **ground-truth caption** as the decoder input at each timestep during training, rather than the model's own predictions. This speeds up convergence but can cause **exposure bias** at inference time. **Scheduled Sampling** (mixing ground-truth and predictions) is a common fix.

### Can I use pretrained word embeddings (GloVe, etc.)?
Yes! Use the `load_pretrained_embeddings()` method in the `DecoderWithAttention` class after creating the decoder in `train.py`. Make sure to also adjust the `emb_dim` parameter to match your embedding dimensions.

### How do I track which tensors allow gradients?
In PyTorch 0.4+, tensors have a `requires_grad` attribute. By default, it's `False` for new tensors and `True` for parameters of `nn.Module` layers. Tensors derived from ones with `requires_grad=True` also inherit it.

### How do I compute all BLEU scores (BLEU-1 to BLEU-4)?
Modify `eval.py` to call `corpus_bleu()` with different `weights` tuples: `(1,0,0,0)` for BLEU-1, `(0.5,0.5,0,0)` for BLEU-2, etc. See [issue #37](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/issues/37) for a detailed explanation.

### Can I use Beam Search during training?
Not with the current loss function. Beam search is typically an **inference-time** technique. There are research papers exploring training with beam search, but it's not standard practice.

---

## 📚 References

- **Paper:** [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044) — Xu et al., ICML 2015[web:18]
- **Original Implementation:** [arctic-captions](https://github.com/kelvinxu/arctic-captions)
- **Dataset:** [MS COCO](https://cocodataset.org/)
- **Karpathy Splits:** [Dataset JSON files](https://github.com/Delphboy/karpathy-splits)[web:25]

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

> 💡 **Tip:** If you're new to PyTorch, start with the official [Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) and [Learning PyTorch with Examples](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html).
>
> Questions, suggestions, or bug reports? Please [open an issue](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/issues).
