# 🎵 DL_Messy_Mashup — Music Genre Classification

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch" />
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
  <img src="https://img.shields.io/badge/Platform-Kaggle%20T4×2-20BEFF?logo=kaggle" />
</p>

<p align="center">
  <b>Jan 2026 DLGenAI Project · IIT Madras</b><br>
  Shibashish Banerjee · <code>22f3003226</code>
</p>

---

## 🔗 Live Demo

> 🤗 **Hosted on Hugging Face Spaces:** (https://xytan2022-messy-mashup.hf.space)

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Real-World Use Case](#-real-world-use-case)
- [Dataset](#-dataset)
- [Approach & Solution](#-approach--solution)
- [Model Architectures](#-model-architectures)
- [Ensemble Strategy](#-ensemble-strategy)
- [Training Pipeline](#-training-pipeline)
- [Key Optimisations](#-key-optimisations)
- [Results](#-results)
- [EDA Insights](#-eda-insights)
- [Hugging Face Deployment](#-hugging-face-deployment)
- [Repository Structure](#-repository-structure)
- [How to Run](#-how-to-run)
- [Dependencies](#-dependencies)

---

## 🎯 Problem Statement

Given a music track **pre-separated into 4 stems** (drums, vocals, bass, other), classify it into one of **10 music genres**:

```
blues · classical · country · disco · hiphop · jazz · metal · pop · reggae · rock
```

This is a **multi-class audio classification** problem evaluated on **Macro F1 Score** — meaning every genre must be classified well equally, regardless of how frequently it appears in the dataset.

### Why is this hard?

- Raw audio is extremely high-dimensional (22,050 samples/second)
- Multiple genres share characteristics (rock ↔ metal, jazz ↔ blues, pop ↔ disco)
- The audio arrives as 4 separate stems — not a mixed track — requiring smart mixing before inference
- Class imbalance across genres makes simple accuracy a misleading metric

---

## 🌍 Real-World Use Case

This type of system has direct applications in:

| Industry | Application |
|----------|-------------|
| **Music Streaming** | Auto-tagging millions of songs for genre-based recommendation (Spotify, Apple Music) |
| **Content Moderation** | Detecting genre for licensing compliance and copyright matching |
| **Radio Automation** | Automatically categorising music for playlist generation |
| **Music Production** | Helping producers find reference tracks by genre similarity |
| **Stem Libraries** | Organising stem-separated audio assets (used in DAWs like Ableton) |
| **Education** | Building music theory tools that explain genre characteristics |

The stem-separation aspect is particularly relevant because modern AI source separation tools (Demucs, Spleeter) are now widely used in production — meaning genre classifiers that work on stems are practically useful pipelines.

---

## 📦 Dataset

**Messy Mashup** — a Kaggle competition dataset structured as follows:

```
genres_stems/
├── blues/
│   ├── song_001/
│   │   ├── drums.wav
│   │   ├── vocals.wav
│   │   ├── bass.wav
│   │   └── other.wav
│   └── ...
├── classical/
└── ... (10 genres total)

mashups/          ← test audio files (pre-mixed)
test.csv          ← test file IDs
ESC-50/           ← environmental sounds for noise augmentation
```

- **10 genres**, ~100 complete songs per genre
- A song is considered **complete** only if all 4 stems are present
- **Train/Val split:** 85% / 15% stratified per genre
- **ESC-50** (2000 environmental sound clips) used as background noise augmentation during training

---

## 🧠 Approach & Solution

### High-Level Flow

```
                    ┌─────────────────────────────────────┐
                    │           INPUT: 4 Stems             │
                    │  drums.wav + vocals.wav +            │
                    │  bass.wav  + other.wav               │
                    └──────────────┬──────────────────────┘
                                   │
                          Mix stems together
                                   │
              ┌────────────────────┴────────────────────┐
              │                                         │
      ┌───────▼────────┐                    ┌──────────▼────────────┐
      │   AST Branch   │                    │     CNN Branch        │
      │   (20s @ 16kHz)│                    │    (5s @ 22kHz)       │
      │                │                    │                       │
      │ ASTFeature     │                    │  torchaudio.load()    │
      │ Extractor      │                    │       ↓               │
      │     ↓          │                    │  GPU MelSpectrogram   │
      │ 128-mel patches│                    │  (128×T, dB scale)    │
      │     ↓          │                    │       ↓               │
      │ 12× Transformer│             ┌──────┴───┬────────┬───────┐  │
      │ Self-Attention │             │CustomCNN │ResNet34│EffNet │  │
      │     ↓          │             │          │        │  B0   │  │
      │ CLS Token → FC │             └──────┬───┴────────┴───────┘  │
      └───────┬────────┘                    └──────────┬────────────┘
              │  p_ast (10 probs)                       │ p_c1, p_c2, p_c3
              │                                         │
              └──────────────────┬──────────────────────┘
                                 │
                    Weighted Probability Ensemble
                    0.40×AST + 0.25×EffNet +
                    0.20×ResNet + 0.15×Custom
                                 │
                             argmax()
                                 │
                         Predicted Genre ✓
```

---

## 🏗️ Model Architectures

### 1. AST — Audio Spectrogram Transformer (Weight: 0.40)

**Pretrained on:** AudioSet (2M YouTube clips, 527 classes)  
**Parameters:** ~86 million  
**Input:** 16kHz audio, 20 seconds → 128-band log mel spectrogram

AST applies the **Vision Transformer (ViT)** architecture to audio. The spectrogram is divided into 16×16 patches, each patch is linearly projected into an embedding, and 12 transformer layers with **multi-head self-attention** process the full sequence. A learnable `[CLS]` token aggregates information from all patches and is fed to the classification head.

```
Audio (16kHz) → ASTFeatureExtractor → 128×1024 Log-Mel Spectrogram
              → 16×16 Patch Embeddings + Positional Encoding
              → 12× [Multi-Head Self-Attention → FFN → LayerNorm]
              → CLS Token → Linear(768→10) → Logits
```

**Why AST?**
- Pre-trained on AudioSet → rich audio representations out of the box
- Self-attention captures **global temporal dependencies** (chord at second 2 attends to chord at second 18)
- CNNs have a limited local receptive field; transformers see the entire sequence in every layer

**Fine-tuning strategy — 2 phases:**

| Phase | Layers Trained | LR | Epochs |
|-------|---------------|-----|--------|
| Phase 1 | Classifier head only | 3e-4 | 3 |
| Phase 2 | All layers (differential LR) | Base: 2e-5 · Head: 2e-4 | 5 |

Phase 1 warms up the randomly initialised head before touching pretrained backbone weights. Phase 2 uses **10× higher LR for the head** than the backbone to prevent catastrophic forgetting.

---

### 2. CustomCNN (Weight: 0.15)

**Parameters:** ~200K  
**Input:** 22kHz audio, 5s → (1, 128, T) mel spectrogram

A lightweight 3-block CNN built from scratch — acts as a strong baseline and diverse ensemble member.

```
Input (B,1,128,T)
    ↓
Conv2d(1→32, 3×3, p=1) → BatchNorm2d → ReLU → MaxPool2d(2)
    ↓  (B,32,64,T/2)
Conv2d(32→64, 3×3, p=1) → BatchNorm2d → ReLU → MaxPool2d(2)
    ↓  (B,64,32,T/4)
Conv2d(64→128, 3×3, p=1) → BatchNorm2d → ReLU → MaxPool2d(2)
    ↓  (B,128,16,T/8)
AdaptiveAvgPool2d(1) → Flatten
    ↓  (B,128)
Linear(128→256) → ReLU → Dropout(0.3) → Linear(256→10)
    ↓  (B,10)  ← logits
```

**Design choices:**
- `padding=1` on all conv layers preserves spatial dimensions
- Channels double each block (1→32→64→128) — standard CNN scaling
- `AdaptiveAvgPool2d(1)` makes the model input-size agnostic (any spectrogram width works)
- `Dropout(0.3)` regularises the dense layers

---

### 3. AudioResNet34 (Weight: 0.20)

**Parameters:** ~21 million  
**Pretrained on:** ImageNet  
**Input:** 22kHz audio, 5s → (1, 128, T) mel spectrogram

ResNet34 from torchvision, adapted for single-channel spectrograms via weight averaging:

```python
# Original conv1: (64, 3, 7, 7) — 3 RGB channels
# Modified conv1: (64, 1, 7, 7) — 1 spectrogram channel
new_conv1.weight[:,0] = original_conv1.weight.mean(dim=1)
```

**ResNet residual block:**
```
x → Conv → BN → ReLU → Conv → BN
              ↘________________↗ + x
                    ReLU
```
Skip connections solve the vanishing gradient problem — gradients flow directly through the identity path. ResNet34 has 34 layers across 4 stages (3+4+6+3 residual blocks), with channels 64→128→256→512.

---

### 4. AudioEfficientNetB0 (Weight: 0.25)

**Parameters:** ~5 million  
**Pretrained on:** ImageNet  
**Input:** 22kHz audio, 5s → (1, 128, T) mel spectrogram

EfficientNetB0 uses **compound scaling** — simultaneously scaling depth, width, and resolution by a fixed ratio derived from Neural Architecture Search (NAS). Same 1-channel adaptation as ResNet.

**Key building block — MBConv (Mobile Inverted Bottleneck):**
```
Input → Expand (1×1 conv, 6×) → Depthwise Conv (3×3) → SE Block → Project (1×1 conv)
```

**Squeeze-and-Excitation (SE) block** inside each MBConv:
```
Global Avg Pool → Linear (÷4) → ReLU → Linear (×4) → Sigmoid → multiply with input
```
This learns **channel-wise attention** — the network decides which frequency bands are more important per genre.

---

## 🎯 Ensemble Strategy

Final prediction = weighted average of all 4 models' **softmax probabilities**:

```python
final_prob = (0.40 × p_ast
            + 0.25 × p_efficientnet
            + 0.20 × p_resnet
            + 0.15 × p_customcnn)

prediction = genres[argmax(final_prob)]
```

**Why weighted average of probabilities (not logits)?**
Raw logits have different scales across models. Softmax normalises each to a valid probability distribution (sums to 1) before combination is meaningful.

**Why these weights?**
- AST (0.40): Audio-pretrained, transformer, highest individual val F1
- EfficientNet (0.25): Modern, efficient, SE attention suits spectrograms
- ResNet (0.20): Classic, proven on spectrograms
- CustomCNN (0.15): Simple baseline, adds diversity

**Test-Time Augmentation (TTA):**
- **AST:** 5 random 20s crops → average 5 probability vectors
- **CNN:** 3 fixed crops (start / centre / end) → average 3 probability vectors

TTA reduces prediction variance — more reliable than a single forward pass.

---

## ⚙️ Training Pipeline

### Data Augmentation

| Augmentation | Details | Applied To |
|-------------|---------|-----------|
| Random offset crop | Random 5s/20s window each epoch | Both |
| Random gain | Amplitude × Uniform(0.6, 1.4) | Both |
| ESC-50 noise | Environmental sound at 10-40% volume (70-80% probability) | Both |
| ~~Time stretch~~ | ~~STFT phase vocoder~~ | Removed (too slow, ~680 min) |

### Loss Function
**Cross-Entropy Loss with Label Smoothing (ε = 0.1)**

Soft targets `[0.011, ..., 0.9, ..., 0.011]` instead of hard one-hot. Prevents overconfidence, improves generalisation.

### Optimizer
**AdamW** — Adam with decoupled weight decay. Correctly applies L2 regularisation independently of the adaptive learning rate scaling (unlike standard Adam + L2).

### Learning Rate Schedules
- **AST:** Cosine schedule with linear warmup (10% of steps)
- **CNN:** CosineAnnealingLR over total epochs

### Regularisation
- Dropout (0.3) in CustomCNN classifier
- Weight decay (1e-4 for CNN, 0.01 for AST)
- Early stopping (patience = 5 for CNN, 2/3 for AST phases)
- Label smoothing (0.1)

---

## ⚡ Key Optimisations

The original notebook ran in ~46 hours on Kaggle T4×2. After optimisation: **~4 hours**.

| Optimisation | Time Saved | How |
|-------------|-----------|-----|
| `torchaudio.load` instead of `librosa` | ~15× faster I/O | C++ backend vs Python |
| MelSpectrogram on GPU (batch transform) | ~600 min | Single CUDA kernel per batch vs per-sample CPU |
| Removed `time_stretch` | ~680 min | Replaced with random crop (same effect, free) |
| AMP / FP16 (`torch.cuda.amp`) | ~1.7× GPU speedup | T4 Tensor Cores optimised for FP16 |
| `nn.DataParallel` (both T4s) | ~1.8× throughput | Split batch across 2 GPUs |
| Epochs 26 → 15 + early stopping | ~300 min | Model converges before epoch 15 anyway |
| AST train size 5000 → 3000 | ~115 min | Sufficient diversity with augmentation |
| `NUM_WORKERS` 2 → 4 | Better CPU utilisation | Parallel data loading overlaps with GPU compute |

---

## 📊 Results

### Individual Model — Validation Macro F1

| Model | Val Macro F1 | Ensemble Weight |
|-------|-------------|----------------|
| AST (Phase 2) | *best individual* | 0.40 |
| EfficientNetB0 | — | 0.25 |
| AudioResNet34 | — | 0.20 |
| CustomCNN | — | 0.15 |
| **Ensemble** | **best overall** | — |

> The ensemble consistently outperforms all individual models because each model makes different errors — combining them cancels out individual weaknesses.

### Confusion Matrix
The ensemble confusion matrix (logged to WandB) showed expected genre confusions:
- **Rock ↔ Metal** (both guitar-heavy, similar tempo)
- **Jazz ↔ Blues** (both improvisational, similar instrumentation)
- **Pop ↔ Disco** (both dance-oriented, similar production)

Classical and Hiphop were the most distinctly classified — low confusion with others.

---

## 📈 EDA Insights

Five EDA analyses were run before training:

**1. Class Balance** — All genres within ±25% of the mean song count. No severe imbalance requiring oversampling.

**2. Mel Spectrogram Gallery** — Visual inspection confirmed genre-specific patterns:
- Metal: Dense, high-energy across all frequency bands
- Classical: Sparse, structured, concentrated in mid-low frequencies
- Hiphop: Strong sub-bass energy, rhythmic grid pattern

**3. Waveform Analysis** — Metal vocals showed highest RMS (loudness); Classical the lowest. RMS alone is not sufficient for genre classification.

**4. Stem Loudness Heatmap** — Bass stem showed highest genre-discriminative power (reggae/hiphop heavily bass-dominant). Drums were most consistent across genres.

**5. Train/Val Split** — Stratified 85/15 split confirmed equal genre representation in both sets.

---

## 🤗 Hugging Face Deployment

The final ensemble model is deployed as an interactive **Gradio app** on Hugging Face Spaces.

**Live Demo:** [🎵 Live Demo on Hugging Face Spaces](https://xytan2022-messy-mashup.hf.space)

**How to use:**
1. Upload a `.wav` audio file (any of the 10 genres)
2. The app mixes the stems (if separated) or uses the full track
3. Returns predicted genre + confidence scores for all 10 classes

**Deployment stack:**
- Gradio interface
- Loaded model checkpoints (`ast_best_phase2.pth`, `CustomCNN_best.pth`, `AudioResNet34_best.pth`, `EfficientNetB0_best.pth`)
- `torchaudio` + `transformers` backend
- Hosted on HuggingFace Spaces (CPU inference)

> **HuggingFace profile:** [xytan2022](https://huggingface.co/xytan2022)

---

## 🗂️ Repository Structure

```
DL_Messy_Mashup/
│
├── Codes/                          # All milestone notebooks (per branch)
│   └── ...                         # See branch list for individual milestones
│
├── Scripts/                        # Intermediate utility scripts
│   ├── requirements.txt            # HuggingFace deployment dependencies
│   └── ...
│
├── Final/                          # Final deliverables
│   ├── ensemble_optimised.ipynb    # Main optimised training notebook
│   ├── EDA graphs/                 # All EDA visualisations
│   └── submission_ensemble.csv     # Final Kaggle submission
│
├── LICENSE                         # MIT License
└── README.md                       # This file
```

**Branch structure:**
```
main              ← final optimised version
├── milestone-1   ← initial EDA + baseline CNN
├── milestone-2   ← AST integration
├── milestone-3   ← ensemble + ResNet
├── milestone-4   ← EfficientNet + optimisations
├── milestone-5   ← HuggingFace deployment
└── milestone-6   ← final submission
```

---

## 🚀 How to Run

### On Kaggle (Recommended — T4×2 GPU)

```python
# 1. Add the dataset to your Kaggle notebook
# Dataset: jan-2026-dl-gen-ai-project

# 2. Install dependencies
!pip install -q transformers torchmetrics wandb

# 3. Set environment variables
import os
os.environ['WANDB_API_KEY'] = 'your_wandb_key'

# 4. Run the notebook
# Final/ensemble_optimised.ipynb
```

### Locally

```bash
# Clone the repo
git clone https://github.com/22f3003226/DL_Messy_Mashup.git
cd DL_Messy_Mashup

# Install dependencies
pip install -r Scripts/requirements.txt

# Run inference only (load checkpoints)
python Scripts/inference.py --audio path/to/your/audio.wav
```

### Hugging Face Spaces

No setup needed — use the live demo directly:
> [INSERT YOUR HUGGING FACE SPACES LINK HERE]

---

## 📦 Dependencies

```
torch>=2.0.0
torchaudio>=2.0.0
torchvision>=0.15.0
transformers>=4.35.0
torchmetrics>=1.0.0
librosa>=0.10.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
wandb>=0.15.0
gradio>=4.0.0
tqdm>=4.65.0
```

---

## 🔬 Key Technical Concepts

| Concept | Used For |
|---------|---------|
| Mel Spectrogram | Converting raw waveform → 2D image for CNN input |
| Transfer Learning | AST (AudioSet), ResNet/EfficientNet (ImageNet) → Genre classification |
| Self-Attention | AST capturing global temporal audio patterns |
| Two-Phase Fine-Tuning | Preventing catastrophic forgetting in AST |
| Differential Learning Rates | Head LR 10× > backbone LR in AST Phase 2 |
| AMP / FP16 | Training speedup via T4 Tensor Cores |
| Label Smoothing | Preventing overconfidence, better calibration |
| TTA | Reducing prediction variance at inference time |
| Weighted Ensemble | Combining 4 model probability distributions |

---

## 🔮 Future Work

- [ ] **DistributedDataParallel (DDP)** instead of DataParallel — better multi-GPU scaling
- [ ] **Stacking ensemble** — train a meta-learner on validation predictions instead of fixed weights
- [ ] **SpecAugment** — frequency and time masking directly on spectrograms
- [ ] **Mixup augmentation** — blend two spectrogram samples with interpolated labels
- [ ] **Larger AST variant** — MIT AST with finer patch stride for higher resolution
- [ ] **Wav2Vec 2.0 / HuBERT** — raw waveform transformers as AST alternatives
- [ ] **Cross-validation** — more reliable val estimates with k-fold
- [ ] **Hyperparameter optimisation** — Bayesian search over LR, batch size, ensemble weights
- [ ] **ONNX export** — for faster CPU inference in the Hugging Face demo
- [ ] **Stem-aware ensemble** — train separate models per stem, combine at genre level

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- **IIT Madras** — Jan 2026 DLGenAI Course
- **MIT CSAIL** — [Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) (Gong et al., 2021)
- **Kaggle** — T4×2 GPU compute + competition platform
- **HuggingFace** — Model hosting and Spaces deployment
- **ESC-50 Dataset** — Piczak, K.J. (2015) — noise augmentation

---

<p align="center">
  Made with 🎵 by <b>Shibashish Banerjee</b> · 22f3003226 · IIT Madras
</p>
