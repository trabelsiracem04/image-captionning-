# Image Captioning — Show, Attend and Tell

PyTorch implementation of
[*Show, Attend and Tell: Neural Image Caption Generation with Visual Attention*](https://arxiv.org/abs/1502.03044)
(Xu et al., ICLR 2015).

## Architecture

```
Image → CNN Encoder → Spatial Features → Attention → LSTM Decoder → Caption
```

A convolutional encoder extracts spatial feature vectors from the image.
At each decoding step, Bahdanau attention computes a weighted summary
of those features, which conditions the LSTM to predict the next word.

The paper introduces two attention variants: **soft** (deterministic) and
**hard** (stochastic). We implement **soft attention only**.

## Our Implementation vs the Paper

| Component | Paper (Xu et al.) | Our implementation | Rationale |
|---|---|---|---|
| Encoder CNN | VGGnet (14×14×512 =196 regions) | **ResNet50** (7×7×2048 = **49 regions**) | More modern architecture with residual connections |
| Spatial features | 196 annotation vectors | **49 spatial features** | Natural output of ResNet50 layer4 (7×7 grid) |
| Attention | Soft + Hard (stochastic, REINFORCE) | **Soft attention only** | Hard attention adds training complexity; soft is sufficient |
| Vocab size | Capped at 10,000 | **19,687** (min_freq=1, no cap) | Preserve rare words in Flickr30k |
| Output layer | Deep output: L_o(Ey + L_h h + L_z z) | **fc(dropout(h))** | Simpler, effective for our dataset size |
| BatchNorm | Not used | **Added on embedding + context** | Stabilizes training |
| Optimizer | Adam | **AdamW** | Decoupled weight decay, better generalization |
| Grad clipping | Not mentioned | **clip_grad_norm at 5.0** | Stabilizes LSTM training |
| BLEU evaluation | Without brevity penalty | **With brevity penalty** | Standard NLTK implementation |
| Doubly stochastic λ | Regularization term included | **Not implemented** | Optional, adds complexity |
| Xavier init | Not specified | **Used on attention layers** | Better initialization than default |
| LR scheduling | None (Adam adaptive only) | **ReduceLROnPlateau** | Monitors val BLEU, reduces on plateau |
| Early stopping | BLEU (likely BLEU-4) | **BLEU-1** | Faster feedback, simpler to track |

## Dataset

Flickr30k: 31,783 images × 5 captions each.

| Split | Images |
|---|---|
| Train | 29,783 |
| Val | 1,000 |
| Test | 1,000 |

Tokenization: Unicode NFKC normalization, curly-quote mapping, lowercasing.

## Project Structure

```
image_captionning/
├── configs/config.yaml          training & model config
├── src/
│   ├── models/
│   │   ├── encoder.py           ResNet50 CNN encoder
│   │   ├── attention.py         Bahdanau additive attention
│   │   ├── decoder.py           LSTM decoder with β-gating
│   │   └── caption_model.py     end-to-end model
│   ├── data/
│   │   ├── dataset.py           ImageCaptioningDataset
│   │   ├── vocabulary.py        word ↔ index mapping
│   │   ├── tokenizer.py         text normalization & tokenization
│   │   ├── transforms.py        image preprocessing
│   │   └── split.py             train/val/test splitting
│   ├── training/
│   │   ├── trainer.py           training loop with early stopping
│   │   ├── losses.py            masked cross-entropy loss
│   │   ├── validate.py          validation + BLEU evaluation
│   │   ├── checkpoints.py       save/load/resume
│   │   └── logging.py           file + console logging
│   ├── inference/
│   │   ├── greedy.py            greedy search decoding
│   │   └── beam_search.py       beam search decoding
│   ├── evaluation/
│   │   └── metrics.py           BLEU-1..4 scores
│   └── utils/
│       ├── config.py            YAML config + env overrides
│       ├── device.py            GPU/CPU detection
│       └── seeds.py             reproducibility
├── tests/                       97 unit tests
├── scripts/
│   ├── smoke_stage1.py          GPU smoke gate
│   └── make_kaggle_bundle.py    Kaggle data + code bundles
├── kaggle/
│   ├── stage1_training.py       notebook runner
│   └── README.md                Kaggle deployment guide
└── train.py                     main CLI entry point
```

## Model

### Encoder

- ResNet50 pretrained on ImageNet (IMAGENET1K_V2)
- Extracts spatial features from layer4: `[B, 3, 224, 224]` → `[B, 49, 2048]`
- Stage 1: frozen. Stage 2: unfreeze layer4

### Attention

Bahdanau additive (soft attention):

```
e_ti = v^T tanh(W_a a_i + U_a h_{t-1})
α_ti = softmax(e_ti)
context = Σ α_ti * a_i
```

Xavier uniform initialization on all attention layers.

### Decoder

- LSTMCell, input = `[embedding(word), context_bn(context)]`
- β gating: `β_t = sigmoid(f_β(h_{t-1}))`, context = β × context
- Init: h_0, c_0 from mean spatial features via linear layers
- Output: `fc(dropout(h))` → vocab logits
- ~24.3M parameters total

## Training

| Param | Value |
|---|---|
| Batch size | 32 |
| Epochs | 10 |
| LR (decoder) | 1e-3 |
| LR (CNN, Stage 2) | 1e-5 |
| Optimizer | AdamW |
| Grad clip | 5.0 |
| Dropout | 0.5 |
| Early stopping | patience 3, monitor val BLEU-1 |
| Checkpoints | `best.pt` (best val BLEU), `last.pt` |

### Two-stage training

1. **Stage 1** — Encoder frozen, train decoder + attention only
2. **Stage 2** — Unfreeze layer4, lower LR for encoder

## Quickstart (Local)

```bash
# 1. Install
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. Place Flickr30k data
#    data/Images/     ← 31,783 .jpg files
#    data/captions.txt

# 3. Train
python train.py --config configs/config.yaml --epochs 10 --tag stage1

# 4. Checkpoints saved to checkpoints/
```

## Kaggle

```bash
# Build bundles
python scripts/make_kaggle_bundle.py --code
python scripts/make_kaggle_bundle.py --data
python scripts/make_kaggle_bundle.py --data --upload-data

# Upload data to Kaggle
# Create notebook, paste kaggle/stage1_training.py
# Set GPU: T4 x2, Internet: off
```

See [`kaggle/README.md`](kaggle/README.md) for full instructions.

## Verification

- **97/97** unit tests passing (`pytest tests/ -q`)
- **GPU smoke gate**: 100 images, 2 epochs, loss decreasing, BLEU improving
