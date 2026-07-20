# ApiSacoustic — Bee Hive Sound Classification

Classify bee hive audio into three health states — **Active**, **Queenless**, **Infested** — using deep learning and classical ML models. Designed for deployment on **Raspberry Pi 4** with LINE Messaging API alerts.

## Dataset

- **FRDR (urban-frdr):** ~392 audio chunks from Canadian agricultural hives, extracted from `.tar.gz` archives via `extract-tar.ipynb`. Annotations from hive inspections (2021–2022) plus sensor/weather data.
- **Zenodo:** 5 hives (Hive1, Hive3) recorded 2017–2018, with state labels.

**Class distribution:** Active ~147, Queenless ~83, Infested ~162 (imbalanced; handled via `WeightedRandomSampler`).

## Pipeline

```
Raw audio (16 kHz, 30 s windows) → Mel-spectrogram (128 mel, 224×224) → Normalization (ImageNet stats) → Model → Classification
```

**Augmentations:** Frequency masking, time masking, noise injection. Optional **synthetic** bee sound augmentation to boost Infested recall.

## Notebooks

| Notebook | Purpose |
|---|---|
| `EDAapisacoustic.ipynb` | EDA, class distributions, mel-spectrogram visualization |
| `extract-tar.ipynb` | Extract & organize FRDR tar archives |
| `apisacoustic-synthetic.ipynb` | Training with synthetic data augmentation (best results) |
| `apisacoustic_model_comparison.ipynb` | Multi-model benchmark (20 epochs) |
| `apisacoustic-baseline-svm.ipynb` | SVM baseline (MFCC + spectral features) |
| `noise-cancel-lab.ipynb` | Audio noise reduction experimentation |

## Models

All deep models use `timm` architectures with a custom classifier head (3 classes):

| Model | Test Acc | Test Macro F1 | Test ROC-AUC |
|---|---|---|---|
| **ConvNeXt Tiny** | **99.4%** | **0.989** | **0.999** |
| **ViT Tiny** | 99.4% | 0.988 | 0.999 |
| MobileNetV3 Small | 99.2% | 0.987 | 0.998 |
| RegNetY-004 | 98.8% | 0.980 | 0.999 |
| MobileNetV3 Large | 98.6% | 0.976 | 0.999 |
| ResNet18 | 98.2% | 0.970 | 0.999 |
| EfficientNet-B0 | 98.0% | 0.971 | 0.994 |
| VGG11-BN | 89.2% | 0.875 | 0.974 |

**SVM baseline** (MFCC + RBF kernel): 92% test accuracy, macro F1 0.91.

## Raspberry Pi Deployment

`rasp.py` provides a full CLI/menu-driven classifier for **Raspberry Pi 4**:

- Records via USB microphone (`arecord`/`parecord`)
- Rolling 32-second window, classified every 8 seconds
- Sends **LINE Messaging API** alerts on 8 consecutive Queenless/Infested detections
- Interactive menu: monitor, single classification, view stats, configure LINE token

### Usage

```bash
# Interactive menu
python rasp.py --menu

# Continuous monitoring
python rasp.py --monitor

# Single classification
python rasp.py --single
```

## Requirements

Python 3.12, PyTorch 2.12+, torchaudio, timm, librosa, scikit-learn, etc. Install via `uv` or `pip`:

```bash
pip install -r requirements.txt
```

## Results

Best models saved under `models/`. Confusion matrices and full comparison table in `results/`. ConvNeXt Tiny with synthetic data achieves near-perfect classification (99.4% accuracy, 0.999 ROC-AUC).

## License

Academic research project.
