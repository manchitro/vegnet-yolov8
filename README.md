# Cauli-Det: Modified YOLOv8 for Cauliflower Disease Detection

A fine-tuned, architecturally modified YOLOv8 model for detecting and localizing three cauliflower diseases from smartphone-captured field images. Published in *Frontiers in Plant Science* (2024), 41 citations.

📄 [Paper (DOI: 10.3389/fpls.2024.1373590)](https://doi.org/10.3389/fpls.2024.1373590) · 🎓 [Google Scholar](https://scholar.google.com/citations?user=M2eF33AAAAAJ&hl=en)

> This repo is a fork of [`ultralytics/ultralytics`](https://github.com/ultralytics/ultralytics). Everything below documents what was actually built on top of it — the base YOLOv8 documentation has been moved out of the way.

## Results

| Metric | Value |
|---|---|
| Precision | 93.2% |
| Recall | 82.6% |
| mAP50 | 91.1% |
| mAP50-95 | 70.1% |
| Parameters | 11.32M |

Final configuration: **YOLOv8s backbone + 3 extra Conv blocks in the detection/classification heads + Hard Swish activation + full (unfrozen) fine-tuning**. Detects three disease classes — Downy Mildew, Black Rot, Bacterial Spot Rot — plus healthy plants.

## Problem

Cauliflower crops are vulnerable to several diseases that are hard for smallholder farmers to identify early using the naked eye. This project builds a detector that runs on ordinary smartphone photos, aimed at low-cost, accessible disease triage in the field rather than lab-grade imaging equipment.

## What Was Modified

Starting from stock YOLOv8, this project makes four deliberate, empirically-tested changes (see Experiments below for why each one was chosen):

1. **+3 extra Conv blocks** (kernel size 1) inserted before the output convolutional layer in the detection/classification heads — adds depth without much parameter growth.
2. **Hard Swish activation**, replacing the default SiLU.
3. **Full fine-tuning** — all layers unfrozen, uniform learning rate (backbone freezing was tested and made things dramatically worse — see below).
4. **Custom dataset integration** — a purpose-built annotation pipeline for the VegNet cauliflower disease dataset.

Custom files in this repo (everything else is upstream Ultralytics): `vegnet_training.py`, `vegnet_val.py`, `vegnet_yolo.yaml`.

## Dataset

- **Source:** VegNet dataset, 656 field images captured in Bangladesh (Dec 20, 2021 – Jan 15, 2022) with a Sony Cyber-Shot W-530 (14MP).
- **Classes:** Downy Mildew (177 images), Bacterial Spot Rot (173), Healthy (206), Black Rot (100).
- **Split:** 70% / 15% / 15% train / val / test (460 / 98 / 98 images).
- **Preprocessing:** resized to 256×256; brightness, contrast, hue, and saturation adjusted. Bounding boxes hand-annotated via [Makesense.ai](https://www.makesense.ai/).

## Experiments & Ablations

The final configuration above wasn't a first guess — four separate design questions were tested empirically before settling on it. This is the part that doesn't show up in the code (only the winning configuration is what's currently checked in), so it's documented here instead.

### 1. Base model: which YOLO to start from?

| Model | Precision | Recall | mAP50 | Params |
|---|---|---|---|---|
| YOLOv7 | 97.8% | 88.9% | 92.6% | 37.21M |
| YOLOv8s | 91.4% | 83.2% | 84.1% | 11.14M |
| YOLOv8m | 91.2% | 86.8% | 91.6% | 25.86M |

**Takeaway:** YOLOv8s doesn't top this table — YOLOv7 and YOLOv8m both score higher on raw mAP. It was chosen anyway for its size/accuracy tradeoff: less than a third the parameters of YOLOv7, and the gap closes substantially once the head modifications below are applied. Justified by the deployability goal (smallholder-accessible devices), not by chasing the leaderboard number.

### 2. How many extra Conv blocks in the head?

| Configuration | Precision | Recall | mAP50 | Params |
|---|---|---|---|---|
| YOLOv8s (base) | 91.4% | 83.2% | 84.1% | 11.14M |
| +1 Conv block | 95.5% | 83.7% | 90.5% | 11.20M |
| +3 Conv blocks | 93.1% | 82.9% | 90.6% | 11.32M |
| +5 Conv blocks | 94.6% | 85.7% | 90.4% | 11.45M |

**Takeaway:** +3 blocks hit the sweet spot on mAP50 for minimal added parameters; +5 blocks added more parameters without a corresponding gain. Diminishing returns past +3.

### 3. Learning rate strategy: freeze the backbone or not?

| Strategy | Precision | Recall | mAP50 |
|---|---|---|---|
| Default (uniform, unfrozen) | 93.1% | 82.9% | 90.6% |
| Freeze backbone | 56.6% | 46.1% | 51.1% |
| Fast extra-Conv | 90.6% | 76.5% | 83.9% |
| Fast head-neck | 95.5% | 83.7% | 90.5% |

**Takeaway:** freezing the backbone collapsed performance (mAP50 dropped nearly 40 points) — the pretrained COCO features weren't a good enough starting point on their own for this domain shift. Full fine-tuning was necessary, not optional. (The commented-out frozen-layers code path still in `vegnet_training.py` is a remnant of this experiment.)

### 4. Activation function

| Function | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| SiLU (default) | 93.1% | 82.9% | 90.6% | 69.4% |
| ReLU | 90.6% | 82.9% | 87.5% | 66.8% |
| Hard Swish | 93.2% | 82.6% | 91.1% | 70.1% |

**Takeaway:** Hard Swish won on both accuracy and computational efficiency, so it became the final choice over the framework default.

## Training Setup

| | |
|---|---|
| Framework | PyTorch 2.1.0, CUDA 11.1 |
| Hardware | Tesla T4 GPU (16GB), Intel Xeon CPU, 12.7GB RAM |
| Optimizer | AdamW, lr0 = 0.001429, momentum = 0.9 |
| Loss | Varifocal (classification) + CIoU (box regression) + DFL |
| Epochs | 200 (early stopping patience 50) |
| Batch size | 32 |
| Image size | 256×256 |
| Pretrained weights | COCO |

## How to Run

> The dataset paths below (`vegnet_yolo.yaml`, `vegnet_training.py`) are currently hardcoded to a Google Colab / Google Drive layout (`/content/...`, `/content/drive/MyDrive/...`). This works if you're running in Colab with the dataset mounted at that path; for local use you'd need to edit `vegnet_yolo.yaml`'s `path:` field and `vegnet_training.py`'s output path first. Flagging this as a portability gap rather than fixing it silently — confirm before changing paths that existing results were generated against.

```python
from vegnet_training import vegnet_training
from vegnet_val import vegnet_val

# Train
vegnet_training(model_name="yolov8s", exp_name="original_custom_head")

# Validate / get metrics (mAP50, mAP50-95, mAP75, per-class mAP)
metrics = vegnet_val(model_path="path/to/best.pt", split="test")
```

Dataset config (`vegnet_yolo.yaml`) expects three classes — Downey Mildew, Black Rot, Bacterial Spot Rot — laid out as `images/train` and `images/val` under the configured root.

## Citation

```bibtex
@article{uddin2024caulidet,
  title   = {Cauli-Det: enhancing cauliflower disease detection with modified YOLOv8},
  author  = {Uddin, Md. Sazid and Mazumder, Md. Khairul Alam and Prity, Afrina Jannat and Mridha, M. F. and Alfarhood, Sultan and Safran, Mejdl and Che, Dunren},
  journal = {Frontiers in Plant Science},
  volume  = {15},
  year    = {2024},
  doi     = {10.3389/fpls.2024.1373590}
}
```

## License

**AGPL-3.0**, inherited from upstream [`ultralytics/ultralytics`](https://github.com/ultralytics/ultralytics). This repo builds substantially on Ultralytics' AGPL-3.0-licensed codebase, so AGPL-3.0 applies to the combined work — a copyleft requirement, not a choice made for this project specifically. Ultralytics also offers a separate Enterprise license for closed-source commercial use; see [ultralytics.com/license](https://www.ultralytics.com/license) for details.
