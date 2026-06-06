# Ultralytics YOLO Custom Project

This repository contains a customized Ultralytics YOLO project with added model modules and experiment configurations.

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py
```

The default training script uses `SARDet-100K.yaml` and writes outputs to `runs/train`.

## Notes

- Datasets are not included in this repository.
- Training outputs and model weights are ignored by Git.
- Update `SARDet-100K.yaml` to match your local dataset path before training.
