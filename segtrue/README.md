# Grain-Boundary Preprocessing

Code and fixed data splits for the manuscript *"Conditional Benefits of Preprocessing Channels for Grain-Boundary Segmentation and Downstream Grain Quantification in Degraded Metallographic Images"*.

## Layout

```text
analysis/             IoU, grain measurements, aggregation
cli/                  training/evaluation entry point
data/                 loaders and fixed splits/*.json
degrade/              synthetic degradation definitions
engine/               training, evaluation, checkpoints
models/               U-Net and DeepLabV3+ with ResNet34
preprocess/           preprocessing channels and registry
provenance/           checkpoint identity manifest
utils/                seeding and file helpers
configs/              retained descriptive YAML records
pyproject.toml        installation, dependencies, command entry point
```

## Install

Requires Python >= 3.10. Install PyTorch/torchvision matching your CPU/CUDA environment, then:

```bash
pip install -e .
```

## Usage

Place the datasets under `<data_root>/{316l,tio}/{micrographs,masks}` (`tio` corresponds to TA2), then run:

```bash
segtrue train --dataset tio --arch unet_r34 --preprocess sobel_gabor --seed 42 --data-root ../datasets --device cuda --output outputs/ta2_sg_s42
segtrue evaluate --dataset tio --arch unet_r34 --preprocess sobel_gabor --seed 42 --data-root ../datasets --device cuda --checkpoint outputs/ta2_sg_s42/best.ckpt --output outputs/ta2_sg_s42_eval
```

Fixed train/validation/test split files for all seeds are provided in `data/splits/`.

## Data & License

- **316L**: [Kaggle (Artificial Grains and Real Grains, CC0: Public Domain)](https://www.kaggle.com/datasets/peterwarren/voronoi-artificial-grains-gen)
- **TA2**: [GitHub (MIT License)](https://github.com/xiamuyy114/TA2-Dataset-masks-micrographs)
- **Code**: MIT License (see `LICENSE` and `NOTICE`).