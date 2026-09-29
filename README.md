# Organ segmentation in abdominal MRI

Multilabel segmentation of stomach, small bowel and large bowel, first with CNN / SegFormer baselines, then with
a DINOv3 ViT-S/16 encoder. Every configuration is evaluated with 5-fold cross-validation grouped by patient.

## Results

Mean validation Dice over the 5 folds:

| model | input | augmentation | intensity scaling | Dice |
|---|---|---|---|---|
| U-Net++ | 2D | no | global | 0.571 |
| SegFormer | 2D | no | global | 0.585 |
| U-Net++ | 2.5D | no | global | 0.592 |
| SegFormer | 2.5D | yes | global | 0.599 |
| SegFormer | 2.5D | no | global | 0.614 |
| SegFormer | 2.5D | yes | global | 0.636 |
| SegFormer | 2.5D | yes | per image | 0.693 |
| DINOv3 U-Net | 2.5D | yes | per image | 0.695 |

The analysis behind these numbers, the DINOv3 feature and attention study and the conclusions are in
`organ_segmentation.ipynb`.

## Files

| file | contents |
|---|---|
| `organ_segmentation.ipynb` | data analysis, all experiments, results and the DINOv3 study |
| `sumbission_example.ipynb` | Kaggle submission: 5-fold ensemble over the test set |
| `train_2d.py` | SegFormer / U-Net++ with a ResNet-34 encoder, 2D input |
| `train_25d.py` | the same with 2.5D input (previous, current and next slice) |
| `train_aug_25d.py` | 2.5D + augmentation |
| `train_pernorm_25d.py` | 2.5D + augmentation + per-image intensity scaling |
| `train_dino_pernorm_25d.py` | DINOv3 encoder + U-Net decoder, same data pipeline as `train_pernorm_25d.py` |
| `dino_attention.py` | DINOv3 attention maps, written to `results/dino_attention/` |
| `train.job`, `deploy.sh`, `deploy.ps1` | Slurm job and scripts to submit it |
| `results/<run>/fold<k>/` | `config.json`, `metrics.csv` and prediction samples of every run |

## Setup

```
pip install -r requirements.txt
```

```
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu126
```

The dataset is not included, put it at `./hw3_dataset`.

## Training

One fold locally:

```
python train_pernorm_25d.py --architecture segformer --fold 0 --epochs 60 --output_dir results/run/fold0
```

All 5 folds on the cluster, one Slurm array task per fold:

```
./deploy.sh submit train_pernorm_25d.py segformer
```

## Model weights

The weights of the two final models (SegFormer per-image and DINOv3, 5 folds each) are attached to the repository's release. Each file is a checkpoint without optimizer state. To use them with the notebooks, put each file back where the training run wrote it:

```
for f in *_fold*.pt; do
    run=${f%_fold*}; k=${f##*_fold}; k=${k%.pt}
    mkdir -p results/$run/fold$k && mv "$f" results/$run/fold$k/checkpoint_best.pt
done
```
