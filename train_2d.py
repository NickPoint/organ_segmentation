"""Train an smp segmentation model (ResNet encoder) on 2D input: each slice repeated to 3 channels.

Usage:
    python train_2d.py --architecture segformer --output_dir ./results/run1 --epochs 30
"""
import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from monai.metrics import DiceMetric
from monai.transforms import DivisiblePad, ScaleIntensityRange
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from segmentation_models_pytorch.losses import DiceLoss
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset

CLASSES = ['stomach', 'small_bowel', 'large_bowel']
MASK_H, MASK_W = 266, 266  # every labelled slice is 266x266


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rle_decode(mask_rle: str, shape: tuple[int, int]) -> np.ndarray:
    """'start length' pairs, 1-based over the row-major flattened image -> (H, W) 0/1 mask."""
    s = mask_rle.split()
    starts, lengths = (np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2]))
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


def build_mask_lookup(df: pd.DataFrame) -> dict:
    """{(case, day, slice): (H, W, 3) uint8 mask}, decoded once up front."""
    lookup = {}
    for (case, day, sl), sub in df.groupby(['case', 'day', 'slice']):
        mask = np.zeros((MASK_H, MASK_W, len(CLASSES)), dtype=np.uint8)
        for i, cls in enumerate(CLASSES):
            rle = sub.loc[sub['class'] == cls, 'segmentation'].values
            if len(rle) and not pd.isna(rle[0]):
                mask[..., i] = rle_decode(rle[0], shape=(MASK_H, MASK_W))
        lookup[(case, day, sl)] = mask
    return lookup


class SegDataset(Dataset):
    def __init__(self, df: pd.DataFrame, preprocess_input, scale_intensity, pad_fn):
        super().__init__()
        self.seg_df = df.drop_duplicates('image_path').reset_index(drop=True)
        self.mask_lookup = build_mask_lookup(df)
        self.preprocess_input = preprocess_input
        self.scale_intensity = scale_intensity
        self.pad_fn = pad_fn

    def __len__(self):
        return len(self.seg_df)

    def __getitem__(self, index):
        row = self.seg_df.iloc[index]

        image = cv2.imread(row['image_path'], cv2.IMREAD_UNCHANGED)
        scaled = self.scale_intensity(image.astype(np.float32))
        image_3ch = np.stack([scaled] * 3, axis=-1)

        norm_image = torch.from_numpy(self.preprocess_input(image_3ch).astype(np.float32)).permute(2, 0, 1)
        # MONAI returns a MetaTensor; CUDA conv fails on it ("unable to find an engine")
        padded_image = self.pad_fn(norm_image).as_tensor()

        mask = self.mask_lookup[(row['case'], row['day'], row['slice'])]
        norm_mask = torch.from_numpy(mask.astype(np.float32)).permute(2, 0, 1)
        padded_mask = self.pad_fn(norm_mask).as_tensor()

        return padded_image, padded_mask


def build_splits(data_dir: Path, n_splits: int, fold: int = 0):
    train_df = pd.read_csv(data_dir / 'train.csv')
    train_df['image_path'] = (
        str(data_dir) + '/train/'
        + 'case' + train_df['case'].astype(str)
        + '/day' + train_df['day'].astype(int).astype(str)
        + '/slice_' + train_df['slice'].astype(int).astype(str) + '.png'
    )

    # grouped by case: all scan days of a patient stay on one side
    group_kfold = GroupKFold(n_splits=n_splits)
    train_idx, val_idx = list(group_kfold.split(train_df, y=train_df['class'], groups=train_df['case']))[fold]
    new_train_df, val_df = train_df.iloc[train_idx], train_df.iloc[val_idx]
    assert not set(new_train_df['case']) & set(val_df['case']), 'a patient is in both train and validation'
    return new_train_df, val_df


def estimate_intensity_range(df: pd.DataFrame, sample_size: int, seed: int):
    sample_paths = df['image_path'].drop_duplicates().sample(
        n=min(sample_size, df['image_path'].nunique()), random_state=seed
    )
    sample_pixels = np.concatenate(
        [cv2.imread(p, cv2.IMREAD_UNCHANGED).ravel() for p in sample_paths]
    )
    p1, p99 = np.percentile(sample_pixels, [1, 99])
    return float(p1), float(p99)


def build_transforms(train_df: pd.DataFrame, encoder_name: str, sample_size: int, seed: int):
    p1, p99 = estimate_intensity_range(train_df, sample_size, seed)
    print(f'Intensity range from train sample: p1={p1:.1f}, p99={p99:.1f}', flush=True)
    scale_intensity = ScaleIntensityRange(a_min=p1, a_max=p99, b_min=0.0, b_max=255.0, clip=True)
    pad = DivisiblePad(k=32)
    preprocess_input = get_preprocessing_fn(encoder_name, pretrained='imagenet')
    return preprocess_input, scale_intensity, pad



def save_checkpoint(path: Path, epoch: int, model, optimizer, best_val_dice: float) -> None:
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'best_val_dice': best_val_dice,
    }, path)


def save_prediction_samples(model, dataloader, device, out_dir: Path, n_samples: int = 4) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    # the val loader is unshuffled and starts with empty edge slices: keep only slices with an organ
    collected_images, collected_masks = [], []
    for images_batch, masks_batch in dataloader:
        nonempty = masks_batch.flatten(1).any(dim=1)
        collected_images.append(images_batch[nonempty])
        collected_masks.append(masks_batch[nonempty])
        if sum(m.shape[0] for m in collected_masks) >= n_samples:
            break

    images = torch.cat(collected_images)[:n_samples].to(device)
    masks = torch.cat(collected_masks)[:n_samples]
    if images.shape[0] == 0:
        print('No non-empty masks found in val set for sample export; skipping.')
        return

    with torch.no_grad():
        preds = torch.sigmoid(model(images)) > 0.5

    for i in range(images.shape[0]):
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        axes[0].imshow(images[i, 0].cpu(), cmap='gray')
        axes[0].set_title('input')
        axes[1].imshow(masks[i].permute(1, 2, 0).cpu())
        axes[1].set_title('ground truth')
        axes[2].imshow(preds[i].permute(1, 2, 0).float().cpu())
        axes[2].set_title('prediction')
        for ax in axes:
            ax.axis('off')
        fig.tight_layout()
        fig.savefig(out_dir / f'sample_{i}.png')
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, default=Path('./hw3_dataset'))
    parser.add_argument('--output_dir', type=Path, default=Path('./results/run'))
    parser.add_argument('--architecture', default='unetplusplus', help="any smp architecture, e.g. unet, unetplusplus, segformer")
    parser.add_argument('--encoder_name', default='resnet34')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--fold', type=int, default=0, help='which of the n_splits folds is the validation set (0 ... n_splits-1)')
    parser.add_argument('--intensity_sample_size', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=Path, default=None, help='checkpoint to resume from')
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device('cpu')
    print('Device:', device, flush=True)

    new_train_df, val_df = build_splits(args.data_dir, args.n_splits, args.fold)
    val_cases = sorted(int(c) for c in val_df['case'].unique())
    print(f'Fold {args.fold}: validating on cases {val_cases}', flush=True)

    config: dict = {k: str(v) for k, v in vars(args).items()}
    config['val_cases'] = val_cases
    with open(args.output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    preprocess_input, scale_intensity, pad = build_transforms(
        new_train_df, args.encoder_name, args.intensity_sample_size, args.seed
    )

    train_dataset = SegDataset(new_train_df, preprocess_input, scale_intensity, pad)
    val_dataset = SegDataset(val_df, preprocess_input, scale_intensity, pad)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    model = smp.create_model(
        args.architecture, encoder_name=args.encoder_name, encoder_weights='imagenet', classes=len(CLASSES)
    ).to(device)
    # multilabel: an independent sigmoid per organ (organs can overlap, background has no channel)
    dice_loss = DiceLoss(mode='multilabel')
    bce_loss = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    dice_metric = DiceMetric(include_background=True, reduction='mean')

    start_epoch = 0
    best_val_dice = -1.0
    if args.resume is not None and args.resume.exists():
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_dice = checkpoint['best_val_dice']
        print(f'Resumed from {args.resume} at epoch {start_epoch}', flush=True)

    metrics_path = args.output_dir / 'metrics.csv'
    if start_epoch == 0:
        metrics_path.write_text('epoch,train_loss,val_loss,val_dice,seconds\n')

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        model.train()
        train_loss = 0.0
        for images_batch, gt_masks in train_dataloader:
            images_batch, gt_masks = images_batch.to(device), gt_masks.to(device)

            optimizer.zero_grad()
            predicted_mask = model(images_batch)
            loss_value = dice_loss(predicted_mask, gt_masks) + bce_loss(predicted_mask, gt_masks)
            loss_value.backward()
            optimizer.step()

            train_loss += loss_value.item()

        model.eval()
        val_loss = 0.0
        dice_metric.reset()
        with torch.no_grad():
            for images_batch, gt_masks in val_dataloader:
                images_batch, gt_masks = images_batch.to(device), gt_masks.to(device)

                predicted_mask = model(images_batch)
                val_loss += (dice_loss(predicted_mask, gt_masks) + bce_loss(predicted_mask, gt_masks)).item()

                preds_bin = (torch.sigmoid(predicted_mask) > 0.5).float()
                dice_metric(y_pred=preds_bin, y=gt_masks)

        train_loss /= len(train_dataloader)
        val_loss /= len(val_dataloader)
        val_dice = dice_metric.aggregate().item()
        elapsed = time.time() - t0

        print(
            f'epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} '
            f'val_dice={val_dice:.4f} ({elapsed:.1f}s)', flush=True
        )
        with open(metrics_path, 'a') as f:
            f.write(f'{epoch},{train_loss:.4f},{val_loss:.4f},{val_dice:.4f},{elapsed:.1f}\n')

        save_checkpoint(args.output_dir / 'checkpoint_last.pt', epoch, model, optimizer, best_val_dice)
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            save_checkpoint(args.output_dir / 'checkpoint_best.pt', epoch, model, optimizer, best_val_dice)

    save_prediction_samples(model, val_dataloader, device, args.output_dir / 'samples')
    print('Done.', flush=True)


if __name__ == '__main__':
    main()
