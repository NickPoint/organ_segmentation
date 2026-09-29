"""DINOv3 ViT-S/16 encoder + smp's U-Net decoder, with the data pipeline of train_pernorm_25d.py.

Usage:
    python train_dino_pernorm_25d.py --output_dir ./results/dino2 --epochs 60
"""
import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose, DivisiblePad, RandAdjustContrastd, RandAffined, RandBiasFieldd, RandGridDistortiond,
    RandScaleIntensityd, RandShiftIntensityd,
)
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.losses import DiceLoss
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset

CLASSES = ['stomach', 'small_bowel', 'large_bowel']
MASK_H, MASK_W = 266, 266  # every labelled slice is 266x266
IMG_SIZE = 288  # 266 padded to a multiple of 32 = 18 patches of 16 px


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
    def __init__(self, df: pd.DataFrame, preprocess_input, scale_intensity, pad_fn, augment=None):
        super().__init__()
        self.seg_df = df.drop_duplicates('image_path').reset_index(drop=True)
        self.mask_lookup = build_mask_lookup(df)
        # (case, day, slice) -> path, to look up a slice's neighbours
        self.path_of = dict(zip(zip(df['case'], df['day'], df['slice']), df['image_path']))
        self.preprocess_input = preprocess_input
        self.scale_intensity = scale_intensity
        self.pad_fn = pad_fn
        self.augment = augment

    def __len__(self):
        return len(self.seg_df)

    def __getitem__(self, index):
        row = self.seg_df.iloc[index]
        case, day, sl = row['case'], row['day'], row['slice']

        # at the first / last slice of a scan the slice itself stands in for the missing neighbour
        previous_key = (case, day, sl - 1) if (case, day, sl - 1) in self.path_of else (case, day, sl)
        next_key = (case, day, sl + 1) if (case, day, sl + 1) in self.path_of else (case, day, sl)

        slices = []
        for key in [previous_key, (case, day, sl), next_key]:
            image = cv2.imread(self.path_of[key], cv2.IMREAD_UNCHANGED)
            slices.append(self.scale_intensity(image.astype(np.float32)))
        image_3ch = np.stack(slices, axis=-1)   # (H, W, 3): previous, current and next slice
        mask = self.mask_lookup[(case, day, sl)].astype(np.float32)   # (H, W, 3): one channel per organ

        # augment in 0-255, before ImageNet normalisation; MONAI is channel-first: (H, W, 3) -> (3, H, W) and back
        if self.augment is not None:
            augmented = self.augment({'image': image_3ch.transpose(2, 0, 1),
                                      'label': mask.transpose(2, 0, 1)})
            image_3ch = np.asarray(augmented['image']).transpose(1, 2, 0)
            mask = np.asarray(augmented['label']).transpose(1, 2, 0)

        norm_image = torch.from_numpy(self.preprocess_input(image_3ch).astype(np.float32)).permute(2, 0, 1)
        padded_image = self.pad_fn(norm_image).as_tensor()

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


def scale_intensity(image):
    """Map one image to 0-255 by its own 1st/99th percentile."""
    p1, p99 = np.percentile(image, [1, 99])
    return np.clip((image - p1) / max(p99 - p1, 1.0), 0.0, 1.0) * 255.0


def build_augmentation():
    """Applied to the 0-255 image and the 0/1 mask; spatial transforms warp both with the same
    parameters, nearest interpolation keeps the mask 0/1."""
    return Compose([
        # patient position
        RandAffined(keys=['image', 'label'], prob=0.7, rotate_range=0.26, scale_range=(0.1, 0.1),
                    translate_range=(20, 20), mode=['bilinear', 'nearest'], padding_mode='zeros',
                    shear_range=(0.26, 0.26)),
        # soft-tissue deformation
        RandGridDistortiond(keys=['image', 'label'], prob=0.3, distort_limit=0.2,
                            mode=['bilinear', 'nearest']),
        # scanner / session intensity
        RandScaleIntensityd(keys='image', factors=0.2, prob=0.5),
        RandShiftIntensityd(keys='image', offsets=15.0, prob=0.5),
        RandAdjustContrastd(keys='image', gamma=(0.7, 1.5), prob=0.3),
        # MRI coil bias field
        RandBiasFieldd(keys='image', prob=0.3),
    ])


# 0-255 -> ImageNet mean/std, as smp does for ResNet and as DINOv3 was trained.
# Module level, so DataLoader workers can pickle the dataset that holds it.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_input(image_3ch):
    return (image_3ch / 255.0 - IMAGENET_MEAN) / IMAGENET_STD


def build_transforms(train_df: pd.DataFrame, encoder_name: str, sample_size: int, seed: int):
    pad = DivisiblePad(k=32)
    return preprocess_input, scale_intensity, pad


class DinoUnet(nn.Module):
    """DINOv3 ViT-S/16 tokens -> 6-level feature pyramid -> smp's UnetDecoder."""

    def __init__(self, encoder_name: str, classes: int, freeze_encoder: bool):
        super().__init__()
        self.vit = timm.create_model(encoder_name, pretrained=True, img_size=IMG_SIZE, num_classes=0)
        self.embed_dim = self.vit.embed_dim                      # 384
        self.grid = IMG_SIZE // self.vit.patch_embed.patch_size[0]   # 288 / 16 = 18
        self.num_prefix_tokens = self.vit.num_prefix_tokens      # 5: 1 cls + 4 storage tokens
        self.block_indices = (2, 5, 8, 11)                       # early -> late, 12 blocks total

        if freeze_encoder:
            for p in self.vit.parameters():
                p.requires_grad = False

        # 1x1 conv: 384 -> C channels per token, then resized to its level's size.
        # BatchNorm evens out the few ViT channels that stay 10-100x larger than the rest (norm=True does not).
        self.proj_s4 = nn.Sequential(nn.Conv2d(self.embed_dim, 64, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(64))     # -> 72x72
        self.proj_s8 = nn.Sequential(nn.Conv2d(self.embed_dim, 128, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(128))    # -> 36x36
        self.proj_s16 = nn.Sequential(nn.Conv2d(self.embed_dim, 256, kernel_size=1, bias=False),
                                      nn.BatchNorm2d(256))   # -> 18x18 (native)
        self.proj_s32 = nn.Sequential(nn.Conv2d(self.embed_dim, 512, kernel_size=1, bias=False),
                                      nn.BatchNorm2d(512))   # -> 9x9

        # nothing finer than stride 16 in a ViT: level 0 is the image (UnetDecoder drops it, as for a ResNet),
        # level 1 a stride-2 stem, 3 -> 32 channels at 144x144
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.decoder = UnetDecoder(
            encoder_channels=(3, 32, 64, 128, 256, 512),
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
        )
        self.head = SegmentationHead(in_channels=16, out_channels=classes, kernel_size=3)

    def forward(self, x):
        assert x.shape[-2:] == (IMG_SIZE, IMG_SIZE), f'expected {IMG_SIZE}x{IMG_SIZE} input, got {tuple(x.shape[-2:])}'

        tokens_per_block = self.vit.forward_intermediates(
            x, indices=list(self.block_indices), norm=True, output_fmt='NLC', intermediates_only=True
        )

        spatial = []
        for tokens in tokens_per_block:
            # keep the 324 patch tokens (timm already drops the 5 prefix tokens in 'NLC' mode)
            tokens = tokens[:, tokens.shape[1] - self.grid * self.grid:, :]
            b, n, c = tokens.shape
            # (B, 324, 384) -> (B, 384, 18, 18): token i is the patch at row i // 18, column i % 18
            spatial.append(tokens.transpose(1, 2).reshape(b, c, self.grid, self.grid))

        f4 = F.interpolate(self.proj_s4(spatial[0]), size=(72, 72), mode='bilinear', align_corners=False)
        f8 = F.interpolate(self.proj_s8(spatial[1]), size=(36, 36), mode='bilinear', align_corners=False)
        f16 = self.proj_s16(spatial[2])                                   # already 18x18
        f32 = F.interpolate(self.proj_s32(spatial[3]), size=(9, 9), mode='bilinear', align_corners=False)

        # strides 1, 2, 4, 8, 16, 32 - the pyramid a ResNet encoder would give
        features = [x, self.stem(x), f4, f8, f16, f32]
        return self.head(self.decoder(features))


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
        axes[0].imshow(images[i, 1].cpu(), cmap='gray')
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
    parser.add_argument('--architecture', default='dino_unet', help='kept for compatibility; this script always builds DinoUnet')
    parser.add_argument('--encoder_name', default='vit_small_patch16_dinov3.lvd1689m', help='timm model name of the ViT encoder')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--fold', type=int, default=0, help='which of the n_splits folds is the validation set (0 ... n_splits-1)')
    parser.add_argument('--intensity_sample_size', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=Path, default=None, help='checkpoint to resume from')
    parser.add_argument('--freeze_encoder', action='store_true', help='train only the decoder, keep the DINOv3 weights fixed')
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

    train_dataset = SegDataset(new_train_df, preprocess_input, scale_intensity, pad,
                               augment=build_augmentation())
    val_dataset = SegDataset(val_df, preprocess_input, scale_intensity, pad)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    model = DinoUnet(args.encoder_name, classes=len(CLASSES), freeze_encoder=args.freeze_encoder).to(device)
    encoder_params = sum(p.numel() for p in model.vit.parameters())
    decoder_params = sum(p.numel() for p in model.parameters()) - encoder_params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Params: encoder {encoder_params/1e6:.1f}M, decoder+projections {decoder_params/1e6:.1f}M, '
          f'trainable {trainable_params/1e6:.1f}M', flush=True)

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
