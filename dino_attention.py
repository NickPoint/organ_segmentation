"""Explore and quantify DINOv3 self-attention on the abdominal-MRI slices of this project.

Model: timm 'vit_small_patch16_dinov3.lvd1689m' (frozen, pretrained on natural images only,
no medical fine-tuning), run at 288x288 -> 18x18 = 324 patch tokens + 5 prefix tokens
(1 CLS + 4 DINOv3 "storage"/register tokens).

timm's EvaAttention uses fused attention, which never materialises the attention matrix.
We therefore hook every block's attention module and recompute the softmax matrix from the
same qkv projection and the same rotary position embedding the module used. The hook asserts
that (attn @ v) -> norm -> proj reproduces the module's real output, so the recomputed
matrices are the ones the network actually used.

What it writes into results/dino_attention/:
  heads_last_block_*.png   CLS->patch attention of each of the 6 heads of block 11
  depth_*.png              CLS->patch attention (head mean) at blocks 0, 5, 11
  rollout_all_slices.png   attention rollout (all 12 blocks) for every example slice
  quantitative_*.png       region enrichment / boundary-vs-interior / per-head padding bars
  quantitative_report.txt  all the numbers, averaged over the sampled slices

Caveats that the numbers have to be read with:
  * The 288x288 input is a 266x266 MRI slice plus 11 px of padding on each side
    (DivisiblePad k=32). The padding is applied AFTER ImageNet normalisation, so the border
    is a constant 0 in normalised space - an artificial, perfectly uniform frame that covers
    ~14.5% of the input area and ~0.7 patch on each edge.
  * A 16x16 patch is coarse for these organs: an organ spans only ~6-10 patches and the
    average patch that touches an organ is only about a third organ. Patch-level attention
    can therefore not resolve organ boundaries finely.

Run:  python dino_attention.py
"""
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn.functional as F
from timm.layers import apply_rot_embed_cat

import train_25d

CLASSES = ['stomach', 'small_bowel', 'large_bowel']
OUT_DIR = Path('results/dino_attention')
PAD = 11          # DivisiblePad k=32 on 266 -> 288: 11 px each side
PATCH = 16
GRID = 18         # 288 / 16
N_PATCHES = GRID * GRID
N_PREFIX = 5      # 1 CLS + 4 storage tokens
N_SLICES_PER_CASE = 9   # -> ~108 slices for the quantitative part

OUT_DIR.mkdir(parents=True, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----------------------------------------------------------------------------- model + hooks
vit = timm.create_model('vit_small_patch16_dinov3.lvd1689m', pretrained=True,
                        img_size=288, num_classes=0).eval().to(device)
attn_store = {}   # block index -> (6, 329, 329) attention probabilities
recompute_error = []   # relative error of the recomputed attention output vs the module's own output


def make_attention_hook(block_index):
    """Recompute the attention matrix of one EvaAttention module (fused attn hides it)."""
    def hook(module, args, kwargs, output):
        x = args[0]
        rope = kwargs.get('rope')
        B, N, C = x.shape
        if module.q_bias is None:
            qkv = module.qkv(x)
        else:
            qkv_bias = torch.cat((module.q_bias, module.k_bias, module.v_bias))
            qkv = F.linear(x, weight=module.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = module.q_norm(q), module.k_norm(k)
        if rope is not None:
            npt = module.num_prefix_tokens
            half = getattr(module, 'rotate_half', False)
            q = torch.cat([q[:, :, :npt], apply_rot_embed_cat(q[:, :, npt:], rope, half=half)], dim=2).type_as(v)
            k = torch.cat([k[:, :, :npt], apply_rot_embed_cat(k[:, :, npt:], rope, half=half)], dim=2).type_as(v)
        attn = ((q * module.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
        attn_store[block_index] = attn[0].detach().float().cpu().numpy()

        # sanity check: our recomputation is the one the module really used
        y = (attn @ v).transpose(1, 2).reshape(B, N, C)
        y = module.proj(module.norm(y))
        error = (y - output).abs().max().item() / output.abs().max().item()
        recompute_error.append(error)
        assert error < 0.02, f'block {block_index}: attention recompute mismatch (rel. error {error:.3g})'
    return hook


for i, blk in enumerate(vit.blocks):
    blk.attn.register_forward_hook(make_attention_hook(i), with_kwargs=True)


def attention_of_slice(image):
    """image: (3,288,288) tensor from SegDataset. Returns list of 12 arrays (6,329,329)."""
    attn_store.clear()
    with torch.no_grad():
        vit.forward_features(image[None].to(device))
    return [attn_store[i] for i in range(len(vit.blocks))]


# ----------------------------------------------------------------------------- data
tr, va = train_25d.build_splits(Path('hw3_dataset'), 5)
prep, scale_intensity, pad_fn = train_25d.build_transforms(tr, 'resnet34', 100, 42)
ds = train_25d.SegDataset(tr, prep, scale_intensity, pad_fn)

# every slice that actually contains an organ, grouped by patient
candidates = []   # (index, case, day, slice, area_stomach, area_small, area_large, n_small_blobs)
for i in range(len(ds)):
    row = ds.seg_df.iloc[i]
    key = (row['case'], row['day'], row['slice'])
    mask = ds.mask_lookup[key]
    if not mask.any():
        continue
    n_blobs = cv2.connectedComponents(mask[..., 1])[0] - 1
    candidates.append((i, int(row['case']), int(row['day']), int(row['slice']),
                       int(mask[..., 0].sum()), int(mask[..., 1].sum()), int(mask[..., 2].sum()), n_blobs))
cases = sorted({c[1] for c in candidates})
print(f'{len(candidates)} organ-containing slices in {len(cases)} patients: {cases}', flush=True)


def load_slice(index):
    """Returns (model input tensor, display image 288x288 in 0..255, masks (3,288,288) 0/1)."""
    image, mask = ds[index]
    row = ds.seg_df.iloc[index]
    raw = cv2.imread(ds.path_of[(row['case'], row['day'], row['slice'])], cv2.IMREAD_UNCHANGED)
    display = np.pad(np.asarray(scale_intensity(raw.astype(np.float32))), PAD)
    assert display.shape == (288, 288) and image.shape == (3, 288, 288)
    return image, display, mask.numpy().astype(np.uint8)


def patch_coverage(pixel_map):
    """Fraction of each 16x16 patch covered by a binary 288x288 map -> (18,18)."""
    return pixel_map.reshape(GRID, PATCH, GRID, PATCH).mean(axis=(1, 3))


def region_maps(display, mask):
    """Disjoint pixel-level regions of the 288x288 input, as patch-coverage maps (18,18).

    padding      the DivisiblePad frame (constant, non-anatomical)
    stomach / small_bowel / large_bowel   ground truth (overlaps resolved in this order)
    body         remaining bright tissue inside the real 266x266 field of view
    air          remaining dark voxels inside the field of view (outside-the-patient air,
                 bowel gas and other signal voids)
    """
    inside = np.zeros((288, 288), bool)
    inside[PAD:288 - PAD, PAD:288 - PAD] = True

    p99 = np.percentile(display[inside], 99)
    bright = (display >= 0.10 * max(p99, 1e-6)) & inside
    bright = cv2.morphologyEx(bright.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) > 0

    taken = ~inside            # padding is taken first
    regions = {'padding': ~inside}
    for c, name in enumerate(CLASSES):
        organ = (mask[c] > 0) & inside & ~taken
        regions[name] = organ
        taken = taken | organ
    regions['body'] = bright & ~taken
    taken = taken | regions['body']
    regions['air'] = inside & ~taken

    return {name: patch_coverage(m.astype(np.float32)) for name, m in regions.items()}


def cls_patch_attention(attn, head=None):
    """CLS->patch attention, renormalised over the 324 patches (drops the 5 prefix tokens).

    attn: (6,329,329). head=None averages the heads. Returns (18,18) summing to 1.
    """
    row = attn[:, 0, N_PREFIX:] if head is None else attn[head:head + 1, 0, N_PREFIX:]
    a = row.mean(axis=0)
    return (a / a.sum()).reshape(GRID, GRID)


def rollout(attn_per_block):
    """Attention rollout: A_hat = 0.5*A + 0.5*I, row-normalised, multiplied across blocks."""
    n = attn_per_block[0].shape[-1]
    result = np.eye(n, dtype=np.float64)
    for attn in attn_per_block:
        a = attn.mean(axis=0).astype(np.float64)          # average the heads
        a = 0.5 * a + 0.5 * np.eye(n)
        a = a / a.sum(axis=-1, keepdims=True)
        result = a @ result
    a = result[0, N_PREFIX:]
    return (a / a.sum()).reshape(GRID, GRID)


def upsample(attention_map):
    return cv2.resize(attention_map, (288, 288), interpolation=cv2.INTER_LINEAR)


def show_slice(ax, display, mask, title):
    ax.imshow(display, cmap='gray')
    for c, color in zip(range(3), ['#4ec9ff', '#ffd166', '#ff5d8f']):
        if mask[c].any():
            ax.contour(mask[c], levels=[0.5], colors=color, linewidths=1.2)
    ax.add_patch(plt.Rectangle((PAD - 0.5, PAD - 0.5), 266, 266, fill=False,
                               ec='#7cff7c', lw=0.8, ls='--'))
    ax.set_title(title, fontsize=9)
    ax.axis('off')


def show_attention(ax, display, mask, attention_map, title):
    ax.imshow(display, cmap='gray')
    ax.imshow(upsample(attention_map), cmap='inferno', alpha=0.55,
              vmin=0, vmax=attention_map.max())
    for c, color in zip(range(3), ['#4ec9ff', '#7cff7c', '#ffffff']):
        if mask[c].any():
            ax.contour(mask[c], levels=[0.5], colors=color, linewidths=1.0)
    ax.set_title(title, fontsize=9)
    ax.axis('off')


# ----------------------------------------------------------------------------- example slices
# picked by rule so the figures are reproducible: a big stomach, a big large bowel,
# the most scattered small bowel, and the biggest small bowel - each from a different patient.
by_stomach = sorted(candidates, key=lambda c: -c[4])
by_large = sorted(candidates, key=lambda c: -c[6])
by_small = sorted(candidates, key=lambda c: -c[5])
scattered = sorted([c for c in candidates if 300 <= c[5] <= 1500], key=lambda c: -c[7])

examples = []
used_cases = set()
for pick, label in [(by_stomach, 'large stomach'), (by_large, 'large large-bowel'),
                    (scattered, 'small scattered small-bowel'), (by_small, 'large small-bowel')]:
    for c in pick:
        if c[1] not in used_cases:
            examples.append((c, label))
            used_cases.add(c[1])
            break

print('example slices:')
for c, label in examples:
    print(f'  case{c[1]} day{c[2]} slice{c[3]}  ({label}); areas stomach={c[4]} small={c[5]} large={c[6]} '
          f'small-bowel blobs={c[7]}', flush=True)

# ----------------------------------------------------------------------------- figures
rollout_maps = []
for c, label in examples:
    index, case, day, sl = c[0], c[1], c[2], c[3]
    tag = f'case{case}_day{day}_slice{sl}'
    image, display, mask = load_slice(index)
    attn_per_block = attention_of_slice(image)

    # (1) the six heads of the last block
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.4))
    show_slice(axes[0, 0], display, mask, f'{tag}\n{label}\nstomach=blue small=yellow large=pink')
    axes[0, 1].axis('off')
    axes[0, 1].text(0.0, 0.5, 'DINOv3 ViT-S/16 (frozen)\nblock 11, CLS -> patch attention\n'
                              'each head normalised to its own max\ngreen dashed box = real 266x266 image,\n'
                              'outside it is DivisiblePad padding',
                    fontsize=9, va='center')
    for h in range(6):
        ax = axes.flat[2 + h]
        a = cls_patch_attention(attn_per_block[11], head=h)
        pad_share = float((a * region_maps(display, mask)['padding']).sum())
        show_attention(ax, display, mask, a, f'head {h}   padding share {pad_share:.2f}')
    fig.suptitle(f'Last-block CLS attention per head - {tag}', fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f'heads_last_block_{tag}.png', dpi=130)
    plt.close(fig)

    # (2) depth: early / middle / late block
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    show_slice(axes[0], display, mask, f'{tag}\n{label}')
    for ax, b in zip(axes[1:], [0, 5, 11]):
        show_attention(ax, display, mask, cls_patch_attention(attn_per_block[b]),
                       f'block {b} (head mean)')
    fig.suptitle(f'CLS attention with depth - {tag}', fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f'depth_{tag}.png', dpi=130)
    plt.close(fig)

    rollout_maps.append((tag, label, display, mask, rollout(attn_per_block)))
    print(f'figures done for {tag}', flush=True)

# (3) rollout for all example slices in one figure
fig, axes = plt.subplots(len(rollout_maps), 2, figsize=(7.5, 3.7 * len(rollout_maps)))
for r, (tag, label, display, mask, roll) in enumerate(rollout_maps):
    show_slice(axes[r, 0], display, mask, f'{tag}\n{label}')
    show_attention(axes[r, 1], display, mask, roll, 'attention rollout (12 blocks)')
fig.suptitle('Attention rollout, A_hat = 0.5A + 0.5I', fontsize=11)
fig.tight_layout()
fig.savefig(OUT_DIR / 'rollout_all_slices.png', dpi=130)
plt.close(fig)

# ----------------------------------------------------------------------------- quantitative
# evenly spaced organ-containing slices from every patient
sample = []
for case in cases:
    of_case = [c for c in candidates if c[1] == case]
    for j in np.linspace(0, len(of_case) - 1, min(N_SLICES_PER_CASE, len(of_case))).astype(int):
        sample.append(of_case[j])
print(f'\nquantitative pass over {len(sample)} slices', flush=True)

region_names = ['padding', 'stomach', 'small_bowel', 'large_bowel', 'body', 'air']
share_cls = {name: [] for name in region_names}       # attention share, last block, head mean
share_roll = {name: [] for name in region_names}      # attention share, rollout
area = {name: [] for name in region_names}            # area share of the same region

head_pad_share = [[] for _ in range(6)]               # per head of block 11
head_organ_share = [[] for _ in range(6)]
depth_organ_enrich = [[] for _ in range(12)]          # head-mean CLS organ enrichment per block
depth_pad_enrich = [[] for _ in range(12)]
prefix_mass = [[] for _ in range(12)]                 # CLS mass on cls+4 storage tokens

within_tissue_enrich = []   # organ enrichment measured against body tissue only (no air, no padding)
top5_region_cls = {name: 0 for name in ['padding', 'stomach', 'small_bowel', 'large_bowel', 'body', 'air']}
top5_region_roll = dict(top5_region_cls)   # where the 5 strongest patches of each map land

# boundary/interior with two patch-level definitions of "organ patch": a patch counts as organ
# if >=50% (strict) or >=10% (loose) of its pixels are organ. The strict one is the honest
# definition but leaves almost no interior patches, because an organ spans only ~6-10 patches.
thresholds = [0.5, 0.1]
mean_boundary = {t: [] for t in thresholds}
mean_interior = {t: [] for t in thresholds}
mean_surround = {t: [] for t in thresholds}
mean_elsewhere = {t: [] for t in thresholds}

for n, c in enumerate(sample):
    image, display, mask = load_slice(c[0])
    attn_per_block = attention_of_slice(image)
    regions = region_maps(display, mask)

    a_cls = cls_patch_attention(attn_per_block[11])
    a_roll = rollout(attn_per_block)
    for name in region_names:
        share_cls[name].append(float((a_cls * regions[name]).sum()))
        share_roll[name].append(float((a_roll * regions[name]).sum()))
        area[name].append(float(regions[name].mean()))

    # where do the 5 strongest patches of each map sit? (each patch assigned to its dominant region)
    dominant = np.array(region_names)[np.argmax([regions[name] for name in region_names], axis=0)]
    for attention_map, counter in [(a_cls, top5_region_cls), (a_roll, top5_region_roll)]:
        for p in np.argsort(attention_map, axis=None)[-5:]:
            counter[dominant.flat[p]] += 1

    organ_cov = regions['stomach'] + regions['small_bowel'] + regions['large_bowel']
    for h in range(6):
        a_h = cls_patch_attention(attn_per_block[11], head=h)
        head_pad_share[h].append(float((a_h * regions['padding']).sum()))
        head_organ_share[h].append(float((a_h * organ_cov).sum()))

    for b in range(12):
        a_b = cls_patch_attention(attn_per_block[b])
        organ_area = organ_cov.mean()
        pad_area = regions['padding'].mean()
        depth_organ_enrich[b].append(float((a_b * organ_cov).sum()) / organ_area if organ_area > 0 else np.nan)
        depth_pad_enrich[b].append(float((a_b * regions['padding']).sum()) / pad_area)
        prefix_mass[b].append(float(attn_per_block[b][:, 0, :N_PREFIX].mean(axis=0).sum()))

    # are organs preferred over the body tissue they sit in? (air and padding excluded)
    organ_in = float((a_cls * organ_cov).sum())
    body_in = float((a_cls * regions['body']).sum())
    organ_ar, body_ar = organ_cov.mean(), regions['body'].mean()
    if organ_in + body_in > 0:
        within_tissue_enrich.append((organ_in / (organ_in + body_in)) / (organ_ar / (organ_ar + body_ar)))

    # boundary vs interior at patch resolution
    kernel = np.ones((3, 3), np.uint8)
    for t in thresholds:
        organ_patch = (organ_cov >= t).astype(np.uint8)
        if organ_patch.sum() == 0:
            continue
        interior = cv2.erode(organ_patch, kernel)
        boundary = organ_patch - interior
        surround = cv2.dilate(organ_patch, kernel) - organ_patch
        elsewhere = 1 - cv2.dilate(organ_patch, kernel)
        if interior.sum() > 0:
            mean_interior[t].append(a_cls[interior > 0].mean())
        if boundary.sum() > 0:
            mean_boundary[t].append(a_cls[boundary > 0].mean())
        if surround.sum() > 0:
            mean_surround[t].append(a_cls[surround > 0].mean())
        mean_elsewhere[t].append(a_cls[elsewhere > 0].mean())

    if (n + 1) % 20 == 0:
        print(f'  {n + 1}/{len(sample)}', flush=True)

uniform = 1.0 / N_PATCHES

lines = []
lines.append('DINOv3 ViT-S/16 (vit_small_patch16_dinov3.lvd1689m, frozen, no medical fine-tuning)')
lines.append(f'288x288 input = 266x266 MRI + {PAD} px DivisiblePad; 18x18 = 324 patches of 16x16')
lines.append(f'averaged over {len(sample)} organ-containing slices from {len(cases)} patients {cases}')
lines.append('')
lines.append('1) WHERE DOES CLS ATTENTION MASS GO? (share = fraction of the 324-patch CLS attention,')
lines.append('   area = fraction of the input area; enrichment = share / area, 1.0 = chance)')
lines.append('')
lines.append(f'{"region":<13}{"area":>8}{"share(blk11)":>14}{"enrich":>9}{"share(rollout)":>16}{"enrich":>9}')
for name in region_names:
    ar = np.mean(area[name])
    sc, sr = np.mean(share_cls[name]), np.mean(share_roll[name])
    lines.append(f'{name:<13}{ar:>8.4f}{sc:>14.4f}{sc / ar:>9.2f}{sr:>16.4f}{sr / ar:>9.2f}')
organ_area = sum(np.mean(area[n]) for n in CLASSES)
organ_cls = sum(np.mean(share_cls[n]) for n in CLASSES)
organ_roll = sum(np.mean(share_roll[n]) for n in CLASSES)
lines.append(f'{"ALL ORGANS":<13}{organ_area:>8.4f}{organ_cls:>14.4f}{organ_cls / organ_area:>9.2f}'
             f'{organ_roll:>16.4f}{organ_roll / organ_area:>9.2f}')
lines.append('')
lines.append('1b) ARE ORGANS PREFERRED OVER THE BODY TISSUE AROUND THEM? (air and padding excluded,')
lines.append('    so this asks whether the organs are special or only the body is; 1.0 = no preference)')
lines.append('')
lines.append(f'    within-tissue organ enrichment (block 11 CLS) = {np.mean(within_tissue_enrich):.2f} '
             f'+- {np.std(within_tissue_enrich):.2f} (std over slices)')
lines.append('')
lines.append('2) BOUNDARY VS INTERIOR (3x3 erosion/dilation of the patch-level organ mask;')
lines.append(f'   mean attention PER PATCH, uniform attention would be 1/324 = {uniform:.5f})')
for t in thresholds:
    lines.append('')
    lines.append(f'   organ patch = patch is >= {t:.0%} organ:')
    lines.append(f'     boundary patches : {np.mean(mean_boundary[t]):.5f}  '
                 f'({np.mean(mean_boundary[t]) / uniform:.2f}x uniform, n={len(mean_boundary[t])} slices)')
    lines.append(f'     interior patches : {np.mean(mean_interior[t]):.5f}  '
                 f'({np.mean(mean_interior[t]) / uniform:.2f}x uniform, n={len(mean_interior[t])} slices '
                 f'that have any interior patch at all)')
    lines.append(f'     surrounding ring : {np.mean(mean_surround[t]):.5f}  '
                 f'({np.mean(mean_surround[t]) / uniform:.2f}x uniform)')
    lines.append(f'     everything else  : {np.mean(mean_elsewhere[t]):.5f}  '
                 f'({np.mean(mean_elsewhere[t]) / uniform:.2f}x uniform; includes air and padding)')
    lines.append(f'     boundary / interior ratio = {np.mean(mean_boundary[t]) / np.mean(mean_interior[t]):.2f}')
lines.append('')
lines.append('3) PER-HEAD BEHAVIOUR, block 11 (share of that head\'s CLS->patch mass)')
lines.append('')
pad_area = np.mean(area['padding'])
lines.append(f'{"head":<6}{"padding share":>15}{"enrich":>9}{"organ share":>13}{"enrich":>9}')
for h in range(6):
    ps, os_ = np.mean(head_pad_share[h]), np.mean(head_organ_share[h])
    lines.append(f'{h:<6}{ps:>15.4f}{ps / pad_area:>9.2f}{os_:>13.4f}{os_ / organ_area:>9.2f}')
lines.append(f'{"mean":<6}{np.mean([np.mean(x) for x in head_pad_share]):>15.4f}'
             f'{"":>9}{np.mean([np.mean(x) for x in head_organ_share]):>13.4f}')
lines.append('')
lines.append('4) DEPTH (head-mean CLS attention; prefix mass = CLS row mass on CLS + 4 storage tokens,')
lines.append('   i.e. attention that never reaches the image at all)')
lines.append('')
lines.append(f'{"block":<7}{"organ enrich":>14}{"padding enrich":>16}{"prefix mass":>13}')
for b in range(12):
    lines.append(f'{b:<7}{np.nanmean(depth_organ_enrich[b]):>14.2f}'
                 f'{np.mean(depth_pad_enrich[b]):>16.2f}{np.mean(prefix_mass[b]):>13.3f}')
lines.append('')
lines.append('5) WHERE THE 5 STRONGEST PATCHES OF EACH MAP LAND (each patch assigned to the region that')
lines.append(f'   covers most of it; {5 * len(sample)} patches in total per map)')
lines.append('')
lines.append(f'{"region":<13}{"block 11 CLS":>14}{"rollout":>10}')
for name in region_names:
    lines.append(f'{name:<13}{top5_region_cls[name] / (5 * len(sample)):>14.2f}'
                 f'{top5_region_roll[name] / (5 * len(sample)):>10.2f}')
report = '\n'.join(lines)
print('\n' + report)
(OUT_DIR / 'quantitative_report.txt').write_text(report, encoding='utf-8')

# ----------------------------------------------------------------------------- quantitative figures
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
x = np.arange(len(region_names))
axes[0].bar(x - 0.2, [np.mean(area[n]) for n in region_names], 0.4, label='area share')
axes[0].bar(x + 0.2, [np.mean(share_cls[n]) for n in region_names], 0.4, label='CLS attention share (blk 11)')
axes[0].set_xticks(x)
axes[0].set_xticklabels(region_names, rotation=30, ha='right')
axes[0].set_ylabel('share of total')
axes[0].set_title('Attention share vs area share')
axes[0].legend(fontsize=8)

enrich_cls = [np.mean(share_cls[n]) / np.mean(area[n]) for n in region_names]
enrich_roll = [np.mean(share_roll[n]) / np.mean(area[n]) for n in region_names]
axes[1].bar(x - 0.2, enrich_cls, 0.4, label='block 11 CLS')
axes[1].bar(x + 0.2, enrich_roll, 0.4, label='rollout')
axes[1].axhline(1.0, color='k', ls='--', lw=1)
axes[1].set_xticks(x)
axes[1].set_xticklabels(region_names, rotation=30, ha='right')
axes[1].set_ylabel('enrichment = share / area')
axes[1].set_title('Enrichment (1.0 = chance)')
axes[1].legend(fontsize=8)

groups = ['boundary', 'interior', 'surrounding\nring', 'elsewhere']
offsets = {0.5: -0.2, 0.1: 0.2}
for t in thresholds:
    values = [np.mean(mean_boundary[t]) / uniform, np.mean(mean_interior[t]) / uniform,
              np.mean(mean_surround[t]) / uniform, np.mean(mean_elsewhere[t]) / uniform]
    axes[2].bar(np.arange(4) + offsets[t], values, 0.4, label=f'organ patch >= {t:.0%} organ')
axes[2].axhline(1.0, color='k', ls='--', lw=1)
axes[2].set_xticks(np.arange(4))
axes[2].set_xticklabels(groups)
axes[2].set_ylabel('mean attention per patch / uniform')
axes[2].set_title('Organ boundary vs interior')
axes[2].legend(fontsize=8)
fig.suptitle(f'DINOv3 CLS attention, {len(sample)} slices, {len(cases)} patients', fontsize=11)
fig.tight_layout()
fig.savefig(OUT_DIR / 'quantitative_regions.png', dpi=130)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
heads = np.arange(6)
axes[0].bar(heads - 0.2, [np.mean(head_pad_share[h]) / pad_area for h in heads], 0.4, label='padding')
axes[0].bar(heads + 0.2, [np.mean(head_organ_share[h]) / organ_area for h in heads], 0.4, label='organs')
axes[0].axhline(1.0, color='k', ls='--', lw=1)
axes[0].set_xlabel('head (block 11)')
axes[0].set_ylabel('enrichment')
axes[0].set_title('Per-head enrichment on padding and on organs')
axes[0].legend(fontsize=8)

blocks = np.arange(12)
axes[1].plot(blocks, [np.nanmean(depth_organ_enrich[b]) for b in blocks], 'o-', label='organ enrichment')
axes[1].plot(blocks, [np.mean(depth_pad_enrich[b]) for b in blocks], 's-', label='padding enrichment')
axes[1].plot(blocks, [np.mean(prefix_mass[b]) for b in blocks], '^-', label='CLS mass on prefix tokens')
axes[1].axhline(1.0, color='k', ls='--', lw=1)
axes[1].set_xlabel('block')
axes[1].set_title('Depth trend')
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR / 'quantitative_heads_depth.png', dpi=130)
plt.close(fig)

print(f'\nmax relative error of the recomputed attention output vs the module\'s own output: '
      f'{max(recompute_error):.2e} (on GPU the fused kernel and this explicit matmul differ '
      f'slightly in floating point)')
print(f'wrote figures and quantitative_report.txt to {OUT_DIR}')
