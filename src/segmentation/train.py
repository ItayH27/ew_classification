"""
Training loop for the burn-scar segmentation model.

Supports CUDA (with AMP), Apple MPS, and CPU. Checkpoints are saved in
HuggingFace diffusers format so the model can be reloaded with
BurnScarSegmenter.load_pretrained(output_dir/best_model).
"""

from __future__ import annotations

import random
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import BurnScarDataset
from .losses import CombinedLoss
from .metrics import SegmentationMetrics
from .model import BurnScarSegmenter


def train(
    batch_size: int = 8,
    lr: float = 1e-4,
    epochs: int = 30,
    output_dir: str = "/home/yandex/DLWorkShop2026b/itayhazan/ew_classifier/checkpoints/segmentation",
    accumulation_steps: int = 1,
    pos_weight: float = 10.0,
    alpha: float = 0.5,
    num_workers: int = 4,
    negative_dir: str | None = None,
    negative_fraction: float = 0.0,
    seed: int | None = None,
    weight_decay: float = 1e-4,
) -> None:
    """
    Fine-tune BurnScarSegmenter on the HLS Burn Scar Scenes dataset.

    Saves the best checkpoint (by validation IoU) to:
        <output_dir>/best_model/   — HuggingFace diffusers format
        <output_dir>/training_state.pt — optimizer state + epoch for resuming

    negative_dir/negative_fraction: optionally mix in GEE-sourced negative
    (no-burn-scar) samples (see src/retrieval/negative_sampling.py) at a
    target fraction of each training batch via WeightedRandomSampler.
    Validation also draws from negative_dir when set, but unweighted, so
    val_iou/val_fpr reflect real exposure to benign scenes.

    seed: if given, seeds random/numpy/torch (and CUDA) before dataset/model
    construction, so DataLoader worker shuffling (PyTorch derives worker RNGs
    from torch.initial_seed()) and BurnScarDataset's flip augmentation are
    reproducible run-to-run. Full determinism (torch.use_deterministic_algorithms)
    is not enabled — this is for comparable runs, not bitwise reproducibility.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    start_time = time.perf_counter()
    device  = _get_device()
    use_amp = device.type == "cuda"
    out     = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}  |  AMP: {use_amp}")

    train_ds = BurnScarDataset(split="train",      augment=True,  negative_dir=negative_dir)
    val_ds   = BurnScarDataset(split="validation", augment=False, negative_dir=negative_dir)

    if negative_dir and negative_fraction > 0:
        is_neg = torch.tensor(train_ds.is_negative)
        n_neg, n_pos = int(is_neg.sum()), int((~is_neg).sum())
        if n_neg == 0:
            raise ValueError(
                f"negative_fraction={negative_fraction} given but no negative samples "
                f"found under {negative_dir}"
            )
        w_neg = negative_fraction / n_neg
        w_pos = (1 - negative_fraction) / n_pos
        weights = torch.where(is_neg, w_neg, w_pos)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights, num_samples=len(train_ds), replacement=True,
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=use_amp,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=use_amp,
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_amp,
    )

    model     = BurnScarSegmenter().to(device)
    criterion = CombinedLoss(alpha=alpha, pos_weight=pos_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4,
    )
    scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)
    metrics   = SegmentationMetrics()

    best_val_iou = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = _train_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, use_amp, accumulation_steps,
        )

        model.eval()
        val_loss, val_result = _val_epoch(
            model, val_loader, criterion, metrics, device, use_amp,
        )

        lr_before = optimizer.param_groups[0]["lr"]
        scheduler.step(val_result["iou"])
        lr_after = optimizer.param_groups[0]["lr"]
        if lr_after < lr_before:
            print(f"  --> Reducing LR: {lr_before:.2e} -> {lr_after:.2e}")

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_iou={val_result['iou']:.4f} | "
            f"val_dice={val_result['dice']:.4f} | "
            f"val_fpr={val_result['fpr']:.4f} | "
            f"val_fnr={val_result['fnr']:.4f}"
        )

        if val_result["iou"] > best_val_iou:
            best_val_iou = val_result["iou"]
            model.save_pretrained(str(out / "best_model"))
            torch.save(
                {
                    "epoch": epoch,
                    "best_val_iou": best_val_iou,
                    "best_val_dice": val_result["dice"],
                    "best_val_fpr": val_result["fpr"],
                    "best_val_fnr": val_result["fnr"],
                    "optimizer_state": optimizer.state_dict(),
                },
                out / "training_state.pt",
            )
            print(f"  --> New best val IoU {best_val_iou:.4f} — checkpoint saved.")

    elapsed = timedelta(seconds=round(time.perf_counter() - start_time))
    print(f"Total training time: {elapsed}")


def _train_epoch(
    model: BurnScarSegmenter,
    loader: DataLoader,
    criterion: CombinedLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    accum_steps: int,
) -> float:
    total_loss = 0.0
    optimizer.zero_grad()

    for step, (images, masks) in enumerate(tqdm(loader, desc="train", leave=False)):
        images = images.to(device)
        masks  = masks.to(device)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss   = criterion(logits, masks) / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps

    # Flush any remaining accumulated gradients at the end of the epoch
    if len(loader) % accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return total_loss / len(loader)


def _val_epoch(
    model: BurnScarSegmenter,
    loader: DataLoader,
    criterion: CombinedLoss,
    metrics: SegmentationMetrics,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, dict[str, float]]:
    total_loss = 0.0
    metrics.reset()

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="val", leave=False):
            images = images.to(device)
            masks  = masks.to(device)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss   = criterion(logits, masks)

            total_loss += loss.item()
            metrics.update(logits, masks)

    return total_loss / len(loader), metrics.compute()


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
