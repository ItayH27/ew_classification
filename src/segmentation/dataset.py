"""
PyTorch Dataset for the HLS Burn Scar Scenes dataset.

The datasets library v3+ dropped support for loading scripts, so we download
the raw tar.gz from HuggingFace Hub and read the GeoTIFF pairs directly with
rasterio. On first use the archive is downloaded and extracted to ~/.cache/
hls_burn_scars/ (or a custom cache_dir). Subsequent runs are instant.

Archive structure (after extraction):
    training/
        <id>_merged.tif      — 6-band HLS image, uint16, 512×512
        <id>.mask.tif        — binary mask, uint8, 512×512  (0=bg, 1=burn)
    validation/
        <same pattern>
"""

from __future__ import annotations

from pathlib import Path as _Path
from dotenv import load_dotenv
load_dotenv(_Path(__file__).resolve().parent.parent.parent / ".env")

import tarfile
from glob import glob
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset

# Per-band statistics estimated from the HLS Burn Scar training split.
# Bands are in HLS order: Blue, Green, Red, Narrow NIR, SWIR1, SWIR2.
# The GeoTIFFs store float32 surface reflectance directly in [0, ~0.5].
HLS_BAND_MEANS = [0.033, 0.047, 0.044, 0.220, 0.195, 0.113]
HLS_BAND_STDS  = [0.018, 0.022, 0.028, 0.060, 0.080, 0.065]

_REPO_ID  = "ibm-nasa-geospatial/hls_burn_scars"
_FILENAME = "hls_burn_scars.tar.gz"
_SPLIT_DIRS = {"train": "training", "validation": "validation"}


class BurnScarDataset(Dataset):
    """
    HLS Burn Scar Scenes (512×512 GeoTIFFs) as a PyTorch Dataset.

    Args:
        split:       "train" or "validation"
        target_size: resize target for both H and W (default 256)
        augment:     random horizontal/vertical flips; only on train split
        band_means:  per-band means for normalisation (6 values, HLS order)
        band_stds:   per-band stds  for normalisation (6 values, HLS order)
        cache_dir:   where to store the downloaded + extracted dataset.
                     Defaults to ~/.cache/hls_burn_scars
        negative_dir: optional directory of GEE-sourced negative (no-burn-scar)
                     samples, same on-disk layout, from
                     src/retrieval/negative_sampling.py. Appended after the
                     HLS images; does not raise if missing/empty, since this
                     is optional infra. Native resolution may differ from
                     HLS's 512×512 — reconciled by the resize-to-target_size
                     below regardless of source shape.
    """

    def __init__(
        self,
        split: str = "train",
        target_size: int = 256,
        augment: bool = True,
        band_means: list[float] = HLS_BAND_MEANS,
        band_stds: list[float] = HLS_BAND_STDS,
        cache_dir: str | None = None,
        negative_dir: str | None = None,
    ) -> None:
        if split not in _SPLIT_DIRS:
            raise ValueError(f"split must be one of {list(_SPLIT_DIRS)}, got {split!r}")

        self.split       = split
        self.target_size = target_size
        self.augment     = augment and (split == "train")
        self.means = torch.tensor(band_means, dtype=torch.float32).view(6, 1, 1)
        self.stds  = torch.tensor(band_stds,  dtype=torch.float32).view(6, 1, 1)

        data_root = self._ensure_data(cache_dir)
        split_dir = data_root / _SPLIT_DIRS[split]

        hls_paths = sorted(glob(str(split_dir / "*_merged.tif")))
        if not hls_paths:
            raise RuntimeError(
                f"No *_merged.tif files found under {split_dir}. "
                "Try deleting the cache dir and re-running to re-extract."
            )

        negative_paths: list[str] = []
        if negative_dir:
            negative_split_dir = Path(negative_dir) / _SPLIT_DIRS[split]
            negative_paths = sorted(glob(str(negative_split_dir / "*_merged.tif")))

        self._image_paths = hls_paths + negative_paths
        self.is_negative = [False] * len(hls_paths) + [True] * len(negative_paths)

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image: (6, target_size, target_size) float32 — normalised HLS bands
            mask:  (1, target_size, target_size) float32 — binary {0., 1.}
        """
        img_path  = self._image_paths[idx]
        mask_path = img_path.replace("_merged.tif", ".mask.tif")

        image = self._read_image(img_path)   # (6, H, W) float32 [0, ~1]
        mask  = self._read_mask(mask_path)   # (1, H, W) float32 {0, 1}

        image = self._resize(image, mode="bilinear")
        mask  = self._resize(mask,  mode="nearest")

        image = (image - self.means) / self.stds

        if self.augment:
            image, mask = _random_flips(image, mask)

        return image, mask

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_image(path: str) -> torch.Tensor:
        with rasterio.open(path) as src:
            arr = src.read().astype(np.float32)   # (6, H, W), already float32 reflectance
        return torch.from_numpy(arr)

    @staticmethod
    def _read_mask(path: str) -> torch.Tensor:
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)  # (H, W)
        # -1 encodes nodata/cloud pixels; treat them as background (0)
        arr = np.clip(arr, 0.0, 1.0)
        return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)

    def _resize(self, t: torch.Tensor, mode: str) -> torch.Tensor:
        t = t.unsqueeze(0)  # (1, C, H, W)
        t = F.interpolate(
            t,
            size=(self.target_size, self.target_size),
            mode=mode,
            align_corners=False if mode == "bilinear" else None,
        )
        return t.squeeze(0)

    @staticmethod
    def _ensure_data(cache_dir: str | None) -> Path:
        """
        Download and extract the dataset archive if not already present.
        Returns the root directory containing training/ and validation/.
        """
        if cache_dir:
            root = Path(cache_dir)
        else:
            # Absolute path so it works even when run from Slurm batch jobs
            # where the working directory may differ.
            root = Path("/home/yandex/DLWorkShop2026b/itayhazan/ew_classifier/dataset")
            
        sentinel = root / ".extracted"

        if sentinel.exists():
            return root

        root.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {_FILENAME} from HuggingFace Hub …")
        archive_path = hf_hub_download(
            repo_id=_REPO_ID,
            filename=_FILENAME,
            repo_type="dataset",
            local_dir=str(root),
        )

        print(f"Extracting to {root} …")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=root)

        sentinel.touch()
        return root


def _random_flips(
    image: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.rand(1).item() > 0.5:
        image = torch.flip(image, dims=[2])   # horizontal
        mask  = torch.flip(mask,  dims=[2])
    if torch.rand(1).item() > 0.5:
        image = torch.flip(image, dims=[1])   # vertical
        mask  = torch.flip(mask,  dims=[1])
    return image, mask
