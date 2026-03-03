"""
AU FER Dataset
===============
CSV → PyTorch tensors.

두 가지 CSV 스키마 호환:
  1. 새 스키마 (au_extractor.py 출력):
     path, label, work_w, work_h, forehead_cx, forehead_cy, ...
  2. 기존 스키마 (2_AU_crop_csv_copy_3.py 출력):
     path, label, rot_deg, work_w, work_h, patch_in, patch_out,
     forehead_69_cx, forehead_69_cy, forehead_69_wx1, ...

자동 감지해서 _cx 컬럼 사용. _cx가 없고 _wx1만 있으면 중점 계산.
"""

import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as TF


class AUFERDataset(Dataset):
    def __init__(self,
                 csv_path: str,
                 label2id: Dict[str, int],
                 region_prefixes: List[str],
                 img_size: int = 224,
                 mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
                 std: Tuple[float, ...] = (0.229, 0.224, 0.225),
                 is_train: bool = True):
        super().__init__()

        self.df = pd.read_csv(csv_path)
        self.label2id = label2id
        self.region_prefixes = region_prefixes
        self.img_size = img_size
        self.is_train = is_train
        self.mean = mean
        self.std = std

        # 좌표 컬럼 감지
        self.cx_cols = []
        self.cy_cols = []
        self.use_midpoint = False

        sample_prefix = region_prefixes[0]
        if f"{sample_prefix}_cx" in self.df.columns:
            self.cx_cols = [f"{p}_cx" for p in region_prefixes]
            self.cy_cols = [f"{p}_cy" for p in region_prefixes]
        elif f"{sample_prefix}_wx1" in self.df.columns:
            self.use_midpoint = True
            self.wx1_cols = [f"{p}_wx1" for p in region_prefixes]
            self.wy1_cols = [f"{p}_wy1" for p in region_prefixes]
            self.wx2_cols = [f"{p}_wx2" for p in region_prefixes]
            self.wy2_cols = [f"{p}_wy2" for p in region_prefixes]
        else:
            raise KeyError(
                f"CSV에 '{sample_prefix}_cx' 또는 '{sample_prefix}_wx1' 컬럼이 없습니다.\n"
                f"사용 가능 컬럼: {list(self.df.columns)}"
            )

        cols_to_check = self.cx_cols + self.cy_cols if not self.use_midpoint else \
                        self.wx1_cols + self.wy1_cols + self.wx2_cols + self.wy2_cols
        for col in cols_to_check:
            if col not in self.df.columns:
                raise KeyError(f"CSV missing column: {col}")

        self.swap_pairs = self._find_swap_pairs()

    def _find_swap_pairs(self) -> List[Tuple[int, int]]:
        """좌우 대칭 AU 쌍 찾기."""
        pairs = []
        names = self.region_prefixes
        used = set()
        for i, name_i in enumerate(names):
            if i in used:
                continue
            if "left" in name_i:
                partner = name_i.replace("left", "right")
                for j, name_j in enumerate(names):
                    if name_j == partner and j not in used:
                        pairs.append((i, j))
                        used.add(i)
                        used.add(j)
                        break
            else:
                base_i = name_i.rsplit("_", 1)[0] if "_" in name_i else name_i
                for j, name_j in enumerate(names):
                    if j <= i or j in used:
                        continue
                    base_j = name_j.rsplit("_", 1)[0] if "_" in name_j else name_j
                    if base_i == base_j:
                        pairs.append((i, j))
                        used.add(i)
                        used.add(j)
                        break
        return pairs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = str(row["path"])
        label_name = str(row["label"])
        label_id = self.label2id[label_name]
        work_w = float(row["work_w"])
        work_h = float(row["work_h"])

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.fromarray(
                np.zeros((int(work_h), int(work_w), 3), dtype=np.uint8)
            )

        if self.use_midpoint:
            wx1 = np.array([float(row[c]) for c in self.wx1_cols], dtype=np.float32)
            wy1 = np.array([float(row[c]) for c in self.wy1_cols], dtype=np.float32)
            wx2 = np.array([float(row[c]) for c in self.wx2_cols], dtype=np.float32)
            wy2 = np.array([float(row[c]) for c in self.wy2_cols], dtype=np.float32)
            cxs = (wx1 + wx2) / 2.0
            cys = (wy1 + wy2) / 2.0
        else:
            cxs = np.array([float(row[c]) for c in self.cx_cols], dtype=np.float32)
            cys = np.array([float(row[c]) for c in self.cy_cols], dtype=np.float32)

        # Scale: work resolution → img_size
        cxs = cxs * (self.img_size / work_w)
        cys = cys * (self.img_size / work_h)

        img = img.resize((self.img_size, self.img_size), resample=Image.BILINEAR)

        if self.is_train:
            img, cxs, cys = self._augment(img, cxs, cys)

        img_tensor = TF.to_tensor(img)
        mean_t = torch.tensor(self.mean, dtype=torch.float32).view(3, 1, 1)
        std_t = torch.tensor(self.std, dtype=torch.float32).view(3, 1, 1)
        img_tensor = (img_tensor - mean_t) / std_t

        au_coords = torch.tensor(
            np.stack([cxs, cys], axis=-1), dtype=torch.float32
        )

        return {
            "image": img_tensor,
            "au_coords": au_coords,
            "label": torch.tensor(label_id, dtype=torch.long),
            "path": img_path,
        }

    def _augment(self, img, cxs, cys):
        if random.random() < 0.5:
            img = TF.hflip(img)
            cxs = self.img_size - cxs
            for i, j in self.swap_pairs:
                cxs[i], cxs[j] = cxs[j], cxs[i]
                cys[i], cys[j] = cys[j], cys[i]

        img = TF.adjust_brightness(img, 1 + (random.random() - 0.5) * 0.4)
        img = TF.adjust_contrast(img, 1 + (random.random() - 0.5) * 0.4)
        img = TF.adjust_saturation(img, 1 + (random.random() - 0.5) * 0.2)
        img = TF.adjust_hue(img, (random.random() - 0.5) * 0.04)

        return img, cxs, cys


def build_label_mapping(csv_path: str) -> Dict[str, int]:
    df = pd.read_csv(csv_path)
    labels = sorted(df["label"].unique().tolist())
    return {lb: i for i, lb in enumerate(labels)}


def find_region_prefixes(csv_path: str) -> List[str]:
    df = pd.read_csv(csv_path, nrows=2)
    prefixes = []
    for c in df.columns:
        if c.endswith("_cx"):
            prefixes.append(c[:-3])
        elif c.endswith("_wx1") and f"{c[:-4]}_cx" not in df.columns:
            prefixes.append(c[:-4])
    return sorted(set(prefixes))


def collate_fn(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "au_coords": torch.stack([b["au_coords"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "path": [b["path"] for b in batch],
    }
