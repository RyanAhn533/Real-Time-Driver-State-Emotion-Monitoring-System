#!/usr/bin/env python3
"""
Expert Feature Extraction (Stage 1 for Agent Training)
=========================================================
Runs frozen FER Expert (+ optional A/V Expert) on the dataset
and saves per-frame features to disk.

These features are then windowed and used by train_agent.py (Stage 2).

Output per frame z(t):
  [fer_probs(7), fer_quality(1), fer_uncertainty(1),
   av_probs(7), av_arousal(1), av_valence(1), av_quality(1), av_uncertainty(1),
   ear_avg(1), eyes_closed(1)]
  = 22-dim vector (without A/V: 11-dim)

Usage:
  python scripts/extract_expert_features.py \
      --config configs/pipeline.yaml \
      --csv /mnt/hdd/ajy_25/au_csv/index_train.csv \
      --out_dir pipeline_output/expert_features
"""

import argparse
import yaml
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import AUFERDataset, build_label_mapping, find_region_prefixes, collate_fn
from models.fer_model import AUFERModel
from utils.seed import set_seed, ensure_dir


def extract(config_path: str, csv_path: str, out_dir: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(out_dir)
    ensure_dir(out_path)

    # ── Load FER Expert ──
    fer_ckpt_path = cfg["paths"]["fer_checkpoint"]
    ckpt = torch.load(fer_ckpt_path, map_location=device, weights_only=False)
    fer_cfg = ckpt.get("config", {}).get("model", cfg["fer_expert"])

    label2id = build_label_mapping(csv_path)
    region_prefixes = find_region_prefixes(csv_path)
    num_au = len(region_prefixes)
    num_classes = len(label2id)

    model = AUFERModel(
        backbone_name=fer_cfg.get("backbone", "mobilevitv2_100"),
        pretrained=False,
        num_au=num_au,
        num_classes=num_classes,
        d_emb=fer_cfg.get("d_emb", 384),
        n_heads=fer_cfg.get("n_heads", 8),
        n_fusion_layers=fer_cfg.get("n_fusion_layers", 1),
        roi_mode=fer_cfg.get("roi_mode", "bilinear"),
        roi_spatial=fer_cfg.get("roi_spatial", 1),
        dropout=0.0,
        head_dropout=0.0,
    ).to(device)

    state = ckpt.get("model_state_dict", ckpt.get("model", {}))
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"FER Expert loaded: {fer_ckpt_path}")

    mean = model.backbone.norm_mean
    std = model.backbone.norm_std

    # ── Dataset ──
    dataset = AUFERDataset(
        csv_path, label2id, region_prefixes,
        img_size=224, mean=mean, std=std, is_train=False,
    )
    loader = DataLoader(
        dataset, batch_size=64, shuffle=False,
        num_workers=8, pin_memory=True, collate_fn=collate_fn,
    )

    # ── Extract ──
    all_z = []
    all_labels = []
    id2label = {v: k for k, v in label2id.items()}

    print(f"Extracting features from {len(dataset)} samples...")

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting"):
            images = batch["image"].to(device)
            au_coords = batch["au_coords"].to(device)
            labels = batch["label"]

            logits = model(images, au_coords)  # [B, C]
            probs = F.softmax(logits, dim=-1)

            # Uncertainty
            entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)
            max_entropy = np.log(num_classes)
            uncertainty = entropy / max_entropy

            # Quality (heuristic: based on AU coord validity)
            valid = (au_coords >= 0).all(dim=-1).all(dim=-1).float()
            in_range = (au_coords <= 224).all(dim=-1).all(dim=-1).float()
            quality = valid * in_range

            B = logits.size(0)
            for i in range(B):
                parts = [
                    probs[i].cpu().numpy(),                      # [C] fer probs
                    np.array([quality[i].item()]),                # [1] quality
                    np.array([uncertainty[i].item()]),            # [1] uncertainty
                    # A/V Expert placeholder (zeros when not available)
                    np.zeros(num_classes),                        # [C] av probs
                    np.zeros(4),                                  # [4] arousal, valence, quality, uncertainty
                    # PerClos placeholder
                    np.zeros(2),                                  # [2] ear_avg, eyes_closed
                ]
                z = np.concatenate(parts).astype(np.float32)
                all_z.append(z)
                all_labels.append(labels[i].item())

    all_z = np.stack(all_z)          # [N, z_dim]
    all_labels = np.array(all_labels)  # [N]

    # Save consolidated
    save_path = out_path / "all_features.npz"
    np.savez_compressed(
        str(save_path),
        z_seq=all_z,
        labels=all_labels,
        av=np.zeros((len(all_labels), 2), dtype=np.float32),
        drowsy=np.zeros(len(all_labels), dtype=np.int64),
    )

    print(f"Saved {len(all_z)} frames to {save_path}")
    print(f"Feature dim: {all_z.shape[-1]}")
    print(f"Label distribution: {dict(zip(*np.unique(all_labels, return_counts=True)))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/pipeline.yaml")
    ap.add_argument("--csv", type=str, required=True,
                    help="Training CSV with AU coordinates")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Output directory for features")
    args = ap.parse_args()
    extract(args.config, args.csv, args.out_dir)
