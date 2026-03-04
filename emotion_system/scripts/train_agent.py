#!/usr/bin/env python3
"""
Agent / Gating Training Script
=================================
Trains the Event Encoder + Agent on pre-extracted expert features.

Two-stage approach:
  Stage 1: Run FER Expert (frozen) + optional A/V Expert on dataset
           → save per-frame features to disk (offline extraction)
  Stage 2: Train Event Encoder + Agent on windowed features

This script handles Stage 2. Run extract_expert_features.py for Stage 1.

Usage:
  # Extract features first (Stage 1)
  python scripts/extract_expert_features.py --config configs/pipeline.yaml

  # Train Agent (Stage 2)
  python scripts/train_agent.py --config configs/pipeline.yaml \
      --features_dir pipeline_output/expert_features
"""

import argparse
import time
import yaml
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.event_encoder import build_event_encoder
from models.agent_gating import AgentGating, AgentLoss
from utils.seed import set_seed, ensure_dir
from utils.logging import setup_logger, MetricLogger

try:
    GradScaler = torch.amp.GradScaler
except AttributeError:
    GradScaler = torch.cuda.amp.GradScaler


class ExpertFeatureDataset(Dataset):
    """
    Loads pre-extracted expert features for Agent training.

    Each sample is a window of T frames with:
      - z_seq: [T, z_dim] concatenated expert features
      - gt_emotion: int, ground truth emotion class
      - gt_av: [2] arousal/valence (optional)
      - gt_drowsy: int, drowsiness level (optional)
    """

    def __init__(self, features_dir: str, window_size: int = 30,
                 stride: int = 15):
        self.features_dir = Path(features_dir)
        self.window_size = window_size
        self.stride = stride

        # Load all feature files
        self.data = []
        npz_files = sorted(self.features_dir.glob("*.npz"))

        if len(npz_files) == 0:
            # Try loading single consolidated file
            consolidated = self.features_dir / "all_features.npz"
            if consolidated.exists():
                d = np.load(str(consolidated), allow_pickle=True)
                z_all = d["z_seq"]           # [N_total, z_dim]
                labels = d["labels"]         # [N_total]
                av = d.get("av", None)       # [N_total, 2] or None
                drowsy = d.get("drowsy", None)  # [N_total] or None

                # Window slicing
                N = len(z_all)
                for start in range(0, N - window_size + 1, stride):
                    end = start + window_size
                    self.data.append({
                        "z_seq": z_all[start:end].astype(np.float32),
                        "gt_emotion": int(labels[end - 1]),
                        "gt_av": av[end - 1].astype(np.float32) if av is not None else np.zeros(2, dtype=np.float32),
                        "gt_drowsy": int(drowsy[end - 1]) if drowsy is not None else 0,
                    })
        else:
            for f in npz_files:
                d = np.load(str(f), allow_pickle=True)
                self.data.append({
                    "z_seq": d["z_seq"].astype(np.float32),
                    "gt_emotion": int(d["gt_emotion"]),
                    "gt_av": d["gt_av"].astype(np.float32) if "gt_av" in d else np.zeros(2, dtype=np.float32),
                    "gt_drowsy": int(d["gt_drowsy"]) if "gt_drowsy" in d else 0,
                })

        if len(self.data) == 0:
            raise RuntimeError(
                f"No feature data found in {features_dir}. "
                "Run extract_expert_features.py first."
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        return {
            "z_seq": torch.tensor(d["z_seq"]),
            "gt_emotion": torch.tensor(d["gt_emotion"], dtype=torch.long),
            "gt_av": torch.tensor(d["gt_av"]),
            "gt_drowsy": torch.tensor(d["gt_drowsy"], dtype=torch.long),
        }


def collate_agent(batch):
    return {
        "z_seq": torch.stack([b["z_seq"] for b in batch]),
        "gt_emotion": torch.stack([b["gt_emotion"] for b in batch]),
        "gt_av": torch.stack([b["gt_av"] for b in batch]),
        "gt_drowsy": torch.stack([b["gt_drowsy"] for b in batch]),
    }


def train_agent(config_path: str, features_dir: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))

    output_dir = Path(cfg["paths"]["output_dir"]) / "agent_training"
    ensure_dir(output_dir)

    logger = setup_logger("agent_train", str(output_dir))
    metric_log = MetricLogger(output_dir / "history.jsonl")

    device = torch.device(cfg["runtime"].get("device", "cuda")
                          if torch.cuda.is_available() else "cpu")

    # ── Config ──
    ecfg = cfg["event_encoder"]
    acfg = cfg["agent"]
    tcfg = cfg["agent_training"]
    window_size = cfg["runtime"]["window_size"]
    stride = cfg["runtime"].get("window_stride", window_size // 2)

    # ── Dataset ──
    dataset = ExpertFeatureDataset(features_dir, window_size, stride)
    z_dim = dataset.data[0]["z_seq"].shape[-1]
    logger.info(f"Feature dim: {z_dim}, Windows: {len(dataset)}")

    # Split train/val (80/20)
    n_val = max(1, int(len(dataset) * 0.2))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set, batch_size=tcfg["batch_size"], shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=collate_agent,
    )
    val_loader = DataLoader(
        val_set, batch_size=tcfg["batch_size"], shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=collate_agent,
    )

    # ── Event Encoder ──
    event_encoder = build_event_encoder(
        encoder_type=ecfg.get("type", "gru"),
        input_dim=z_dim,
        hidden_dim=ecfg.get("hidden_dim", 128),
        n_layers=ecfg.get("n_layers", 1),
        dropout=ecfg.get("dropout", 0.1),
        bidirectional=ecfg.get("bidirectional", True),
    ).to(device)

    # ── Compute Agent state dimensions ──
    d_fer_logits = cfg["fer_expert"]["num_classes"]
    d_fer_extra = 2  # quality + uncertainty
    d_av_logits = cfg["av_expert"]["num_classes"] if acfg["use_av_expert"] else 0
    d_av_extra = 4 if acfg["use_av_expert"] else 0  # arousal, valence, quality, uncertainty
    d_perclos = 2  # perclos + q_perclos
    d_drowsy = 1   # p_drowsy

    # ── Agent ──
    agent = AgentGating(
        num_classes=acfg["num_classes"],
        d_event=event_encoder.out_dim,
        d_fer_logits=d_fer_logits,
        d_av_logits=d_av_logits,
        d_fer_extra=d_fer_extra,
        d_av_extra=d_av_extra,
        d_perclos=d_perclos,
        d_drowsy=d_drowsy,
        hidden_dim=acfg["hidden_dim"],
        use_av_expert=acfg["use_av_expert"],
        use_event_encoder=True,
    ).to(device)

    logger.info(f"Event Encoder params: {sum(p.numel() for p in event_encoder.parameters()):,}")
    logger.info(f"Agent params: {sum(p.numel() for p in agent.parameters()):,}")

    # ── Optimizer ──
    params = list(event_encoder.parameters()) + list(agent.parameters())
    optimizer = torch.optim.AdamW(
        params, lr=float(tcfg["lr"]),
        weight_decay=float(tcfg.get("weight_decay", 0.01))
    )

    total_steps = len(train_loader) * tcfg["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6
    )

    criterion = AgentLoss(
        w_cls=tcfg.get("w_cls", 1.0),
        w_av=tcfg.get("w_av", 0.5),
        w_drowsy=tcfg.get("w_drowsy", 0.3),
        w_smooth=tcfg.get("w_smooth", 0.1),
        w_consist=tcfg.get("w_consist", 0.1),
    )

    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    # ── Training Loop ──
    best_f1 = -1.0
    for epoch in range(1, tcfg["epochs"] + 1):
        event_encoder.train()
        agent.train()
        t0 = time.time()

        running_loss = 0.0
        running_correct = 0
        n_samples = 0

        for batch in train_loader:
            z_seq = batch["z_seq"].to(device)
            gt_emotion = batch["gt_emotion"].to(device)
            gt_av = batch["gt_av"].to(device)
            gt_drowsy = batch["gt_drowsy"].to(device)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                h_event = event_encoder(z_seq)

                # Extract last-frame expert features from z_seq
                z_last = z_seq[:, -1, :]  # [B, z_dim]
                fer_logits = z_last[:, :d_fer_logits]
                fer_quality = z_last[:, d_fer_logits]
                fer_uncertainty = z_last[:, d_fer_logits + 1]

                offset = d_fer_logits + d_fer_extra
                if acfg["use_av_expert"]:
                    av_logits = z_last[:, offset:offset + d_av_logits]
                    av_arousal = z_last[:, offset + d_av_logits]
                    av_valence = z_last[:, offset + d_av_logits + 1]
                    av_quality = z_last[:, offset + d_av_logits + 2]
                    av_uncertainty = z_last[:, offset + d_av_logits + 3]
                    offset += d_av_logits + d_av_extra
                else:
                    av_logits = av_arousal = av_valence = av_quality = av_uncertainty = None

                perclos = z_last[:, offset]
                q_perclos = z_last[:, offset + 1] if z_last.size(-1) > offset + 1 else torch.ones_like(perclos)
                p_drowsy = (perclos > 0.4).float()

                pred = agent(
                    fer_logits=fer_logits,
                    fer_quality=fer_quality,
                    fer_uncertainty=fer_uncertainty,
                    perclos=perclos,
                    q_perclos=q_perclos,
                    p_drowsy=p_drowsy,
                    h_event=h_event,
                    av_logits=av_logits,
                    av_arousal=av_arousal,
                    av_valence=av_valence,
                    av_quality=av_quality,
                    av_uncertainty=av_uncertainty,
                )

                loss, loss_dict = criterion(pred, gt_emotion, gt_av, gt_drowsy)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(params, 5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            bs = gt_emotion.size(0)
            running_loss += loss.item() * bs
            running_correct += (pred["emotion_logits"].argmax(1) == gt_emotion).sum().item()
            n_samples += bs

        train_loss = running_loss / max(1, n_samples)
        train_acc = running_correct / max(1, n_samples)
        epoch_time = time.time() - t0

        # ── Validation ──
        event_encoder.eval()
        agent.eval()
        val_correct = 0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                z_seq = batch["z_seq"].to(device)
                gt_emotion = batch["gt_emotion"].to(device)

                h_event = event_encoder(z_seq)
                z_last = z_seq[:, -1, :]
                fer_logits = z_last[:, :d_fer_logits]
                fer_quality = z_last[:, d_fer_logits]
                fer_uncertainty = z_last[:, d_fer_logits + 1]

                offset = d_fer_logits + d_fer_extra
                if acfg["use_av_expert"]:
                    av_logits_v = z_last[:, offset:offset + d_av_logits]
                    av_arousal_v = z_last[:, offset + d_av_logits]
                    av_valence_v = z_last[:, offset + d_av_logits + 1]
                    av_quality_v = z_last[:, offset + d_av_logits + 2]
                    av_uncertainty_v = z_last[:, offset + d_av_logits + 3]
                    offset += d_av_logits + d_av_extra
                else:
                    av_logits_v = av_arousal_v = av_valence_v = av_quality_v = av_uncertainty_v = None

                perclos_v = z_last[:, offset]
                q_perclos_v = z_last[:, offset + 1] if z_last.size(-1) > offset + 1 else torch.ones_like(perclos_v)
                p_drowsy_v = (perclos_v > 0.4).float()

                pred = agent(
                    fer_logits=fer_logits,
                    fer_quality=fer_quality,
                    fer_uncertainty=fer_uncertainty,
                    perclos=perclos_v,
                    q_perclos=q_perclos_v,
                    p_drowsy=p_drowsy_v,
                    h_event=h_event,
                    av_logits=av_logits_v,
                    av_arousal=av_arousal_v,
                    av_valence=av_valence_v,
                    av_quality=av_quality_v,
                    av_uncertainty=av_uncertainty_v,
                )
                val_correct += (pred["emotion_logits"].argmax(1) == gt_emotion).sum().item()
                val_n += gt_emotion.size(0)

        val_acc = val_correct / max(1, val_n)

        log = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_acc": round(val_acc, 4),
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time": round(epoch_time, 1),
        }
        metric_log.log(log)
        logger.info(
            f"[E{epoch:03d}] loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_acc={val_acc:.4f} | lr={log['lr']:.2e} | {epoch_time:.1f}s"
        )

        # Save best
        if val_acc > best_f1:
            best_f1 = val_acc
            torch.save({
                "event_encoder": event_encoder.state_dict(),
                "agent": agent.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "config": cfg,
            }, output_dir / "best_agent.pth")
            logger.info(f"  * New best val_acc: {val_acc:.4f}")

        # Periodic save
        if epoch % 10 == 0:
            torch.save({
                "event_encoder": event_encoder.state_dict(),
                "agent": agent.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
            }, output_dir / f"agent_epoch_{epoch:03d}.pth")

    logger.info(f"[DONE] Best val_acc = {best_f1:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/pipeline.yaml")
    ap.add_argument("--features_dir", type=str, required=True,
                    help="Directory with pre-extracted expert features")
    args = ap.parse_args()
    train_agent(args.config, args.features_dir)
