"""
Main Training Loop
====================
기존 코드 대비 수정사항:
  1. Backbone freeze → unfreeze 시 remaining_steps로 scheduler 재계산
  2. Focal loss 단독 사용 (label_smoothing + class_weight 충돌 제거)
  3. Differentiated LR (backbone vs head)
  4. Expression magnitude scorer의 neutral centroid EMA 업데이트
  5. 단일 backbone → 321x 연산 감소
"""

import time
import yaml
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import AUFERDataset, build_label_mapping, find_region_prefixes, collate_fn
from models.fer_model import AUFERModel
from models.backbones import MobileViTBackbone
from training.losses import FocalLoss, compute_class_weights
from training.scheduler import build_scheduler
from training.evaluator import evaluate, save_confusion_matrix, save_report
from utils.seed import set_seed, ensure_dir
from utils.logging import setup_logger, MetricLogger
from utils.checkpoint import save_checkpoint, load_checkpoint
try:
    GradScaler = torch.amp.GradScaler
    autocast = torch.amp.autocast
except AttributeError:
    GradScaler = torch.cuda.amp.GradScaler
    autocast = torch.cuda.amp.autocast

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train(config_path: str, resume: str = None):
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))

    # Paths
    train_csv = cfg["paths"]["train_csv"]
    val_csv = cfg["paths"]["val_csv"]
    output_dir = Path(cfg["paths"]["output_dir"])
    ensure_dir(output_dir)

    # Logger
    logger = setup_logger("train", str(output_dir))
    metric_log = MetricLogger(output_dir / "history.jsonl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Labels & Regions ──
    label2id = build_label_mapping(train_csv)
    id2label = {v: k for k, v in label2id.items()}
    num_classes = len(label2id)
    region_prefixes = find_region_prefixes(train_csv)
    num_au = len(region_prefixes)

    logger.info(f"Classes ({num_classes}): {label2id}")
    logger.info(f"AU regions ({num_au}): {region_prefixes}")

    # ── Model ──
    mcfg = cfg["model"]
    model = AUFERModel(
        backbone_name=mcfg.get("backbone", "mobilevitv2_100"),
        pretrained=mcfg.get("pretrained", True),
        num_au=num_au,
        num_classes=num_classes,
        d_emb=mcfg.get("d_emb", 384),
        n_heads=mcfg.get("n_heads", 8),
        n_fusion_layers=mcfg.get("n_fusion_layers", 1),
        roi_mode=mcfg.get("roi_mode", "bilinear"),
        roi_spatial=mcfg.get("roi_spatial", 1),
        dropout=mcfg.get("dropout", 0.1),
        head_dropout=mcfg.get("head_dropout", 0.2),
        gate_init=mcfg.get("gate_init", 0.0),
        img_size=cfg["augmentation"].get("global_img_size", 224),
    ).to(device)

    # Normalization from backbone
    mean = model.backbone.norm_mean
    std = model.backbone.norm_std

    # ── Dataset ──
    img_size = cfg["augmentation"].get("global_img_size", 224)
    train_set = AUFERDataset(
        train_csv, label2id, region_prefixes,
        img_size=img_size, mean=mean, std=std, is_train=True
    )
    val_set = AUFERDataset(
        val_csv, label2id, region_prefixes,
        img_size=img_size, mean=mean, std=std, is_train=False
    )

    tcfg = cfg["training"]
    train_loader = DataLoader(
        train_set, batch_size=tcfg["batch_size"], shuffle=True,
        num_workers=tcfg["num_workers"], pin_memory=True,
        drop_last=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_set, batch_size=tcfg["batch_size"], shuffle=False,
        num_workers=tcfg["num_workers"], pin_memory=True,
        drop_last=False, collate_fn=collate_fn
    )

    # ── Loss ──
    loss_type = tcfg.get("loss", "focal")
    if loss_type == "focal":
        alpha = compute_class_weights(train_csv, label2id).to(device)
        criterion = FocalLoss(gamma=tcfg.get("focal_gamma", 2.0), alpha=alpha)
    else:
        alpha = compute_class_weights(train_csv, label2id).to(device)
        criterion = nn.CrossEntropyLoss(weight=alpha)

    # ── Training phases ──
    base_lr = float(tcfg["base_lr"])
    backbone_lr_scale = float(tcfg.get("backbone_lr_scale", 0.1))
    grad_clip = float(tcfg.get("grad_clip_norm", 5.0))
    freeze_epochs = int(tcfg.get("freeze_backbone_epochs", 3))
    epochs = int(tcfg["epochs"])
    use_amp = tcfg.get("amp", True) and device.type == "cuda"

    # Neutral class ID for expression magnitude scorer
    neutral_id = label2id.get("neutral", label2id.get("Neutral", -1))

    # ── Phase 1: Backbone frozen ──
    model.freeze_backbone()
    optimizer = torch.optim.AdamW(
        model.get_param_groups(base_lr, backbone_lr_scale),
        weight_decay=tcfg.get("weight_decay", 0.05)
    )
    phase1_steps = len(train_loader) * freeze_epochs
    scheduler = build_scheduler(optimizer, phase1_steps,
                                warmup_ratio=tcfg.get("warmup_ratio", 0.05))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_f1 = -1.0
    start_epoch = 1

    # Resume
    if resume:
        start_epoch, best_f1 = load_checkpoint(resume, model, optimizer, scheduler)
        start_epoch += 1
        logger.info(f"Resumed from epoch {start_epoch - 1}, best_f1={best_f1:.4f}")

    step = 0
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        t0 = time.time()

        # ── Phase transition: unfreeze backbone ──
        if epoch == freeze_epochs + 1:
            logger.info("=== Unfreezing backbone ===")
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(
                model.get_param_groups(base_lr, backbone_lr_scale),
                weight_decay=tcfg.get("weight_decay", 0.05)
            )
            # FIXED: remaining steps만으로 scheduler 재계산
            remaining_steps = len(train_loader) * (epochs - freeze_epochs)
            scheduler = build_scheduler(optimizer, remaining_steps,
                                        warmup_ratio=tcfg.get("warmup_ratio", 0.05))
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        running_loss = 0.0
        running_correct = 0
        n_samples = 0

        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            au_coords = batch["au_coords"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, global_feat = model(images, au_coords, return_features=True)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1

            # Update expression magnitude centroid
            if neutral_id >= 0:
                model.expr_scorer.update_centroid(
                    global_feat.detach(), labels, neutral_id
                )

            bs = labels.size(0)
            running_loss += loss.item() * bs
            running_correct += (logits.argmax(1) == labels).sum().item()
            n_samples += bs

        train_loss = running_loss / max(1, n_samples)
        train_acc = running_correct / max(1, n_samples)
        epoch_time = time.time() - t0

        # ── Validation ──
        val_result = evaluate(model, val_loader, device)

        log = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_result["loss"], 4),
            "val_f1": round(val_result["f1_macro"], 4),
            "val_acc": round(val_result["accuracy"], 4),
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time": round(epoch_time, 1),
        }
        metric_log.log(log)
        logger.info(
            f"[E{epoch:03d}] train_loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_result['loss']:.4f} f1={val_result['f1_macro']:.4f} "
            f"acc={val_result['accuracy']:.4f} | lr={log['lr']:.2e} | {epoch_time:.1f}s"
        )

        # ── Save last ──
        save_checkpoint(
            output_dir / "last.pth", model, optimizer, scheduler,
            epoch, best_f1, cfg
        )

        # ── Save best ──
        if val_result["f1_macro"] > best_f1:
            best_f1 = val_result["f1_macro"]
            save_checkpoint(
                output_dir / "best.pth", model, optimizer, scheduler,
                epoch, best_f1, cfg
            )
            class_names = [id2label[i] for i in range(num_classes)]
            save_confusion_matrix(
                val_result["trues"], val_result["preds"],
                class_names, output_dir / "confusion_best.png"
            )
            save_report(
                val_result["trues"], val_result["preds"],
                class_names, output_dir / "report_best.txt"
            )
            logger.info(f"  ★ New best F1: {best_f1:.4f}")

        # ── Periodic save ──
        if epoch % tcfg.get("save_every", 10) == 0:
            save_checkpoint(
                output_dir / f"epoch_{epoch:03d}.pth", model, optimizer,
                scheduler, epoch, best_f1, cfg
            )

    logger.info(f"[DONE] Best macro-F1 = {best_f1:.4f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--resume", type=str, default=None)
    args = ap.parse_args()
    train(args.config, args.resume)
