"""Evaluation utilities."""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from pathlib import Path


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_trues, all_losses = [], [], []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        au_coords = batch["au_coords"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        logits = model(images, au_coords)
        loss = F.cross_entropy(logits, labels)
        all_losses.append(loss.item())

        preds = logits.argmax(dim=1).cpu().numpy()
        trues = labels.cpu().numpy()
        all_preds.extend(preds)
        all_trues.extend(trues)

    f1 = f1_score(all_trues, all_preds, average="macro")
    acc = accuracy_score(all_trues, all_preds)
    avg_loss = np.mean(all_losses)

    return {
        "loss": avg_loss,
        "f1_macro": f1,
        "accuracy": acc,
        "preds": all_preds,
        "trues": all_trues,
    }


def save_confusion_matrix(trues, preds, class_names, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cm = confusion_matrix(trues, preds, labels=list(range(len(class_names))))
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title("Confusion Matrix", fontsize=14)
        fig.colorbar(im, ax=ax)

        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(class_names)

        # Annotate cells
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], "d"),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=9)

        ax.set_ylabel("True", fontsize=12)
        ax.set_xlabel("Predicted", fontsize=12)
        plt.tight_layout()
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] save_confusion_matrix failed: {e}")


def save_report(trues, preds, class_names, out_path):
    report = classification_report(
        trues, preds, target_names=class_names, digits=4
    )
    Path(out_path).write_text(report, encoding="utf-8")
    return report
