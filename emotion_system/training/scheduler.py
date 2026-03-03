"""
LR Scheduler
==============
Warmup + Cosine Decay.
기존 코드 문제: backbone unfreeze 시 total_steps 재계산 안 함.
"""

import math
import torch


def build_scheduler(optimizer, total_steps: int, warmup_ratio: float = 0.05,
                    min_lr: float = 1e-6):
    """
    Cosine decay with linear warmup.

    Args:
        optimizer: optimizer instance
        total_steps: total training steps (for THIS phase)
        warmup_ratio: fraction of steps for warmup
        min_lr: minimum learning rate
    """
    warmup_steps = int(total_steps * warmup_ratio)

    # Get base LR from first param group
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    def lr_lambda_factory(base_lr):
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(min_lr / base_lr, cosine)
        return lr_lambda

    # Create per-group lambda (handles different LRs for backbone vs head)
    lambdas = [lr_lambda_factory(lr) for lr in base_lrs]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambdas)
    return scheduler
