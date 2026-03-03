#!/usr/bin/env python3
"""Training entry point."""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.trainer import train


def main():
    ap = argparse.ArgumentParser(description="Train AU-FER Model")
    ap.add_argument("--config", type=str, required=True,
                    help="Path to config YAML")
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint to resume from")
    args = ap.parse_args()
    train(args.config, args.resume)


if __name__ == "__main__":
    main()
