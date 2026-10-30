#!/usr/bin/env python3
"""E5 exploratory LoRA for DINOv3, Rad-JEPA-3D and CT-CLIP.

Thin entry point around run_lung1_e5_lora.py, which accepts only COLIPRI and TANGERINE; the
training, prediction and evaluation code and its protocol hash are reused unchanged.
"""
import argparse
from pathlib import Path

import torch

import run_lung1_e5_lora as e5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--precision', choices=('bf16-mixed', '16-mixed', '32-true'), default='bf16-mixed')
    parser.add_argument('--model', choices=('dinov3', 'radjepa', 'ctclip'), required=True)
    parser.add_argument('--stage', choices=('train', 'predict', 'evaluate'), required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    e5.PRECISION = args.precision
    output = args.output_root / args.model
    output.mkdir(parents=True, exist_ok=True)
    if args.stage != 'evaluate':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required')
        if args.precision == 'bf16-mixed' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('BF16 unsupported; choose FP16 or FP32')
    getattr(e5, args.stage)(output, args.model)


if __name__ == '__main__':
    main()
