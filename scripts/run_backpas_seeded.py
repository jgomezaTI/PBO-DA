"""Run the upstream BackPaS trainer with deterministic process state."""

from __future__ import annotations

import argparse
import os
import random
import runpy
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.arguments and args.arguments[0] == "--":
        args.arguments = args.arguments[1:]
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    random.seed(args.seed)

    import numpy as np
    import torch

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    sys.path.insert(0, str(args.script.parent))
    sys.argv = [str(args.script), *args.arguments]
    runpy.run_path(str(args.script), run_name="__main__")


if __name__ == "__main__":
    main()
