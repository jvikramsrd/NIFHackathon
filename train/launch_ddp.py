"""Small torchrun entrypoint for DDP experiments.

Example:
    torchrun --standalone --nproc_per_node=2 train/launch_ddp.py --stage 1
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from utils.ddp import cleanup_ddp, setup_ddp
from utils.logger import crash_logged, get_logger

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["1", "2a"], required=True)
    args = parser.parse_args()

    state = setup_ddp()
    try:
        with crash_logged(log, f"DDP stage {args.stage}"):
            if args.stage == "1":
                from train.train_stage1 import train_stage1

                train_stage1()
            else:
                from train.train_stage2 import train_stage2a

                train_stage2a()
    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()
