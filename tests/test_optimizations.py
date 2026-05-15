import numpy as np
import torch


def test_class_weights_matches_brute_force():
    """bincount result must be identical to the old per-sample loop."""
    from data.dataset import RooftopDataset
    from pathlib import Path

    # Build a minimal samples list without touching disk
    class_names = ["RCC", "Tiled", "Tin", "Other"]
    samples = [(Path("x.png"), 0)] * 10 + [(Path("x.png"), 1)] * 5 + \
              [(Path("x.png"), 2)] * 3 + [(Path("x.png"), 3)] * 2

    ds = RooftopDataset.__new__(RooftopDataset)
    ds.samples = samples
    ds.class_names = class_names

    # Brute-force reference
    counts = torch.zeros(len(class_names))
    for _, lbl in samples:
        counts[lbl] += 1
    w_ref = 1.0 / (counts + 1e-6)
    w_ref = w_ref / w_ref.sum() * len(class_names)

    w_got = ds.class_weights()
    assert torch.allclose(w_ref, w_got, atol=1e-5), f"mismatch: {w_ref} vs {w_got}"
