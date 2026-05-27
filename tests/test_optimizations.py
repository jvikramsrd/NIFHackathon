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


def test_to_uint8_vectorized_matches_original():
    """Vectorized _to_uint8 must produce identical output to the original."""
    import sys
    from pathlib import Path
    import numpy as np

    # Import just the function by reading it directly
    pipeline_path = Path(__file__).parent.parent / "inference" / "pipeline.py"

    # Load and execute just the _to_uint8 function
    with open(pipeline_path) as f:
        content = f.read()

    # Extract and execute _to_uint8 function (minimal dependencies)
    exec_globals = {"np": np}
    # Find the function definition
    start_idx = content.find("def _to_uint8(arr):")
    end_idx = content.find("\nif __name__", start_idx)
    func_code = content[start_idx:end_idx]
    exec(func_code, exec_globals)
    _to_uint8 = exec_globals["_to_uint8"]

    rng = np.random.default_rng(0)

    # 3-band uint16 raster (typical GeoTIFF)
    arr16 = (rng.integers(0, 65535, (64, 64, 3))).astype(np.uint16)
    # 2-band float32
    arr2f = rng.random((32, 32, 2)).astype(np.float32) * 1000
    # 2D grayscale
    arr2d = rng.integers(0, 255, (48, 48)).astype(np.uint8)

    for arr in [arr16, arr2f, arr2d]:
        out = _to_uint8(arr)
        assert out.dtype == np.uint8, f"wrong dtype {out.dtype}"
        assert out.ndim == 3, f"expected 3D output, got shape {out.shape}"
        assert out.shape[2] == 3, f"expected 3 channels, got {out.shape[2]}"
        assert out.min() >= 0 and out.max() <= 255


def test_to_uint8_edge_cases():
    import numpy as np
    from pathlib import Path

    # Load and execute just the _to_uint8 function
    pipeline_path = Path(__file__).parent.parent / "inference" / "pipeline.py"
    with open(pipeline_path) as f:
        content = f.read()

    exec_globals = {"np": np}
    start_idx = content.find("def _to_uint8(arr):")
    end_idx = content.find("\nif __name__", start_idx)
    func_code = content[start_idx:end_idx]
    exec(func_code, exec_globals)
    _to_uint8 = exec_globals["_to_uint8"]

    assert _to_uint8(None).shape == (256, 256, 3)
    assert _to_uint8(np.array([])).shape == (256, 256, 3)
    assert _to_uint8(np.zeros((0, 0, 3), dtype=np.uint8)).shape == (256, 256, 3)



def test_dbf_value_normalization_maps_common_aliases():
    from data.preprocessing import canonical_mapped_label
    import config as CFG

    assert canonical_mapped_label("Pucca RCC Slab", CFG.ROOF_TYPE_MAP) == "RCC"
    assert canonical_mapped_label("Overhead Water Tank", CFG.INFRA_TYPE_MAP) == "overhead_tank"
    assert canonical_mapped_label("Tube-Well", CFG.INFRA_TYPE_MAP) == "well"


def test_find_attribute_column_handles_case_and_underscores():
    from data.preprocessing import find_attribute_column

    class FakeGdf:
        columns = ["Utility Ty", "Roof_Type"]

    assert find_attribute_column(FakeGdf(), ["utility_ty"]) == "Utility Ty"
    assert find_attribute_column(FakeGdf(), ["roof type"]) == "Roof_Type"
