import math

import pytest
import torch


def test_sam_optimizer_updates_parameters():
    from utils.sam import SAM

    model = torch.nn.Linear(2, 1)
    before = model.weight.detach().clone()
    base = torch.optim.SGD(model.parameters(), lr=0.1)
    opt = SAM(base, rho=0.05, adaptive=False)
    x = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
    y = torch.tensor([[1.0], [0.0]])
    loss_fn = torch.nn.MSELoss()

    loss = loss_fn(model(x), y)
    loss.backward()
    opt.first_step(zero_grad=True)
    loss_fn(model(x), y).backward()
    opt.second_step(zero_grad=True)

    assert not torch.allclose(before, model.weight.detach())


def test_boundary_hausdorff_loss_is_finite():
    pytest.importorskip("segmentation_models_pytorch")
    from models.stage1_segmentation import TriLoss

    loss_fn = TriLoss(
        num_classes=2,
        dice_weight=0.4,
        ce_weight=0.2,
        focal_weight=0.2,
        boundary_weight=0.2,
        class_weights=[1.0, 1.0],
    )
    logits = torch.randn(1, 2, 16, 16, requires_grad=True)
    targets = torch.zeros(1, 16, 16, dtype=torch.long)
    targets[:, 4:12, 4:12] = 1

    loss = loss_fn(logits, targets)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None


def test_obb_label_format_has_four_points():
    pytest.importorskip("geopandas")
    pytest.importorskip("rasterio")
    from data.preprocessing import _format_yolo_label

    label = _format_yolo_label(1, 0.5, 0.5, 0.2, 0.4, use_obb=True)
    parts = label.split()

    assert parts[0] == "1"
    assert len(parts) == 9
    assert all(0.0 <= float(v) <= 1.0 for v in parts[1:])


def test_context_polygons_are_pixel_gates():
    pytest.importorskip("geopandas")
    pytest.importorskip("rasterio")
    from affine import Affine
    import geopandas as gpd
    from shapely.geometry import box

    from inference.pipeline import GeoIntelPipeline

    pipe = GeoIntelPipeline.__new__(GeoIntelPipeline)
    gdf = gpd.GeoDataFrame({"class_id": [1]}, geometry=[box(10, -20, 20, -10)], crs="EPSG:3857")
    gates = pipe._gather_context_polygons(".", "missing", gdf, Affine.identity())

    assert gates
    assert gates[0].intersects(box(10, -20, 20, -10))


def test_clf_train_transforms_include_fog_and_sunflare():
    from data.dataset import get_clf_train_transforms
    import albumentations as A

    tf = get_clf_train_transforms(224)
    type_names = [type(t).__name__ for t in tf.transforms]

    assert "RandomFog" in type_names, "RandomFog missing from clf train transforms"
    assert "RandomSunFlare" in type_names, "RandomSunFlare missing from clf train transforms"


def test_sahi_overlap_reads_from_config():
    import pathlib
    src = pathlib.Path("models/stage2_models.py").read_text(encoding="utf-8")
    assert "overlap_height_ratio=0.30" not in src, \
        "overlap_height_ratio is hardcoded 0.30 — should read from cfg['sahi_overlap_ratio']"
