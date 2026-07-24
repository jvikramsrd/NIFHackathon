import pytest


def _make_test_raster(path, crs):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    data = np.zeros((3, 32, 32), dtype=np.uint8)
    transform = from_origin(1_000_000, 1_000_032, 1, 1)
    profile = {
        "driver": "GTiff",
        "height": 32,
        "width": 32,
        "count": 3,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
    }

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)

    return transform


def test_recursive_discovery_and_crs_rasterization(tmp_path):
    pytest.importorskip("geopandas")
    pytest.importorskip("rasterio")
    pytest.importorskip("shapely")
    pytest.importorskip("pyproj")

    import geopandas as gpd
    import numpy as np
    import rasterio
    from pyproj import Transformer
    from shapely.geometry import box

    from data.preprocessing import (
        discover_processing_units,
        process_raster,
        _load_vector_layer,
        _rasterize_window,
    )

    root = tmp_path / "dataset"
    nested = root / "cg" / "village_a"
    nested.mkdir(parents=True)

    raster_path = nested / "scene.tif"
    transform = _make_test_raster(raster_path, "EPSG:3857")

    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    minx, miny = transformer.transform(1_000_000, 1_000_000)
    maxx, maxy = transformer.transform(1_000_032, 1_000_032)
    midx = (minx + maxx) / 2.0
    midy = (miny + maxy) / 2.0

    building_gdf = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[box(minx, miny, midx, midy)],
        crs="EPSG:4326",
    )
    road_gdf = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[box(midx, midy, maxx, maxy)],
        crs="EPSG:4326",
    )

    building_shp = nested / "Built_Up_Area_type.shp"
    road_shp = nested / "Road.shp"
    building_gdf.to_file(building_shp)
    road_gdf.to_file(road_shp)

    (nested / "scene.tif.pyrx").write_text("noise", encoding="utf-8")
    (nested / "scene.tif.aux.xml").write_text("noise", encoding="utf-8")

    units = discover_processing_units(root)
    assert len(units) == 1
    assert units[0].rasters == [raster_path]
    assert set(units[0].shapefiles) == {building_shp, road_shp}

    with rasterio.open(raster_path) as src:
        building_layer = _load_vector_layer(building_shp, src.crs, 1)
        road_layer = _load_vector_layer(road_shp, src.crs, 2)
        assert building_layer is not None
        assert road_layer is not None
        mask = _rasterize_window(
            src,
            [building_layer, road_layer],
            rasterio.windows.Window(0, 0, src.width, src.height),
            32,
        )

    assert mask.shape == (32, 32)
    assert mask.dtype == np.uint8
    assert mask.max() == 2
    assert np.count_nonzero(mask == 1) > 0
    assert np.count_nonzero(mask == 2) > 0

    summary = process_raster(
        raster_path=raster_path,
        shapefiles=units[0].shapefiles,
        output_root=root,
        role_map={"Built_Up_Area_type": 1, "Road": 2},
        patch_size=32,
        overlap=0.25,
    )
    assert summary["saved"] >= 1
    assert (root / "patches").exists()
    assert (root / "patch_masks").exists()
