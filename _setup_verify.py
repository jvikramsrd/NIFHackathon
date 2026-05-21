"""Called by setup_venv.bat and install.sh for verification steps."""
import sys


def _get_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


mode = sys.argv[1] if len(sys.argv) > 1 else "--all"

if mode in ("--torch", "--all"):
    try:
        import torch
        d = _get_device()
        print(f"  torch {torch.__version__}  device={d}")
        if d == "cuda":
            print(f"  GPU:  {torch.cuda.get_device_name(0)}")
            p = torch.cuda.get_device_properties(0)
            print(f"  VRAM: {p.total_memory/1024**3:.1f} GB")
        elif d == "mps":
            print("  GPU:  Apple Silicon (MPS)")
    except Exception as e:
        print(f"  torch check failed: {e}")

if mode in ("--ecw", "--all"):
    try:
        from osgeo import gdal
        names = [gdal.GetDriver(i).ShortName for i in range(gdal.GetDriverCount())]
        print("  ECW driver:", "YES" if "ECW" in names else "NO (conda install -c conda-forge libgdal-ecw)")
    except Exception as e:
        print("  ECW: cannot check -", e)

if mode in ("--verify", "--all"):
    import importlib
    ok, fail = [], []

    try:
        import torch
        ok.append(("torch", torch.__version__))
        d = _get_device()
        ok.append(("device", d))
        if d == "cuda":
            ok.append(("GPU", torch.cuda.get_device_name(0)))
            p = torch.cuda.get_device_properties(0)
            ok.append(("VRAM", f"{p.total_memory/1024**3:.1f} GB"))
        elif d == "mps":
            ok.append(("GPU", "Apple Silicon (MPS)"))
    except Exception as e:
        fail.append(("torch", str(e)))

    for pkg, attr in [("timm", "__version__"), ("ultralytics", "__version__"),
                      ("segmentation_models_pytorch", "__version__"),
                      ("cv2", "__version__"), ("albumentations", "__version__")]:
        try:
            m = importlib.import_module(pkg)
            ok.append((pkg, getattr(m, attr)))
        except Exception as e:
            fail.append((pkg, str(e)))

    for pkg in ["rasterio", "geopandas", "shapely", "pyproj"]:
        try:
            m = importlib.import_module(pkg)
            ok.append((pkg, m.__version__))
        except Exception as e:
            fail.append((pkg, str(e)))

    for pkg, attr in [("numpy", "__version__"), ("pandas", "__version__"),
                      ("matplotlib", "__version__"),
                      ("tensorboard", "__version__"), ("onnx", "__version__")]:
        try:
            m = importlib.import_module(pkg)
            v = getattr(m, attr, None)
            if callable(v):
                v = v()
            ok.append((pkg, v or getattr(m, "__version__", "?")))
        except Exception as e:
            fail.append((pkg, str(e)))

    try:
        from PyQt6.QtCore import PYQT_VERSION_STR
        ok.append(("PyQt6", PYQT_VERSION_STR))
    except Exception as e:
        fail.append(("PyQt6", str(e)))

    print()
    for k, v in ok:
        print(f"  ✓  {k:<34} {v}")
    for k, v in fail:
        print(f"  ✗  {k:<34} MISSING ({v})")
    if fail:
        sys.exit(1)
