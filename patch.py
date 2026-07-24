import sys

def main():
    path = "c:/NIFHackathon/inference/pipeline.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    
    target = (
        "        # Cache the cosine blending window once. _segment() was recomputing this\n"
        "        # on every call (a constant given (patch_size, overlap)) — the cost is\n"
        "        # small but it's pure waste.\n"
        "        from utils.window import cosine_window\n"
        "        _ps = int(CFG.STAGE1.get(\"patch_size\", 512))\n"
        "        _ov = int(CFG.STAGE1.get(\"overlap\", 128))\n"
        "        self._seg_window = cosine_window(_ps, _ov).astype(np.float32)"
    )
    
    replacement = "        # utils/inference.py handles gaussian_window internally"
    
    if target in content:
        new_content = content.replace(target, replacement)
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print("Success: Replaced target block.")
    else:
        print("Error: Target block not found.")

if __name__ == "__main__":
    main()
