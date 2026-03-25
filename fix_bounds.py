import re

with open("data/preprocessing.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix bounds in infra extraction
infra_old = """                pt = geom.centroid
                pr, pc = src.index(pt.x, pt.y)
                tr = (int(pr) // tile_size) * tile_size
                tc = (int(pc) // tile_size) * tile_size
                tile_groups[(tr, tc)].append((cid, int(pc), int(pr)))
                total += 1"""

infra_new = """                pt = geom.centroid
                pr, pc = src.index(pt.x, pt.y)
                if pr < 0 or pr >= H or pc < 0 or pc >= W:
                    continue
                tr = (int(pr) // tile_size) * tile_size
                tc = (int(pc) // tile_size) * tile_size
                tile_groups[(tr, tc)].append((cid, int(pc), int(pr)))
                total += 1"""

if infra_old in content:
    content = content.replace(infra_old, infra_new)

with open("data/preprocessing.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Fixed out-of-bounds geometries.")
