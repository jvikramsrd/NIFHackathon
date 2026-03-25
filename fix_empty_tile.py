with open("data/preprocessing.py", "r", encoding="utf-8") as f:
    content = f.read()

old_code = """                if tile.dtype != np.uint8:
                    tile = _to_uint8(tile)
            except Exception:
                continue

            name = f"infra_{tr:06d}_{tc:06d}\""""

new_code = """                if tile.dtype != np.uint8:
                    tile = _to_uint8(tile)
            except Exception:
                continue

            if tile is None or tile.size == 0 or 0 in tile.shape:
                continue

            name = f"infra_{tr:06d}_{tc:06d}\""""

if old_code in content:
    content = content.replace(old_code, new_code)
    with open("data/preprocessing.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Fixed empty tile.")
else:
    print("Not found.")
