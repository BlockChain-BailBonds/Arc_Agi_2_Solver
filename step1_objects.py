# arc_solver/step1_objects.py
from typing import List, Dict, Tuple
import numpy as np
from arc_solver.step0_utils import ensure_integrity

def find_objects(grid: np.ndarray) -> List[Dict]:
    """
    Find connected non-zero pixel clusters (4-neighbor).
    Adds:
      • bbox
      • centroid (y,x)
      • dominant color
      • rotation / mirror signatures
    """
    g = ensure_integrity(grid)
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    objects = []
    obj_id = 0

    for y in range(h):
        for x in range(w):
            if g[y, x] == 0 or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            pixels = []
            color_counts = {}
            y1 = y2 = y
            x1 = x2 = x

            while stack:
                cy, cx = stack.pop()
                pixels.append((cy, cx))
                val = int(g[cy, cx])
                color_counts[val] = color_counts.get(val, 0) + 1
                y1, y2 = min(y1, cy), max(y2, cy)
                x1, x2 = min(x1, cx), max(x2, cx)
                for ny, nx in ((cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)):
                    if 0<=ny<h and 0<=nx<w and not visited[ny,nx] and g[ny,nx]!=0:
                        visited[ny,nx]=True
                        stack.append((ny,nx))

            mask = np.zeros((y2-y1+1, x2-x1+1), bool)
            for py, px in pixels:
                mask[py-y1, px-x1] = True

            dom_color = max(color_counts, key=color_counts.get)
            centroid = (
                float(np.mean([p[0] for p in pixels])),
                float(np.mean([p[1] for p in pixels]))
            )

            # rotation/mirror signatures for later matching
            sig_rot90 = np.rot90(mask)
            sig_mirror = np.fliplr(mask)

            objects.append({
                "id": obj_id,
                "bbox": (y1, x1, y2+1, x2+1),
                "mask": mask,
                "colors": color_counts,
                "dominant": dom_color,
                "centroid": centroid,
                "sig_rot90": sig_rot90,
                "sig_mirror": sig_mirror,
            })
            obj_id += 1

    return objects
