
# arc_solver/step2_compare.py
from typing import List, Dict, Tuple
import numpy as np
def compare_objects(in_objs: List[Dict], out_objs: List[Dict]) -> Dict[str, any]:
    """
    Compare input and output object lists.
    Detects simple pattern types:
      - color_map: same shapes, colors changed
      - translation: same shapes and colors, shifted positions
    Returns summary dictionary.
    """
    result = {"type": "unknown", "color_map": {}, "shift": (0, 0)}

    if len(in_objs) != len(out_objs):
        return result

    # try color remap
    color_map = {}
    same_shape_count = 0
    for i_obj, o_obj in zip(in_objs, out_objs):
        if i_obj["mask"].shape == o_obj["mask"].shape:
            same_shape_count += 1
            in_main = max(i_obj["colors"], key=i_obj["colors"].get)
            out_main = max(o_obj["colors"], key=o_obj["colors"].get)
            color_map[in_main] = out_main

    if same_shape_count == len(in_objs):
        result["type"] = "color_map"
        result["color_map"] = color_map
        return result

    # try translation
    shifts = []
    for i_obj, o_obj in zip(in_objs, out_objs):
        y1_i, x1_i, *_ = i_obj["bbox"]
        y1_o, x1_o, *_ = o_obj["bbox"]
        shifts.append((y1_o - y1_i, x1_o - x1_i))
    uniq = list(set(shifts))
    if len(uniq) == 1:
        result["type"] = "translation"
        result["shift"] = uniq[0]
        return result

    return result
