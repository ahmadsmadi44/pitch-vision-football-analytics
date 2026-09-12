"""Interpolate only short, bounded detection gaps; never invent a missing endpoint."""
from copy import deepcopy

def fill_ball_gaps(frames, max_gap_frames):
    result = deepcopy(frames)
    known = [i for i, frame in enumerate(frames) if 1 in frame]
    for left, right in zip(known, known[1:]):
        if right-left-1 > max_gap_frames:
            continue
        a, b = frames[left][1]["bbox"], frames[right][1]["bbox"]
        for i in range(left+1, right):
            fraction = (i-left)/(right-left)
            result[i] = {1: {"bbox": [x+(y-x)*fraction for x,y in zip(a,b)], "interpolated": True}}
    return result
