from __future__ import annotations
import numpy as np

def evaluate_extra(cfg, run_dir, y, prob):
    asr_ceiling = float(np.mean(prob > 0.2))
    asr_blackbox = float(np.mean(prob > 0.8))
    headroom = asr_ceiling - asr_blackbox
    return {
        "asr_blackbox": asr_blackbox,
        "asr_ceiling": asr_ceiling,
        "headroom": headroom,
        "manifold_distance_mean": float(np.mean(np.abs(prob-0.5))),
        "falsification_threshold": 0.05,
        "world2_if_headroom_below": 0.05,
    }

