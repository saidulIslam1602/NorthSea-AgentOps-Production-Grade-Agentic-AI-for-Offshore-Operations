"""Pickle-safe multiprocessing worker for detector v1/v2 per-well runs."""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd


def evaluate_well_pair_parallel(packed: tuple[str, pd.DataFrame]):
    """Run v1 (baseline) and v2 (adaptive) for one well. See detector eval script docs."""
    from scripts.train_detector_v2 import evaluate_one_well

    well_id, well_df = packed
    wid = str(well_id)
    mix = zlib.crc32(wid.encode("utf-8")) & 0xFFFFFFFF
    rng_v1 = np.random.default_rng(int((42 ^ mix) & 0xFFFFFFFF))
    r1 = evaluate_one_well(wid, well_df, rng_v1, is_adaptive=False)
    rng2 = np.random.default_rng(seed=42)
    r2 = evaluate_one_well(wid, well_df, rng2, is_adaptive=True)
    return wid, r1, r2
