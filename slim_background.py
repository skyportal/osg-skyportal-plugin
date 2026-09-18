#!/usr/bin/env python
"""Shrink an aframe timeslide-background hdf5 for cheap OSG staging.

The FAR step only reads ``parameters/detection_statistic`` and the ``Tb`` /
``length`` attrs; the raw file also carries ``detection_time`` and ``shift``
(unused at inference), so a full background is several times larger than needed.

Modes:
- default (full): keep every statistic; drop the unused datasets.
- ``--top-k N``:   keep only the N loudest statistics (sorted, descending). Tiny,
  but only valid for candidates at/above ``tail_min``; the file records
  ``is_tail`` and ``tail_min`` so the FAR consumer can flag out-of-tail scores.

``--float32`` halves the statistic array; the sub-ULP change is irrelevant to a
``>=`` count. Everything is read in chunks so the source is never loaded whole.
"""

from __future__ import annotations

import argparse

import h5py
import numpy as np

CHUNK = 20_000_000


def _copy_scalar_attrs(src, dst):
    dst.attrs["Tb"] = float(src.attrs["Tb"])
    dst.attrs["length"] = int(src.attrs["length"])


def slim_full(src, dst, dtype):
    n = src["parameters/detection_statistic"].shape[0]
    grp = dst.create_group("parameters")
    out = grp.create_dataset(
        "detection_statistic",
        shape=(n,),
        dtype=dtype,
        chunks=(min(CHUNK, n),),
        compression="gzip",
        compression_opts=4,
    )
    ds = src["parameters/detection_statistic"]
    for i in range(0, n, CHUNK):
        out[i : i + CHUNK] = ds[i : i + CHUNK].astype(dtype, copy=False)


def slim_topk(src, dst, dtype, k):
    ds = src["parameters/detection_statistic"]
    n = ds.shape[0]
    k = min(k, n)
    top = np.full(k, -np.inf)
    for i in range(0, n, CHUNK):
        merged = np.concatenate([top, ds[i : i + CHUNK]])
        if merged.size > k:
            top = np.partition(merged, merged.size - k)[merged.size - k :]
        else:
            top = merged
    top = np.sort(top)[::-1].astype(dtype, copy=False)  # loudest first

    grp = dst.create_group("parameters")
    grp.create_dataset("detection_statistic", data=top, compression="gzip")
    dst.attrs["is_tail"] = 1
    dst.attrs["tail_min"] = float(top.min())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="keep only the N loudest statistics (tail mode)",
    )
    p.add_argument(
        "--float32",
        action="store_true",
        help="store statistics as float32 (half the size)",
    )
    args = p.parse_args()
    dtype = np.float32 if args.float32 else np.float64

    with h5py.File(args.input, "r") as src, h5py.File(args.output, "w") as dst:
        _copy_scalar_attrs(src, dst)
        if args.top_k:
            slim_topk(src, dst, dtype, args.top_k)
        else:
            slim_full(src, dst, dtype)

    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
