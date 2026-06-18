"""
Mock a completed analysis (with a model_lightcurve overlay) on a SkyPortal
object — for testing the photometry-plot overlay without running a real fit.

Uses the `upload_only` analysis path: creates/reuses an upload_only
AnalysisService, synthesizes a per-filter model light curve from the object's
own photometry (a low-order polynomial through the detections, so it tracks the
data), and POSTs it. Prints the new analysis id.

  python bin/mock_analysis_result.py --obj-id ZTF20abwysqy \
      --token-file /path/to/skyportal/.tokens.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import requests


def _resolve_token(args) -> str:
    if args.token:
        return args.token
    if args.token_file and Path(args.token_file).exists():
        # .tokens.yaml: "INITIAL_ADMIN: <uuid>"
        for line in Path(args.token_file).read_text().splitlines():
            if ":" in line:
                return line.split(":", 1)[1].strip()
    sys.exit("need --token or --token-file pointing at a .tokens.yaml")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj-id", required=True)
    ap.add_argument("--base-url", default="http://localhost:5000")
    ap.add_argument("--token")
    ap.add_argument("--token-file")
    ap.add_argument("--service-name", default="Mock_Fiesta")
    ap.add_argument(
        "--models",
        default="AfterglowModel,ArnettModel,Bu2025_MLP",
        help="comma-separated models; each uploaded as its own analysis with a distinct shape",
    )
    ap.add_argument("--n-points", type=int, default=80)
    ap.add_argument(
        "--bands",
        default="afterglow=0.05,sn=0.05,arnett=0.05,nickel=0.05,kn=0.5,bu20=0.5,kilonova=0.5",
        help="per-model-family credible-band half-width in mag, as substr=mag pairs",
    )
    args = ap.parse_args()
    band_overrides = {}
    for kv in args.bands.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            band_overrides[k.strip().lower()] = float(v)

    def band_for(model: str) -> float:
        m = model.lower()
        for key, val in band_overrides.items():
            if key in m:
                return val
        return 0.2  # default for unrecognized models

    H = {"Authorization": f"token {_resolve_token(args)}"}
    B = args.base_url.rstrip("/")

    def get(path):
        r = requests.get(f"{B}{path}", headers=H, timeout=30)
        r.raise_for_status()
        return r.json()["data"]

    gids = [g["id"] for g in (get("/api/groups").get("user_groups") or [])]

    # Find or create an upload_only service.
    svcs = get("/api/analysis_service")
    svc = next((s for s in svcs if s["name"] == args.service_name), None)
    if svc:
        sid = svc["id"]
    else:
        r = requests.post(
            f"{B}/api/analysis_service",
            headers=H,
            timeout=30,
            json={
                "name": args.service_name,
                "display_name": args.service_name,
                "url": "http://localhost/none",  # unused for upload_only
                "authentication_type": "none",
                "analysis_type": "lightcurve_fitting",
                "input_data_types": ["photometry"],
                "upload_only": True,
                "group_ids": gids,
            },
        )
        r.raise_for_status()
        sid = r.json()["data"]["id"]
        print(f"created upload_only service id={sid}")

    # Per-filter time grid + magnitude level from the object's photometry, so
    # every synthesized model lands on the data cloud.
    phot = get(f"/api/sources/{args.obj_id}/photometry")
    by_filter: dict[str, list] = {}
    for p in phot:
        if p.get("mjd") is not None:
            by_filter.setdefault(p["filter"], []).append(p)

    base: dict[str, dict] = {}
    for filt, pts in by_filter.items():
        mjds = np.array([p["mjd"] for p in pts])
        mags = [p["mag"] for p in pts if p.get("mag") is not None]
        lims = [p["limiting_mag"] for p in pts if p.get("limiting_mag") is not None]
        level = float(np.median(mags)) if mags else (float(np.median(lims)) if lims else 20.5)
        base[filt] = {
            "grid": np.linspace(float(mjds.min()), float(mjds.max()), args.n_points),
            "level": level,
        }
    if not base:
        sys.exit("no photometry filters to synthesize a curve from")

    # Distinct curve SHAPE per model family, as delta-mag over u in [0,1]
    # (negative = brighter). Keeps the multi-model overlay visually distinct.
    def shape(model: str, u: np.ndarray) -> np.ndarray:
        m = model.lower()
        if "afterglow" in m or "grb" in m:
            return -1.0 + 2.2 * u  # power-law-ish monotonic fade
        if "kn" in m or "bu20" in m or "kilonova" in m:
            return -1.0 * np.exp(-3.0 * u)  # fast early peak, rapid decline
        if "arnett" in m or "sn" in m or "nickel" in m:
            return -0.9 * np.exp(-0.5 * ((u - 0.3) / 0.18) ** 2)  # SN bump
        return -0.7 * np.exp(-0.5 * ((u - 0.4) / 0.2) ** 2)  # generic bump

    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        band = band_for(model)
        mlc: dict[str, list] = {}
        for filt, b in base.items():
            grid = b["grid"]
            u = (grid - grid.min()) / max(grid.max() - grid.min(), 1e-9)
            med = b["level"] + shape(model, u)
            mlc[filt] = [
                [float(t), float(mm), float(mm - band), float(mm + band)]
                for t, mm in zip(grid, med)
            ]
        body = {
            "analysis": {
                "model_lightcurve": mlc,
                "model_name": model,
                "results": {"model": model, "filters": list(mlc), "note": "synthetic mock fit"},
            },
            "group_ids": gids,
            "show_plots": True,
            "show_parameters": True,
            "show_corner": False,
            "message": f"mock {model} ({', '.join(mlc)})",
        }
        r = requests.post(
            f"{B}/api/obj/{args.obj_id}/analysis_upload/{sid}", headers=H, json=body, timeout=60
        )
        if not r.ok:
            print(f"  {model}: {r.status_code} {r.text[:200]}")
            continue
        print(f"OK — {model}: analysis id={r.json()['data']['id']}, filters={list(mlc)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
