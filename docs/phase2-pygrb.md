# Phase 2: the production PyGRB engine (`pycbc_multi_inspiral`)

Phase 1 (`pygrb_bridge._coherent_search`) is a self-contained coherent search
using PyCBC's Python API + a small bank. It recovers GW170817 coherently
(SNR ~20 at the merger, correct chirp mass from a KN-constrained bank) and
produces the trigger-distribution plot, but it does **not** produce a formal
FAR/significance and caps below the deck's ~28 (single-template, no χ² reweight,
no time-slide background).

Phase 2 replaces the engine with `pycbc_multi_inspiral` — the production PyGRB
coherent statistic (dominant-polarization projection + χ² + null-stream vetoes,
reweighted SNR, time-slide background → FAR). This note records what was
validated so the implementation is a fill-in, not a rediscovery.

## Validated `pycbc_multi_inspiral` invocation (GW170817, H1L1)

Everything below was confirmed correct against `pycbc-el8:latest` — the run
reaches template generation with these exact args:

```
pycbc_multi_inspiral --instruments H1 L1 --trigger-time 1187008882 \
  --ra 3.446150 --dec -0.408084          # RADIANS, not degrees
  --bank-file bank.hdf --approximant TaylorF2 \
  --frame-files H1:H1.gwf L1:L1.gwf --channel-name H1:GWOSC-STRAIN L1:GWOSC-STRAIN \
  --gps-start-time H1:T0-350 ... --gps-end-time H1:T0+350 ... \
  --trig-start-time H1:T0-6 ... --trig-end-time H1:T0+6 ... \
  --sample-rate H1:2048 ... --segment-length H1:512 ... \
  --segment-start-pad H1:112 ... --segment-end-pad H1:16 ... --pad-data H1:8 ... \
  --low-frequency-cutoff 25 --psd-estimation H1:median ... \
  --psd-segment-length H1:256 ... --psd-segment-stride H1:128 ... \
  --sngl-snr-threshold 4.0 --chisq-bins 16 \
  --cluster-method window --cluster-window 0.1 --projection standard \
  --output out.hdf
```

Gotchas found (each cost an iteration):

- **`--trigger-time` is an int** (GPS seconds).
- **`--ra`/`--dec` are radians.** The pygrb bridge already has RA/Dec in radians
  (`_resolve_sky`) — reuse that.
- Per-IFO args (`--gps-start-time IFO:T`, `--sample-rate IFO:RATE`, ...) must be
  given for every instrument, and `--segment-start-pad`/`--segment-end-pad` are
  **required** (not just `--pad-data`).
- The **bank needs `template_duration`** precomputed, else the max-length step
  calls `spa_length_in_time` with `phase_order=None` and crashes. Add it with
  `pycbc.waveform.get_waveform_filter_length_in_time("TaylorF2", mass1=, mass2=,
  f_lower=, phase_order=7)`.
- Even so, **per-template generation still needs the PN phase order**, which a
  hand-assembled bank + bare CLI does not propagate (`phase_order=None`). This is
  the point where a hand-rolled call stops.

## The right Phase-2 path: the PyGRB offline workflow

The `phase_order` propagation and the whole on/off-source + time-slide background
+ significance are handled by the **PyGRB offline workflow**, not a single CLI
call:

- `pycbc_make_offline_grb_workflow` (present in the image) builds the DAG:
  bank → `pycbc_multi_inspiral` on-source + off-source + slides →
  `pycbc_pygrb_*` post-processing → the reweighted-SNR trigger distribution and
  FAR. This is exactly Marion Pillas & Noah Jamsin's `run.sh`
  (`/home/marion.pillas/pygrb/test_KN_targeted/run.sh`).
- The waveform PN order, χ² config, veto thresholds, etc. live in the workflow's
  `.ini` (the `[inspiral]`/`[workflow-*]` sections), which is why the bare CLI is
  missing them.

**Recommended implementation.** Wrap the PyGRB workflow rather than a single
`pycbc_multi_inspiral` call:

1. Start from Marion's `run.sh` / its `.ini` as the template.
2. Parameterize it from the SkyPortal trigger: `--ra`/`--dec` from the source,
   trigger time from the KN T0 (bridge already resolves this), and the bank from
   the KN-constrained chirp-mass range (`_build_bank`, deck Test 2).
3. Point the data at **gwdatafind/OSDF `/frames`** using the AP-issued SciToken
   (see `deploy-igwn.md`) for non-public strain — the token path is proven.
4. Parse the workflow's output for the reweighted-SNR triggers + FAR; feed the
   existing `_plot` (extend the y-axis to reweighted SNR, add the background).

Because the workflow is itself a DAG, on OSG this is either a sub-DAG or a single
job that runs the workflow's stages in sequence (the KN-targeted search over one
trigger + a modest off-source is small enough for one node).

## Data note

Frame staging via `pycbc.frame.write_frame` is flaky on a shared login node
(I/O/quota); on OSG execute nodes with real scratch + gwdatafind-provided frames
this is the standard path. V1's GW170817 `write_frame` errored locally, so the
validation used H1L1 (V1 contributes ~5 SNR anyway).
