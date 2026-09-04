#!/usr/bin/env bash
# Register the OSG analysis services with SkyPortal via the canonical helper
# (register_analysis_service.py). Run where baselayer's load_env can read the
# deployment config — i.e. the Fritz web pod — so base_url / api_token /
# incoming_bearer_token come from services.external.osg.params (no secrets on the
# CLI). Set PYTHONPATH to the skyportal checkout first.
#
#   PYTHONPATH=/path/to/skyportal ./bin/register-osg-services.sh
#
# Each <name>_osg service routes to its engine by name (main.py's AnalysisHandler
# defaults backend/wrapper from the analysis name), so the backend/wrapper params
# below are belt-and-suspenders and can be dropped once that routing is deployed.
set -euo pipefail
: "${PYTHONPATH:?set PYTHONPATH to the skyportal checkout so load_env works}"
cd "$(dirname "$0")/.."

# Redback (JAX) — runs in its own redback-jax image (nsarinastro/redback-jax on
# CVMFS). Models complement Fiesta (central-engine + shock-cooling SNe, which
# Fiesta lacks). The SMC fit is slow on CPU (~1 hr), so it defaults to GPU
# (:gpu image + request_gpus=1): seconds-to-minutes and ~1 CPU core. :latest is
# the CPU fallback. Fits peak ~17 GB, so request_memory defaults high (MB).
python register_analysis_service.py \
  --name Redback_OSG --display "Redback (OSG)" \
  --listener-url http://localhost:7100/analysis/redback_osg \
  --input-data-types photometry redshift \
  --optional-params-json '{"source": ["arnett", "magnetar", "magnetar_nickel", "shock_cooling"], "backend": ["redback"], "fix_z": ["True", "False"], "singularity_image": ["/cvmfs/singularity.opensciencegrid.org/nsarinastro/redback-jax:gpu", "/cvmfs/singularity.opensciencegrid.org/nsarinastro/redback-jax:latest", "docker://nsarinastro/redback-jax"], "request_gpus": {"type": "number", "default": 1}, "request_cpus": {"type": "number", "default": 4}, "request_memory": {"type": "number", "default": 24576}}'

# MOSFiT — its own image (ashleyvillar/mosfit, built from the MOSFiT docker
# branch: mosfit + CPU torch + baked ZTF filters). docker:// works per-job now;
# prefer the CVMFS mirror once ashleyvillar/mosfit is added to the OSG sync list.
python register_analysis_service.py \
  --name MOSFiT_OSG --display "MOSFiT (OSG)" \
  --listener-url http://localhost:7100/analysis/mosfit_osg \
  --input-data-types photometry redshift \
  --optional-params-json '{"source": ["default", "slsn", "magnetar", "csm", "csmni", "ia", "tde", "kilonova", "nsbh", "bns"], "wrapper": ["mosfit"], "singularity_image": ["docker://ashleyvillar/mosfit", "/cvmfs/singularity.opensciencegrid.org/ashleyvillar/mosfit:latest"], "fix_z": ["True", "False"]}'

# NGSF — the one service taking spectra, not photometry (spectrum_fitting). Its
# image bakes in the template bank; the fit is ~4-5 min on one core, <1 GB.
python register_analysis_service.py \
  --name NGSF_OSG --display "NGSF (OSG)" \
  --listener-url http://localhost:7100/analysis/ngsf_osg \
  --analysis-type spectrum_fitting \
  --input-data-types spectra redshift \
  --optional-params-json '{"wrapper": ["ngsf"], "singularity_image": ["/cvmfs/singularity.opensciencegrid.org/michaelwcoughlin/ngsf:latest", "docker://michaelwcoughlin/ngsf"], "spectrum_index": {"type": "number"}, "instrument": ["", "NGPS", "GHTS"], "n_results": {"type": "number", "default": 3}, "request_cpus": {"type": "number", "default": 1}, "request_memory": {"type": "number", "default": 2048}}'

# Already registered (kept here for reference / re-registration):
#   Fiesta_OSG      -> backend fiesta, /analysis/fiesta_osg
#   PeriodFind_OSG  -> wrapper periodfind, /analysis/periodfind
