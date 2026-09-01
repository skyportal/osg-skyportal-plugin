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
# CVMFS); only the Arnett model is registered. :latest is CPU, :gpu is CUDA.
python register_analysis_service.py \
  --name Redback_OSG --display "Redback (OSG)" \
  --listener-url http://localhost:7100/analysis/redback_osg \
  --input-data-types photometry redshift \
  --optional-params-json '{"source": ["arnett"], "backend": ["redback"], "fix_z": ["True", "False"], "singularity_image": ["/cvmfs/singularity.opensciencegrid.org/nsarinastro/redback-jax:latest", "/cvmfs/singularity.opensciencegrid.org/nsarinastro/redback-jax:gpu", "docker://nsarinastro/redback-jax"]}'

# MOSFiT — its own image; set singularity_image to the staged mosfit .sif once the
# container build (mcoughlin/MOSFiT docker branch) is published to OSDF/CVMFS.
python register_analysis_service.py \
  --name MOSFiT_OSG --display "MOSFiT (OSG)" \
  --listener-url http://localhost:7100/analysis/mosfit_osg \
  --input-data-types photometry redshift \
  --optional-params-json '{"source": ["default", "slsn", "magnetar", "csm", "csmni", "ia", "tde", "kilonova", "nsbh", "bns"], "wrapper": ["mosfit"], "singularity_image": ["REPLACE_WITH_MOSFIT_SIF"], "fix_z": ["True", "False"]}'

# Already registered (kept here for reference / re-registration):
#   Fiesta_OSG      -> backend fiesta, /analysis/fiesta_osg
#   PeriodFind_OSG  -> wrapper periodfind, /analysis/periodfind
