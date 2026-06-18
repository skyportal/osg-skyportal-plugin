# Lean fiesta-only runtime image for the OSG plugin (pure-Fiesta path).
#
# Docker/OCI build, for CVMFS distribution via the OSG Harbor registry
# (hub.opensciencegrid.org). Mirrors fiesta.def (which targets apptainer + OSDF).
# No NMMA: just fiesta + jax + blackjax (the blackjax-smc sampler) + a baked
# surrogate. The plugin ships fiesta_wrapper.py + fiesta_bridge.py per-job, so
# this image is a generic fiesta runtime (no plugin code baked in).
#
#   docker build -t hub.opensciencegrid.org/<project>/fiesta:v1 -f containers/Dockerfile .
#   docker push   hub.opensciencegrid.org/<project>/fiesta:v1
#
# GPU variant: add a `pip3 install "jax[cuda12]"` layer and run with request_gpus=1.
FROM ubuntu:24.04

ENV PIP_BREAK_SYSTEM_PACKAGES=1 DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y python3 python3-pip git build-essential \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install uv \
 && uv pip install --system --break-system-packages fiestaem blackjax

# Bake the KN surrogate (a CVMFS/read-only image can't download on a worker).
RUN python3 -c "from fiesta.surrogates.manage_surrogates import download_surrogate; download_surrogate('Bu2025_MLP')"

# Fail the build if the runtime isn't importable.
RUN python3 -c "import fiesta, blackjax, jax; print('fiesta+blackjax+jax', jax.__version__)"

CMD ["python3"]
