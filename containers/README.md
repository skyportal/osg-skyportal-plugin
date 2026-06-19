# Fiesta runtime container for OSG jobs

The plugin runs `fiesta_wrapper.py` inside a **lean fiesta-only image** on OSPool
workers. The image is a generic fiesta runtime (fiesta + jax + blackjax + a baked
`Bu2025_MLP` surrogate) — **no NMMA, no plugin code baked in**. Both
`fiesta_wrapper.py` and `fiesta_bridge.py` ship per-job via
`transfer_input_files`, so the image only needs rebuilding when fiesta or its
deps change, not when the adapter changes.

Two distribution paths, both stamped onto jobs as `+SingularityImage`:
- **OSDF** (`fiesta.def` → `.sif`) — fast to iterate; per-job pull (~489 MB).
- **CVMFS** (image built in **fiestaEM** → Docker Hub → `/cvmfs/...`) — no per-job
  pull; matches ~40x more nodes. **Preferred** for the stable release.

---

## OSDF path (`fiesta.def`)

### Build the .sif on the access point

The AP is x86_64 with `apptainer --fakeroot` — build there (no cross-arch
emulation, next to the OSDF origin):

```bash
mkdir -p ~/nmma-build/{cache,tmp}        # any build dir; reusing the existing one
# copy containers/fiesta.def to ~/nmma-build/fiesta.def, then:
cd ~/nmma-build
APPTAINER_CACHEDIR=$PWD/cache APPTAINER_TMPDIR=$PWD/tmp \
  apptainer build --fakeroot --mksquashfs-args "-processors 1" fiesta.sif fiesta.def
```

`fiesta.def` installs `fiestaem` + `blackjax` from PyPI via `uv` and bakes the
surrogate; the build ends with an import check, so a broken runtime fails the
build, not a job.

### Stage to OSDF (workers can't see the AP's /home) and reference it

```bash
cp ~/nmma-build/fiesta.sif /ospool/ap41/data/$USER/fiesta-v1.sif   # versioned: OSDF caches by name
```

Then in `config.yaml` (`...osg.params.htcondor.defaults`):

```yaml
singularity_image: osdf:///ospool/ap41/data/<user>/fiesta-v1.sif
```

### Smoke-test on the AP before submitting

```bash
echo '{"analysis_parameters": {"dry_run": true}}' > /tmp/inputs.json
cd /tmp && apptainer exec ~/nmma-build/fiesta.sif python3 /path/to/fiesta_wrapper.py
# expect a one-line JSON bundle: {"status": "success", ...}
```

---

## CVMFS path (PREFERRED) — no per-job pull, ~40x more matchable nodes

The runtime image now lives in and is owned by **fiestaEM**
(`fiestaEM/containers/Dockerfile`), not this repo — it's a generic fiesta runtime
with no plugin code. CVMFS distributes *unpacked* images under
`/cvmfs/singularity.opensciencegrid.org/`, auto-synced from Docker Hub.

**Why preferred:** a `.sif` over OSDF makes HTCondor add `SINGULARITY_CAN_USE_SIF`
to the job requirements — only ~27 of ~1195 OSPool slots advertise that, vs ~1175
with apptainer. A CVMFS path drops that clause, so the job matches ~40x more nodes
(verified: a CVMFS job matched in minutes and ran exit 0; the `.sif` equivalent sat
idle for hours).

1. **Build + push to Docker Hub** (run in the fiestaEM checkout; repo must be public):

   ```bash
   docker buildx build --platform linux/amd64 -f containers/Dockerfile \
     -t docker.io/michaelwcoughlin/fiesta:latest --push .
   ```

2. **Register for CVMFS sync:** add `michaelwcoughlin/fiesta:latest` to
   `docker_images.txt` in `opensciencegrid/cvmfs-singularity-sync` (PR). Pushes to
   the tag then auto-resync.

3. **Wait for the sync** (periodic — hours for a first image). It lands at:

   ```
   /cvmfs/singularity.opensciencegrid.org/michaelwcoughlin/fiesta:latest
   ```

   Check from the AP: `ls /cvmfs/singularity.opensciencegrid.org/michaelwcoughlin/`.

4. **Flip the config** to the CVMFS path (a path, not a URL):

   ```yaml
   singularity_image: /cvmfs/singularity.opensciencegrid.org/michaelwcoughlin/fiesta:latest
   ```

Notes:
- The wrapper + bridge still ship per-job; CVMFS only removes the image transfer.
- CVMFS fetches only the files a job touches, lazily + cached, so image size is moot.
- Keep OSDF (`fiesta.def`) as a narrow fallback for fast iteration / unreleased builds.
