# Fiesta runtime container for OSG jobs

The plugin runs `fiesta_wrapper.py` inside a **lean fiesta-only image** on OSPool
workers. The image is a generic fiesta runtime (fiesta + jax + blackjax + a baked
`Bu2025_MLP` surrogate) — **no NMMA, no plugin code baked in**. Both
`fiesta_wrapper.py` and `fiesta_bridge.py` ship per-job via
`transfer_input_files`, so the image only needs rebuilding when fiesta or its
deps change, not when the adapter changes.

Two distribution paths, both stamped onto jobs as `+SingularityImage`:
- **OSDF** (`fiesta.def` → `.sif`) — fast to iterate; per-job pull (~489 MB).
- **CVMFS** (`Dockerfile` → Harbor → `/cvmfs/...`) — no per-job pull; sync latency
  on updates. Preferred for the stable release.

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

## CVMFS path (`Dockerfile` → Harbor) — no per-job image pull

CVMFS distributes *unpacked* images under
`/cvmfs/singularity.opensciencegrid.org/`, auto-synced from a registry. You push
a Docker/OCI image; OSG unpacks it; jobs reference a filesystem path.

1. **Get a Harbor project.** Log in to `hub.opensciencegrid.org` (OSG Harbor) and
   create/request a project, e.g. `skyportal-osg`. Projects must be **public** for
   the CVMFS sync, and enabled for auto-sync (the OSG `cvmfs-singularity-sync`
   picks up tagged images).

2. **Build + push** (from any x86_64 docker host; `buildx` for cross-arch):

   ```bash
   docker build -t hub.opensciencegrid.org/skyportal-osg/fiesta:v1 -f containers/Dockerfile .
   docker login hub.opensciencegrid.org
   docker push hub.opensciencegrid.org/skyportal-osg/fiesta:v1
   ```

3. **Wait for the sync** (periodic — hours, not instant). The image lands at:

   ```
   /cvmfs/singularity.opensciencegrid.org/skyportal-osg/fiesta:v1
   ```

   Check from the AP: `ls /cvmfs/singularity.opensciencegrid.org/skyportal-osg/`.

4. **Flip the config** to the CVMFS path (a path, not a URL):

   ```yaml
   singularity_image: /cvmfs/singularity.opensciencegrid.org/skyportal-osg/fiesta:v1
   ```

Notes:
- **Updates** = push a new tag (`fiesta:v2`) + wait for the next sync, then bump
  the config. Keep OSDF for fast iteration / unreleased builds; the two coexist —
  point `singularity_image` at whichever.
- CVMFS fetches only the files a job touches, lazily and cached, so image size is
  a non-issue once synced.
- The wrapper + bridge still ship per-job; CVMFS only removes the image transfer.
