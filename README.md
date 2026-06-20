# osg-skyportal-plugin

A SkyPortal AnalysisService that runs light-curve fits ([fiesta](https://github.com/nuclear-multimessenger-astronomy/fiestaEM))
on the [Open Science Grid](https://osg-htc.org/) via HTCondor. SkyPortal POSTs a
request to `/analysis/<name>`; the plugin submits a Condor job to an OSG access
point, polls it, and POSTs the result back to SkyPortal's callback.

## Setup

1. **OSG SciToken** — point `htcondor.scitoken_path` at a valid token for your
   access point (e.g. `ap41.uw.osg-htc.org`).

2. **Configure** — copy `config.yaml.defaults` to `config.yaml` and set at least:
   - `htcondor.collector` / `htcondor.schedd` / `htcondor.project_name`
   - `defaults.singularity_image` — the fiesta image (CVMFS path or OSDF URL)
   - `osdf.output_prefix` — where wrapper jobs write result bundles
   - `auth.incoming_bearer_token` — shared secret SkyPortal sends

3. **Install + run**
   ```bash
   uv sync
   uv run python main.py --config config.yaml
   ```

4. **Register the AnalysisService** in SkyPortal so the UI can call it:
   ```bash
   uv run python register_analysis_service.py \
       --name fiesta_osg \
       --display "Fiesta (OSG)" \
       --listener-url http://<plugin-host>:7100/analysis/fiesta_osg \
       --group-ids 1
   ```

Now "Run analysis" on a SkyPortal source triggers a fit on OSG.

## How it works

- Submits a wrapper job (`fiesta_wrapper.py` + `fiesta_bridge.py`) to the OSG AP.
  Set `batch.enabled` to coalesce many requests into one submit (itemdata).
- An async poller tracks jobs via `condor_q`/`condor_history` and POSTs the
  result (model light curves + posteriors) back to `callback_url`.
- The schedd **is** the state store: jobs are stamped with `+SkyPortal*`
  ClassAds, and `rehydrate_jobs` rebuilds the in-memory table on restart.
- `GET /jobs` / `GET /jobs/<cluster_id>` — debug introspection.

Tests: `uv run pytest`. Deployment recipe (image, k8s): `skyportal-nrp/osg/`.
