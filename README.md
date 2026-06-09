# osg-skyportal-plugin

SkyPortal microservice that runs analyses on the
[Open Science Grid](https://osg-htc.org/) via HTCondor. SkyPortal users click
"Run analysis" on a source; this service submits a Condor job, polls until done,
and posts results back to SkyPortal's `ObjAnalysis` callback. Designed for NMMA
fits but the OSG plumbing is analysis-agnostic.

## Status

Spike. The current `main.py` ships:

- `POST /analysis/<name>` — the SkyPortal AnalysisService webhook contract
  (`inputs`, `callback_url`, `callback_method`, `resource_id`). Submits a
  default `sleep 60` Condor job; pass `inputs.analysis_parameters.executable`
  to run something else.
- `GET /jobs` / `GET /jobs/<cluster_id>` — debug introspection.
- Async poller updating job status via `condor_q` + `condor_history`; on a
  terminal state, POSTs base64'd stdout (plus stderr tail) back to
  `callback_url` in SkyPortal's expected schema.
- `register_analysis_service.py` — one-shot CLI that creates the
  `AnalysisService` row in SkyPortal pointing at this plugin's URL.

NMMA-specific job spec (priors / model selection / posterior+plot transfer)
comes next.

## Prior art

The webhook contract here is the one defined by
[skyportal/skyportal#3199](https://github.com/skyportal/skyportal/pull/3199)
(closed, blocked on a bilby upstream issue). That PR ran NMMA in-process on a
local microservice — this one moves the compute to OSG and keeps the same
SkyPortal-facing contract so the existing UI just works.

## Architecture

```
SkyPortal source page
        │ click "Run NMMA on OSG"
        ▼
SkyPortal AnalysisService dispatch ──► POST {inputs, callback_url, …}
                                         │
                                         ▼
                                  ┌──────────────────────────┐
                                  │ osg-skyportal-plugin     │
                                  │  /analysis/<name>        │
                                  │     │                    │
                                  │     ▼                    │
                                  │  htcondor.Submit         │
                                  │                          │
                                  │  poller_loop ◄─── condor_q / history
                                  │     │                    │
                                  │     ▼ on completion      │
                                  │  POST callback_url       │
                                  │  {status, message,       │
                                  │   analysis: {results,    │
                                  │     plots, log}}         │
                                  └──────────────────────────┘
                                         │
                                         ▼
                                  OSG schedd / AP
```

## Local prototype

```bash
uv sync
# Make sure $BEARER_TOKEN_FILE or htcondor.scitoken_path points to a valid OSG
# SciToken, then:
uv run python main.py
```

```bash
# Register the AnalysisService row in your SkyPortal so the UI knows about us.
uv run python register_analysis_service.py \
    --name NMMA_OSG \
    --display "NMMA on OSG" \
    --listener-url http://localhost:7100/analysis/nmma_osg \
    --group-ids 1

# Now click "Run analysis" in SkyPortal — or simulate the call directly:
curl -X POST http://localhost:7100/analysis/nmma_osg \
    -H 'Authorization: Bearer SAME_AS_CONFIG' \
    -d '{"inputs": {}, "callback_url": "http://localhost:5000/api/echo", "callback_method": "POST", "resource_id": "ZTF20abc"}'
# → {"status": "pending", "cluster_id": 12345}

curl http://localhost:7100/jobs/12345
curl http://localhost:7100/jobs
```

## Persistence: the schedd IS our database

The plugin stamps every submitted job with custom ClassAds carrying the
SkyPortal binding (`+SkyPortalAnalysisName`, `+SkyPortalCallback`,
`+SkyPortalCallbackMethod`, `+SkyPortalResourceId`). On startup `main.py`
calls `rehydrate_jobs(cfg)`, which queries the schedd's live queue + recent
history for jobs in our `ProjectName` and reconstructs the in-memory `JOBS`
dict. A crash or deploy that takes < 24h gets recovered automatically — no
local SQLite, no SkyPortal-side bookkeeping.

The one gap: jobs that *complete* AND age out of history while the plugin is
down (history retention is schedd-dependent but typically 24-72h on OSG
APs). That's an acceptable failure mode for a microservice that ought to
restart in seconds, but the callback retry would need to come from elsewhere
(SkyPortal-side reaper, or eventually `ObjAnalysis` write-through here).

## Data plane: OSDF / Pelican

`osdf.py` provides `upload(local_path, remote_url)` and `download(remote_url,
local_path)` over plain HTTPS — both use the SciToken at `$BEARER_TOKEN_FILE`
(or an explicit `token_path`) as a bearer header. Intended use:

- NMMA wrapper jobs running on OSG `pelican object put` / use `osdf.upload`
  to push posterior samples + corner plots to a known OSDF origin.
- The plugin's poller fetches them via `osdf.download` before posting the
  callback to SkyPortal.

The spike doesn't wire OSDF into `collect_outputs` yet — that lives in the
next PR alongside the NMMA wrapper.

## Operations

- `bin/check_schedd.py [token_path]` — connect to the configured schedd, run a
  one-row query, print the response. Smoke test for SciToken + collector config.
- `bin/check_token.py [token_path]` — decode the JWT payload of a SciToken,
  print `iss` / `sub` / `aud` / `scope` / `exp` plus time-to-expiry. Useful
  cron-able pre-flight; exits non-zero if the token is expired or expiring soon.

## Open design questions

- **Container.** `defaults.singularity_image` is plumbed but null until we
  publish an NMMA image to OSDF or a registry.
- **Cost gates.** No per-user / per-group concurrent-job caps yet. Easy to
  add once we have a few real users.
- **SciToken refresh.** Stays external (`htvault-config` / `oidc-agent`).
  `bin/check_token.py` makes the expiry visible; the plugin re-reads
  `BEARER_TOKEN_FILE` on every schedd connect so a refreshed file is picked up
  without a restart.
