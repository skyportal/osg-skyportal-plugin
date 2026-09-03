# Submitting to an IGWN access point (for non-public strain)

The plugin submits to OSPool via ap41 by default. To run a job that needs
**non-public IGWN strain** (e.g. the PyGRB/KN targeted search reading `/frames`),
submit instead through an **IGWN-credentialed access point** that mints and
delivers an IGWN-scoped SciToken to the job. This was validated end-to-end on
`condor-f3.ligo.caltech.edu`; production should use the OSG-managed IGWN OSPool AP
(**AP42**), which submits to OSPool but issues IGWN credentials.

The key fact (confirmed with James Clark): **remote submission is invisible to
token generation.** The SciToken is minted by the AP's credmon *at submit time*
off the AP-access trust (a condor IDTOKEN), and refreshed over the job lifetime —
so a headless remote submitter (our Kubernetes pod) needs no interactive
OIDC/MFA. MFA only ever gates interactive SSH to the AP.

## What the AP delivers

With `use_oauth_services = scitokens`, the job receives a token at
`$_CONDOR_CREDS/scitokens.use` scoped (condor-f3's `LOCAL_CREDMON_AUTHZ_TEMPLATE`):

```
read:/frames read:/ligo read:/virgo read:/kagra read:/shared
read:/staging write:/staging/<user> gwdatafind.read dqsegdb.read
```

The job reads it (as `BEARER_TOKEN_FILE`) to authenticate to `datafind.igwn.org`
and pull `/frames` via OSDF. Verified: a remotely-submitted job located
GW170817's proprietary `H1_HOFT_C02` frames (`datafind` HTTP 200, 3 URLs).

## Plugin config (`services.external.osg.params`)

```yaml
htcondor:
  collector: condor.igwn.org            # the IGWN pool collector
  schedd: <ap>.ligo.caltech.edu         # e.g. condor-f3 / AP42 schedd name
  scitoken_path: ~/.condor/tokens.d/<ap>.idtoken   # AP-issued IDTOKEN (see below)
  spool: true
  keepalive: false
defaults:
  use_oauth_services: scitokens
  accounting_group: ligo.dev.o4.burst.allsky.stamp  # a valid LigoSearchTag
  accounting_group_user: <user>          # the AP account (e.g. michael.coughlin)
  pools: "IGWN,CIT"                       # replaces deprecated flock_local
  # transfer_executable is already False in the plugin (env resolves python)
```

These map onto the submit description via `main.py:_apply_igwn_ap` (all default
`""` = plain OSPool submission, so ap41 behaviour is unchanged).

## The IDTOKEN (remote-submission auth)

The plugin authenticates to the AP's schedd with a condor IDTOKEN (not a
SciToken). Mint one **on the AP** and place it where the pod reads
`scitoken_path`:

```bash
# on the AP (interactive, one-time; renew before expiry):
condor_token_fetch -lifetime 864000 -token <ap>.idtoken   # -> ~/.condor/tokens.d/
```

The IDTOKEN's trust domain must match the schedd you submit to (submit to the
AP's *own* schedd, e.g. `condor-f3.ligo.caltech.edu`, not a sibling `citlogin`
schedd behind the same login alias).

## Requirements the AP enforces

- **`LigoSearchTag`** must be a valid accounting tag (`/etc/condor/accounting/valid_tags`);
  set via `accounting_group` (+ `accounting_group_user`). Invalid/`None` →
  `ERROR: Invalid value for search tag`.
- **`flock_local` is deprecated** → use `pools: "IGWN,CIT"`.
- **`transfer_executable = False`** — otherwise HTCondor ships the submitter's
  `python3` (wrong libs on the exec node). Already handled by the plugin.

## Network

The AP schedd/collector must be reachable from the submitter. condor-f3
(`192.84.86.149:9618`) and `condor.igwn.org:9618` were reachable from ap41; the
Fritz pod (GKE) needs the same egress. condor-f3 is a dev AP behind the
`ssh.igwn.org` gateway for *interactive* login, but its schedd port is publicly
reachable for submission.

## In the job

The containerised job (e.g. the pycbc image) uses the delivered token for data
access, e.g.:

```python
os.environ["BEARER_TOKEN_FILE"] = os.path.join(os.environ["_CONDOR_CREDS"], "scitokens.use")
from gwdatafind import find_urls
urls = find_urls("H", "H1_HOFT_C02", start, end, host="datafind.igwn.org")
```

This is the data path the pygrb bridge's Phase-2 engine uses for non-public
strain (vs. the public GWOSC `Merger` fetch in Phase 1).
