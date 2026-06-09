# Notes for AI coding agents

## Comments

- Be concise. A comment should explain _why_ in a line or two, not narrate _what_ the code does.
- No multi-paragraph essays in code or CI config. For rationale, a short sentence plus an issue/PR reference is enough.
- Match the brevity and style of the surrounding code.
- Don't embed issue/PR numbers in inline code comments; they belong in commit messages / PR descriptions.

## Changes

- Keep diffs minimal and focused on the task; don't refactor unrelated code.
- Prefer existing tools, helpers, and patterns over introducing new ones.

## This repo specifically

- The OSG submission path lives in `main.py:submit_job`; the SkyPortal webhook
  contract lives in `AnalysisHandler`. Keep those two boundaries clean — the
  rest of the file is glue.
- `import htcondor` is intentionally lazy (inside `get_schedd` / `submit_job`)
  so the module is importable without HTCondor for tests.
