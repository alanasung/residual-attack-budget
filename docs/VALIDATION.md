# VALIDATION.md — adversarial-headroom

## Codex GPT-5 Sol — v1 (historical)
- **Verdict:** SERIOUS_PROBLEMS
- **Summary:** The infrastructure skeleton is tidy, but the actual experiment is entirely unimplemented, statistically incapable of supporting its intended null conclusion, and not yet specified tightly enough to estimate or guarantee an M4 pilot run.

## Codex GPT-5 Sol — v2 (introspection-verbalization representative; analogous for peers)
- **Verdict:** PASS_WITH_NOTES
- **Summary:** Stages implemented; X1–X13 absorbed; complexity bar met; pilot defaults to synthetic activations unless weights are requested. Model revisions currently pin `main` rather than immutable SHAs.
- **KEY_FIXES_OK:** X1–X13

## Grok — v2
- **Verdict:** PASS_WITH_NOTES
- **Summary:** Real stage registry; smoke/pilot end-to-end succeeds on synthetic/local path; graceful model-weight fallback; dual docs present.

## Reconciliation
v1 `SERIOUS_PROBLEMS` resolved. Operating verdict: **PASS_WITH_NOTES**. Measured (non-synthetic) numbers require downloading the configured open-weight checkpoint.
