# VALIDATION — adversarial-headroom

## Codex v1 (historical)
- Verdict: SERIOUS_PROBLEMS
- Summary: The infrastructure skeleton is tidy, but the actual experiment is entirely unimplemented, statistically incapable of supporting its intended null conclusion, and not yet specified tightly enough to estimate or guarantee an M4 pilot run.

## Codex v2
- Verdict: PASS_WITH_NOTES
- Summary: Analogous to introspection-verbalization Codex v2: X1–X13 OK; stages implemented with a real `make pilot` path; synthetic/proxy pilot default; several model revisions still on `main`.
- KEY_FIXES_OK: X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13

## Grok (dual-validate)
- Verdict: PASS_WITH_NOTES
- Summary: Attack ladder (direct→blackbox→GCG→PGD→steer→prefill), injection track, saturation regime selection, and falsification thresholds are implemented. Pilot is proxy/synthetic-capable on M4; not a frontier ASR campaign.

### Remaining
- Ladder ASR / GCG / PGD stages are implemented against synthetic or proxy success signals suitable for local smoke, not a full white-box optimization on frontier-hardened behaviors.
- Model revisions mostly `main`.

## Reconciliation
v1 unimplemented ladder replaced with domain stages and falsification thresholds. Grok PASS_WITH_NOTES; scientific power on hardened frontier ASR remains out of local pilot scope.
