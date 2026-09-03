# Global migration implementation checklist

- [x] New independent directory; legacy implementation remains untouched.
- [x] Explicit global primary and localized control configuration.
- [x] One centralized steering-spec constructor.
- [x] Per-position norms, 1.0 cap, BOS skipping, generated-token steering.
- [x] Global conditions are not duplicated across local target positions.
- [x] Direction source and intervention application scope are separate.
- [x] Token 97817 is the frozen primary identity at layer 40.
- [x] Raw and Unicode-word-filtered ranks are recorded.
- [x] Baseline is computed once and reused.
- [x] Frozen primary, localized control, and adaptive rescue are separate.
- [x] Raw/normalized output and correctness fields are explicit.
- [x] Requested/effective strength, scope, cap, and application count are logged.
- [x] Primary confidence intervals use samples, not adaptive attempts.
- [x] Analysis is intervention-mode aware.
- [x] Exported parity and frozen-pilot gates precede a full run.
- [x] Item-boundary resume is supported for full runs.
- [x] CPU regression tests cover global/local semantics and analysis isolation.

The 27B parity smoke, frozen pilot, and full run require an appropriate CUDA
host and are intentionally not represented as completed until their artifacts
exist.
