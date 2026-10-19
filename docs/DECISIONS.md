# Design decisions

**Scorers return `None` instead of 0 when a case lacks the field.** A case with no `gold_sources` is not a retrieval
failure. Averages cover only the cases a scorer applies to, and the report shows `n` per metric.

**A crashing target scores 0, not "skipped".** Skipping failed cases would raise the averages of a broken app.
Errors are recorded per case and counted in `error_cases`, which can itself be gated.

**A threshold on a missing metric fails.** If a rule names `recall_at_k` and the run produced none, that is a
misconfigured gate, not a pass. A PR subset with no safety cases failing loudly is the intended behaviour; the demo
uses a `smoke` tag so every subset contains the cases each gate needs.

**Bootstrap intervals, paired for comparisons.** Percentile bootstrap on the mean, 2000 resamples, fixed seed so
reruns agree. Comparisons pair cases by id and bootstrap the differences, which is tighter than comparing two
independent intervals. No numpy dependency: the sets are small and the loop is fast enough.

**Judge scores are 1-5, scaled to 0-1; "pass" means 4 or more.** A coarse scale is easier to hand-label than a
continuous one. The 0.75 pass line is an assumption, not a result.

**Caching judge and (optionally) target calls on disk, keyed by judge name and full prompt.** Reruns cost nothing
and are reproducible. The reliability command deliberately bypasses the cache, otherwise "3 runs" would be one run.

**Judge prompts treat judged text as data.** Retrieved passages and answers can contain instructions; the system
prompt says to ignore them. This lowers the risk but does not remove it.

**Exit codes:** 0 = ok, 1 = threshold or regression failure, 2 = bad input or configuration. CI can tell a broken
app from a broken setup.

**No plugin system, no database, no dashboard.** Out of scope for a 3-day build.
