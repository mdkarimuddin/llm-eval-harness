# Judge reliability study

**Status: not run yet.** No numbers below, on purpose. The tooling exists and is tested with scripted judges;
the study needs a real judge, an API key and 50 items labelled by hand.

## How to run it

1. Produce answers with contexts from the app you care about:
   `llmeval run my_tests.yaml --target my_app:ask --judge anthropic --out results.json`
2. `llmeval export-labels results.json -n 50 --metric faithfulness --out labels.jsonl`
3. Open `labels.jsonl` and set `human_label` to `1` (the answer is supported by the context) or `0` for each item.
   Label before looking at the judge's scores.
4. `llmeval reliability labels.jsonl --metric faithfulness --runs 3 --other-judge-model <second model> --out reliability.json`
5. Repeat for `--metric relevance`.

## What to record here when done

| Item | Value |
|---|---|
| Judge model(s) and date | |
| Items labelled / human pass rate | |
| Agreement and kappa vs human, per run | |
| Run-to-run: mean per-item std, same pass/fail in all runs | |
| Second judge: agreement, kappa, mean score shift | |
| Where the judge was wrong (3-5 concrete examples) | |
| Decision: is this judge trustworthy enough to gate on? | |

Write the "where the judge was wrong" section from the disagreements in `reliability.json`, not from memory.
