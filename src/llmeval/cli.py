"""Command line: llmeval run | compare | export-labels | reliability."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from llmeval import __version__
from llmeval.cache import Cache
from llmeval.compare import compare_runs, format_table
from llmeval.judge import AnthropicJudge, CachedJudge, JudgeError
from llmeval.reliability import (
    LabelError,
    evaluate_judge,
    export_for_labeling,
    format_report,
    load_labels,
)
from llmeval.report import html_report, text_summary
from llmeval.runner import RunError, run_cases, select_cases
from llmeval.schema import TestSetError, load_caseset
from llmeval.targets import CachedTarget, HttpTarget, TargetError, load_callable
from llmeval.thresholds import ThresholdError, check, load_thresholds


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llmeval", description="Evaluation and regression harness for LLM apps.")
    p.add_argument("--version", action="version", version=f"llmeval {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a test set against a target")
    run.add_argument("testset")
    tgt = run.add_mutually_exclusive_group(required=True)
    tgt.add_argument("--target", help="python callable, 'package.module:function'")
    tgt.add_argument("--url", help="HTTP endpoint that accepts POST {'input': ...}")
    run.add_argument("--out", default="eval_results.json")
    run.add_argument("--html", help="write an HTML report here")
    run.add_argument("--thresholds", help="YAML file with pass/fail rules; exit code 1 on violation")
    run.add_argument("--tags", nargs="*", help="only run cases with any of these tags")
    run.add_argument("--subset", type=int, help="run a seeded random subset of N cases (for PR checks)")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--judge", choices=["none", "anthropic"], default="none")
    run.add_argument("--judge-model", default=None)
    run.add_argument("--cache-dir", default=".evalcache")
    run.add_argument("--cache-target", action="store_true", help="also cache target outputs by input")

    cmp_ = sub.add_parser("compare", help="compare run B against baseline run A")
    cmp_.add_argument("run_a")
    cmp_.add_argument("run_b")
    cmp_.add_argument("--fail-on-regression", action="store_true")
    cmp_.add_argument("--seed", type=int, default=0)

    exp = sub.add_parser("export-labels", help="write items for you to label by hand")
    exp.add_argument("results")
    exp.add_argument("--metric", choices=["faithfulness", "relevance"], default="faithfulness")
    exp.add_argument("-n", type=int, default=50)
    exp.add_argument("--seed", type=int, default=0)
    exp.add_argument("--out", default="labels.jsonl")

    rel = sub.add_parser("reliability", help="compare the judge with your hand labels")
    rel.add_argument("labels")
    rel.add_argument("--metric", choices=["faithfulness", "relevance"], default="faithfulness")
    rel.add_argument("--runs", type=int, default=3)
    rel.add_argument("--judge-model", default=None)
    rel.add_argument("--other-judge-model", default=None)
    rel.add_argument("--out", help="write the report as JSON here")
    return p


def _cmd_run(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(Path.cwd()))
    caseset = load_caseset(args.testset)
    cases = select_cases(caseset, args.tags, args.subset, args.seed)
    cache = Cache(args.cache_dir)
    if args.url:
        target: Any = HttpTarget(args.url)
        name = args.url
    else:
        target = load_callable(args.target)
        name = args.target
    if args.cache_target:
        target = CachedTarget(target, cache, name)
    judge = None
    if args.judge == "anthropic":
        kwargs = {"model": args.judge_model} if args.judge_model else {}
        judge = CachedJudge(AnthropicJudge(**kwargs), cache)
    results = run_cases(caseset, cases, target, target_name=name, judge=judge, seed=args.seed)
    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.html:
        Path(args.html).write_text(html_report(results), encoding="utf-8")
    print(text_summary(results))
    if args.thresholds:
        violations = check(results["summary"], load_thresholds(args.thresholds))
        if violations:
            print("\nTHRESHOLDS FAILED")
            for v in violations:
                print(f"  - {v}")
            return 1
        print("\nthresholds passed")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    a = json.loads(Path(args.run_a).read_text(encoding="utf-8"))
    b = json.loads(Path(args.run_b).read_text(encoding="utf-8"))
    rows = compare_runs(a, b, seed=args.seed)
    print(f"A (baseline): {args.run_a}\nB: {args.run_b}\n")
    print(format_table(rows))
    if args.fail_on_regression and any(r.verdict == "regression" for r in rows):
        print("\nregression detected")
        return 1
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    items = export_for_labeling(results, args.metric, args.n, args.seed)
    with Path(args.out).open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"wrote {len(items)} items to {args.out}; set human_label to 1 (pass) or 0 (fail) for each")
    return 0


def _cmd_reliability(args: argparse.Namespace) -> int:
    items = load_labels(args.labels)
    kwargs = {"model": args.judge_model} if args.judge_model else {}
    judge = AnthropicJudge(**kwargs)  # deliberately not cached: repeated runs must call the model
    other = AnthropicJudge(model=args.other_judge_model) if args.other_judge_model else None
    report = evaluate_judge(items, judge, args.metric, runs=args.runs, other_judge=other)
    print(format_report(report))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    handlers = {"run": _cmd_run, "compare": _cmd_compare, "export-labels": _cmd_export,
                "reliability": _cmd_reliability}
    try:
        return handlers[args.command](args)
    except (TestSetError, TargetError, RunError, JudgeError, ThresholdError, LabelError, ValueError,
            FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
