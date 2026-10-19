"""Self-contained HTML report and a plain-text summary."""

from __future__ import annotations

from html import escape
from typing import Any


def text_summary(results: dict[str, Any]) -> str:
    s = results["summary"]
    lines = [
        f"{results['name']}  target={results['target']}  cases={s['totals']['cases']}  "
        f"error_cases={s['totals']['error_cases']}",
        f"{'metric':<14}{'mean':>8}  {'95% CI':<18}{'n':>4}",
    ]
    for name, m in s["metrics"].items():
        lines.append(f"{name:<14}{m['mean']:>8.3f}  [{m['ci_low']:.3f}, {m['ci_high']:.3f}]  {m['n']:>4}")
    t = s["totals"]
    lines.append(
        f"latency mean {t['latency_mean_s']}s, p95 {t['latency_p95_s']}s; "
        f"target tokens in/out {t['target_tokens_in']}/{t['target_tokens_out']}"
    )
    ju = results.get("judge_usage") or {}
    if results.get("judge"):
        lines.append(
            f"judge {results['judge']}: {ju.get('calls', 0)} calls, {ju.get('cache_hits', 0)} cache hits, "
            f"tokens in/out {ju.get('tokens_in', 0)}/{ju.get('tokens_out', 0)}"
        )
    return "\n".join(lines)


def _cell(value: float | None) -> str:
    if value is None:
        return '<td class="na">-</td>'
    cls = "bad" if value < 0.5 else "mid" if value < 1.0 else "ok"
    return f'<td class="{cls}">{value:.2f}</td>'


def html_report(results: dict[str, Any]) -> str:
    metrics = list(results["summary"]["metrics"])
    head = "".join(f"<th>{escape(m)}</th>" for m in metrics)
    summary_rows = "".join(
        f"<tr><td>{escape(n)}</td><td>{m['mean']:.3f}</td><td>[{m['ci_low']:.3f}, {m['ci_high']:.3f}]</td>"
        f"<td>{m['n']}</td></tr>"
        for n, m in results["summary"]["metrics"].items()
    )
    case_rows = []
    for c in results["cases"]:
        cells = "".join(_cell(c["scores"].get(m)) for m in metrics)
        err = f'<div class="err">{escape("; ".join(c["errors"]))}</div>' if c["errors"] else ""
        case_rows.append(
            f"<tr><td>{escape(c['id'])}</td><td>{escape(c['input'])}<div class='ans'>{escape(c['answer'])}</div>"
            f"{err}</td>{cells}</tr>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{escape(results['name'])} - eval report</title>
<style>
body{{font-family:system-ui,sans-serif;margin:2rem;max-width:1100px}}
table{{border-collapse:collapse;width:100%;margin-bottom:2rem}}
th,td{{border:1px solid #ccc;padding:.35rem .6rem;text-align:left;vertical-align:top}}
.ok{{background:#e6f4ea}}.mid{{background:#fff4d6}}.bad{{background:#fde7e9}}.na{{color:#999}}
.ans{{color:#555;font-size:.9em;margin-top:.2rem}}.err{{color:#b00020;font-size:.9em}}
</style></head><body>
<h1>{escape(results['name'])}</h1>
<p>target: {escape(str(results['target']))} | created: {escape(results['created'])}
| cases: {results['summary']['totals']['cases']}
| error cases: {results['summary']['totals']['error_cases']}</p>
<h2>Summary (bootstrap 95% CI)</h2>
<table><tr><th>metric</th><th>mean</th><th>95% CI</th><th>n</th></tr>{summary_rows}</table>
<h2>Cases</h2>
<table><tr><th>id</th><th>question / answer</th>{head}</tr>{''.join(case_rows)}</table>
</body></html>
"""
