import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from llmeval import cli
from llmeval.cache import Cache
from llmeval.report import html_report, text_summary
from llmeval.targets import CachedTarget, HttpTarget, Output, TargetError, coerce, load_callable


def test_coerce_and_load_callable():
    assert coerce("hi").answer == "hi"
    assert coerce({"answer": "x", "sources": ["a"]}).sources == ["a"]
    out = Output(answer="y")
    assert coerce(out) is out
    with pytest.raises(TargetError):
        coerce(5)
    with pytest.raises(TargetError):
        coerce({"answer": 5})
    assert callable(load_callable("examples.demo_rag:ask"))
    for bad in ("nocolon", "no.such.module:f", "examples.demo_rag:missing"):
        with pytest.raises(TargetError):
            load_callable(bad)


def test_cached_target_calls_once_per_input():
    calls = []

    def target(q):
        calls.append(q)
        return Output(answer=q.upper(), sources=["a"])

    ct = CachedTarget(target, Cache(), "t")
    assert ct("hi").answer == "HI" and ct("hi").sources == ["a"]
    assert calls == ["hi"] and ct.hits == 1


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        payload = {"answer": f"echo: {body['input']}", "sources": ["doc1"]}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def test_http_target_against_a_local_server():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        out = HttpTarget(f"http://127.0.0.1:{server.server_port}/ask")("hello")
        assert out.answer == "echo: hello" and out.sources == ["doc1"]
    finally:
        server.shutdown()


def test_http_target_errors_are_clean():
    with pytest.raises(TargetError):
        HttpTarget("ftp://x")
    with pytest.raises(TargetError, match="HTTP target failed"):
        HttpTarget("http://127.0.0.1:9/none", timeout=1)("q")


def fake_results(answer="<script>alert(1)</script>"):
    return {
        "name": "t", "created": "2026-10-19T00:00:00+00:00", "target": "x", "judge": None,
        "cases": [{"id": "c<1>", "input": "q & a", "answer": answer, "errors": ["bad <thing>"],
                   "scores": {"exact_match": 0.0, "mrr": None}}],
        "summary": {"metrics": {"exact_match": {"mean": 0.0, "n": 1, "ci_low": 0.0, "ci_high": 0.0}},
                    "totals": {"cases": 1, "error_cases": 1, "target_tokens_in": 0, "target_tokens_out": 0,
                               "latency_mean_s": 0.0, "latency_p95_s": 0.0}},
    }


def test_html_report_escapes_untrusted_text():
    page = html_report(fake_results())
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page and "c&lt;1&gt;" in page and "bad &lt;thing&gt;" in page


def test_text_summary_lists_metrics():
    assert "exact_match" in text_summary(fake_results())


def run_cli(*args):
    return cli.main([str(a) for a in args])


def test_cli_run_passes_gate_and_writes_outputs(tmp_path, capsys):
    out, page = tmp_path / "r.json", tmp_path / "r.html"
    code = run_cli("run", "examples/rag_tests.yaml", "--target", "examples.demo_rag:ask", "--out", out,
                   "--html", page, "--thresholds", "examples/thresholds_rag.yaml", "--cache-dir", tmp_path / "c")
    assert code == 0 and "thresholds passed" in capsys.readouterr().out
    assert json.loads(out.read_text())["summary"]["metrics"]["recall_at_k"]["mean"] == 1.0
    assert "<table" in page.read_text()


def test_cli_run_fails_gate_on_degraded_app(tmp_path, capsys):
    code = run_cli("run", "examples/rag_tests.yaml", "--target", "examples.demo_rag:ask_v2", "--out",
                   tmp_path / "r.json", "--thresholds", "examples/thresholds_rag.yaml", "--cache-dir", tmp_path / "c")
    assert code == 1 and "THRESHOLDS FAILED" in capsys.readouterr().out


def test_cli_agent_demo_passes(tmp_path):
    assert run_cli("run", "examples/agent_tests.yaml", "--target", "examples.demo_agent:run", "--out",
                   tmp_path / "a.json", "--thresholds", "examples/thresholds_agent.yaml",
                   "--cache-dir", tmp_path / "c") == 0


def test_cli_subset_runs_fewer_cases(tmp_path):
    out = tmp_path / "r.json"
    run_cli("run", "examples/rag_tests.yaml", "--target", "examples.demo_rag:ask", "--subset", "5", "--out", out,
            "--cache-dir", tmp_path / "c")
    assert json.loads(out.read_text())["summary"]["totals"]["cases"] == 5


def test_cli_compare_exit_code(tmp_path, capsys):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    common = ["--cache-dir", str(tmp_path / "c")]
    run_cli("run", "examples/rag_tests.yaml", "--target", "examples.demo_rag:ask", "--out", a, *common)
    run_cli("run", "examples/rag_tests.yaml", "--target", "examples.demo_rag:ask_v2", "--out", b, *common)
    assert run_cli("compare", a, b) == 0
    assert run_cli("compare", a, b, "--fail-on-regression") == 1
    assert run_cli("compare", a, a, "--fail-on-regression") == 0
    assert "regression" in capsys.readouterr().out


def test_cli_errors_return_2_with_message(tmp_path, capsys):
    assert run_cli("run", "missing.yaml", "--target", "examples.demo_rag:ask") == 2
    assert "not found" in capsys.readouterr().err
    assert run_cli("run", "examples/rag_tests.yaml", "--target", "bad.module:f", "--out", tmp_path / "x.json") == 2
    assert run_cli("run", "examples/rag_tests.yaml", "--target", "examples.demo_rag:ask", "--judge", "anthropic",
                   "--out", tmp_path / "x.json") in (2,)  # no API key in the test environment


def test_cli_export_labels(tmp_path):
    res, labels = tmp_path / "r.json", tmp_path / "l.jsonl"
    run_cli("run", "examples/rag_tests.yaml", "--target", "examples.demo_rag:ask", "--out", res,
            "--cache-dir", tmp_path / "c")
    assert run_cli("export-labels", res, "-n", "5", "--out", labels) == 0
    rows = [json.loads(line) for line in labels.read_text().splitlines()]
    assert len(rows) == 5 and all(r["human_label"] is None for r in rows)


def test_cli_scorers_override(tmp_path):
    out = tmp_path / "r.json"
    assert run_cli("run", "examples/rag_tests.yaml", "--target", "examples.demo_rag:ask", "--scorers", "mrr",
                   "--out", out, "--cache-dir", tmp_path / "c") == 0
    assert list(json.loads(out.read_text())["summary"]["metrics"]) == ["mrr"]
    assert run_cli("run", "examples/rag_tests.yaml", "--target", "examples.demo_rag:ask", "--scorers", "vibes",
                   "--out", out) == 2
