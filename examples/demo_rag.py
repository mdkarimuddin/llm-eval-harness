"""Toy RAG app used to demonstrate the harness. Deterministic, no model calls.

Retrieval is plain token overlap over a tiny fictional help centre. It is NOT meant to be a good RAG system;
it exists so the harness has something to measure. ``ask_v2`` is a deliberately worse variant (it does not
strip punctuation or stop words), used to show how ``llmeval compare`` flags a regression.
"""

from __future__ import annotations

import re

from llmeval.targets import Output

DOCS = {
    "returns": "Items can be returned within 30 days of delivery. Refunds are issued to the original payment method within 5 business days.",
    "shipping": "Standard shipping takes 3 to 5 business days. Express shipping takes 1 to 2 business days and costs 12 euros.",
    "warranty": "All bikes carry a 2 year warranty covering frame and fork defects. Tyres and brake pads are not covered.",
    "sizing": "Frame size depends on rider height. Riders between 170 and 180 cm usually choose size M.",
    "payment": "We accept credit cards, PayPal and bank transfer. Invoices are available for company orders.",
    "assembly": "Bikes arrive 90 percent assembled. Fitting the handlebar and pedals takes about 20 minutes.",
    "repairs": "Repairs are done at our Helsinki workshop. The usual turnaround is 3 working days.",
    "privacy": "We store order data for 7 years as required by accounting law. Marketing emails can be switched off at any time.",
}
STOP = {"the", "a", "an", "of", "to", "is", "are", "do", "does", "how", "what", "can", "i", "my", "we", "you",
        "for", "in", "on", "and", "or", "it", "at", "be", "with", "long", "much"}
MIN_OVERLAP = 1
REFUSAL = "I do not have information on that in the help centre."


def _tokens(text: str, clean: bool) -> set[str]:
    if clean:
        words = re.findall(r"[a-z0-9]+", text.lower())
        return {w for w in words if w not in STOP}
    return set(text.lower().split())  # v2: punctuation stays attached, stop words stay in


def _rank(question: str, clean: bool) -> list[tuple[str, int]]:
    q = _tokens(question, clean)
    scored = [(doc_id, len(q & _tokens(text, clean))) for doc_id, text in DOCS.items()]
    return sorted(scored, key=lambda x: (-x[1], x[0]))


def _answer(question: str, clean: bool, refuse: bool) -> Output:
    ranked = _rank(question, clean)
    top_id, top_score = ranked[0]
    if refuse and top_score < MIN_OVERLAP:
        return Output(answer=REFUSAL, sources=[d for d, _ in ranked[:3]], contexts=[DOCS[d] for d, _ in ranked[:3]])
    sources = [d for d, _ in ranked[:3]]
    return Output(answer=DOCS[top_id], sources=sources, contexts=[DOCS[d] for d in sources])


def ask(question: str) -> Output:
    return _answer(question, clean=True, refuse=True)


def ask_v2(question: str) -> Output:
    return _answer(question, clean=False, refuse=False)
