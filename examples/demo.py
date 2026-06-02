"""corroborate demo -- 3 hallucinated answers, all flagged.

Run:
    python demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import corroborate


CASES = [
    {
        "name": "faithful answer (control)",
        "query": "When did World War II end?",
        "chunks": ["World War II ended in 1945 with the surrender of Japan in September."],
        "answer": "World War II ended in 1945, when Japan surrendered.",
        "expect_ok": True,
    },
    {
        "name": "FG002: hallucinated date",
        "query": "When did WW2 end?",
        "chunks": ["World War II ended in 1945 with the surrender of Japan."],
        "answer": "World War II ended in 1944 with Germany's surrender.",
        "expect_ok": False,
    },
    {
        "name": "FG001: hallucinated quantity",
        "query": "How tall is Mount Everest?",
        "chunks": ["Mount Everest is the tallest mountain at 8,849 meters above sea level."],
        "answer": "Mount Everest is 9,200 meters tall and located in Nepal.",
        "expect_ok": False,
    },
    {
        "name": "FG003: hallucinated quote",
        "query": "What did Einstein say?",
        "chunks": ["Einstein remarked that imagination is more important than knowledge."],
        "answer": 'Einstein said "the universe is mostly empty space" in 1922.',
        "expect_ok": False,
    },
    {
        "name": "FG001: percentage hallucination",
        "query": "What was the inflation rate?",
        "chunks": ["Inflation in the eurozone reached 5.5% in mid-2022."],
        "answer": "Inflation in the eurozone peaked at 9.2% in mid-2022.",
        "expect_ok": False,
    },
]


def main() -> int:
    failures = 0
    for case in CASES:
        print(f"--- {case['name']} ---")
        report = corroborate.check_answer(
            query=case["query"],
            retrieved_chunks=case["chunks"],
            answer=case["answer"],
        )
        print(report)
        actual_ok = report.ok()
        print(f"report.ok() = {actual_ok} (expected {case['expect_ok']})")
        if actual_ok != case["expect_ok"]:
            failures += 1
            print("MISMATCH")
        print()

    if failures:
        print(f"FAIL: {failures} mismatches")
        return 1
    print("All cases matched expectations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
