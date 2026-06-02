# corroborate

**Deterministic answer-grounding for LLM and RAG outputs.** One import, one call. Verify every number, date, and quoted string in an LLM answer is supported by the retrieved chunks — with no LLM judge calls, sub-100ms per answer, CI-gateable.

```bash
pip install corroborate           # core (pure Python stdlib)
pip install corroborate[pdf]      # adds PDF report support (fpdf2)
```

```python
import corroborate

report = corroborate.check_answer(
    query="When did WW2 end?",
    retrieved_chunks=["World War II ended in 1945 with the surrender of Japan."],
    answer="World War II ended in 1944 with Germany's surrender.",
)
print(report)
# corroborate: 1 critical
# [X] FG002 CRITICAL: 1 date(s) in answer are not present in any retrieved chunk: ['1944']

if not report.ok():
    raise SystemExit("Block this answer; it's hallucinating a date.")
```

That's the whole API. Strings or `{"text": str}` dicts work as chunks. corroborate **does not** call any LLM — it's deterministic, runs in milliseconds, and depends only on the Python standard library.

---

## Why this exists

LLM-as-judge eval (RAGAS, TruLens, DeepEval) is the dominant way to measure RAG faithfulness. It's also expensive, slow, and inconsistent — multiple studies show LLM judges disagree 23–48% of the time on the same input. You can't run it in CI on every PR, and the cost compounds at scale.

But you don't need an LLM to catch the most common, most expensive hallucinations:

- The model says "1944" when the context says "1945." (a date that wasn't there)
- The model says "9,200 meters" when the context says "8,849." (a number that wasn't there)
- The model wraps a paraphrase in `"quotation marks"` and presents it as a direct quote.

These are **lexical, deterministic, regex-detectable**. corroborate runs in microseconds, costs nothing, and gives you a CI gate that fires on the worst-class-of-hallucination before any LLM judge ever sees the output.

It's the **floor** before LLM-as-judge eval, not a replacement for it.

---

## What it catches (v0.1)

| Code | Severity | What it catches |
|------|----------|-----------------|
| `FG001` | critical | Numbers / quantities / percentages / money in the answer that don't appear in any chunk after normalization (`$1,234` ≡ `1234` ≡ `1,234.00`) |
| `FG002` | critical | Dates / years in the answer that don't appear in any chunk (matches across formats: `1945`, `Sept 2 1945`, `1945-09-02`, `9/2/1945`) |
| `FG003` | critical | Quoted spans (text inside `"…"` / `“…”` / `«…»`) that aren't a verbatim substring of any chunk |

Each finding lists the offending **spans**, the **severity**, and **how to fix it** — and `report.ok()` returns False if any critical finding fires.

These three are intentionally the deterministic spine of corroborate. They produce near-zero false positives on real LLM output. Noisier signals (lexical-overlap-based "answer not grounded" warnings, named-entity grounding) are deliberately deferred to v0.2 behind a `mode="strict"` flag.

---

## Use it in CI

```python
import corroborate, sys

report = corroborate.check_answer(query, retrieved_chunks, answer)
sys.exit(0 if report.ok() else 1)
```

Fast enough that you can put it on every LLM call: a typical (10-chunk, 500-word answer) check runs in well under 100ms.

---

## API reference

```python
corroborate.check_answer(
    query,                               # str
    retrieved_chunks,                    # list[str] or list[{"text": str, ...}]
    answer,                              # str (the LLM output)
    *,
    mode="loose",                        # "loose" (v0.1) | "strict" (reserved for v0.2)
) -> Report
```

`Report`:

- `report.ok()` — `True` if no critical findings.
- `report.findings`, `report.critical`, `report.warnings`, `report.infos` — lists of `Finding`.
- `print(report)` — human-readable terminal summary.
- `report.to_dict()` — JSON-serializable dict.

Each `Finding` has: `code`, `severity`, `message`, `fix`, `spans` (tuple of strings extracted from the answer), `details`.

---

## Scope, on purpose

corroborate is **only** a deterministic floor. It doesn't:

- judge "is this answer good?" (use RAGAS / TruLens / DeepEval / LLM-as-judge),
- check entity-level grounding for paraphrased answers (NER-based grounding is in scope for v0.2),
- check the corpus itself ([chaffer](https://github.com/asmitdash/chaffer) does — sibling library),
- generate citations,
- rewrite or fix the answer.

If `corroborate.check_answer()` flags a finding, the answer has a class-of-hallucination that other tooling will probably miss. If it returns clean, you've ruled out the easy hallucinations — pay for LLM-as-judge to rule out the hard ones.

---

## See also

- **[chaffer](https://github.com/asmitdash/chaffer)** — sibling library: lints the RAG corpus before retrieval. corroborate lints the answer after generation.
- **[dash-mlguard](https://github.com/asmitdash/dash-mlguard)** — same author, same form factor, but for ML training pipelines.

---

## Development

```bash
git clone https://github.com/asmitdash/corroborate
cd corroborate
pip install -e ".[dev]"
pytest
```

---

## License

MIT — see [LICENSE](LICENSE).
