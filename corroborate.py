"""corroborate -- deterministic answer-grounding for LLM / RAG outputs.

One import, one call, one report. Given (query, retrieved_chunks, answer),
verify every number, date, and quoted string in the answer is supported by
the retrieved chunks. No LLM calls, sub-100ms per answer, CI-gateable.

Quickstart:

    import corroborate

    report = corroborate.check_answer(
        query="When did WW2 end?",
        retrieved_chunks=["World War II ended in 1945 with Japan's surrender."],
        answer="World War II ended in 1944 with Germany's surrender.",
    )
    print(report)
    # FG002 critical: 1944 is not present in any chunk.

    if not report.ok():
        # block this answer; it's hallucinating a date.
        ...

The check is deterministic, offline, and cheap. It's the floor before
LLM-as-judge eval (RAGAS / TruLens / DeepEval), not a replacement for end-
to-end QA. PDF output is an optional extra (`pip install corroborate[pdf]`).

Author: Asmit Dash
License: MIT
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence

__version__ = "0.1.0"
__all__ = [
    "check_answer",
    "Report",
    "Finding",
    "Severity",
    "__version__",
]


# ---------------------------------------------------------------------------
# Public types -- vendored from dash-mlguard with prefix swap
# ---------------------------------------------------------------------------


class Severity(str, Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


_SEVERITY_ORDER = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.INFO: 2}
_SEVERITY_GLYPH = {
    Severity.CRITICAL: "[X]",
    Severity.WARNING: "[!]",
    Severity.INFO: "[i]",
}

_FG_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "FG001": {
        "title": "Number / quantity not in retrieved chunks",
        "what": "A numeric value (count, percentage, quantity, year, money) appears in the answer but cannot be found in any retrieved chunk after normalization. The model is reporting a number it wasn't given.",
    },
    "FG002": {
        "title": "Date not in retrieved chunks",
        "what": "A date or year appears in the answer but cannot be matched to any retrieved chunk. Models hallucinate dates particularly often when the corpus contains adjacent dates from the same era.",
    },
    "FG003": {
        "title": "Quoted string not in retrieved chunks",
        "what": "A quoted span in the answer (text between '\"' marks) is not a substring of any retrieved chunk. A quote that isn't verbatim in the source isn't a quote -- it's a paraphrase the model is dressing up.",
    },
}


@dataclass(frozen=True)
class Finding:
    code: str
    severity: Severity
    message: str
    fix: str
    spans: tuple[str, ...] = field(default_factory=tuple)
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        glyph = _SEVERITY_GLYPH[self.severity]
        sps = f" spans={list(self.spans[:5])}" if self.spans else ""
        return (
            f"{glyph} {self.code} {self.severity.value.upper()}: "
            f"{self.message}{sps}\n    fix: {self.fix}"
        )


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)

    def add(self, finding: Finding | None) -> None:
        if finding is not None:
            self.findings.append(finding)

    def extend(self, findings: Iterable[Finding]) -> None:
        for f in findings:
            self.add(f)

    @property
    def critical(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == Severity.CRITICAL]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == Severity.WARNING]

    @property
    def infos(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == Severity.INFO]

    def ok(self) -> bool:
        return not self.critical

    def sorted(self) -> list[Finding]:
        return sorted(self.findings, key=lambda f: (_SEVERITY_ORDER[f.severity], f.code))

    def summary(self) -> str:
        c, w, i = len(self.critical), len(self.warnings), len(self.infos)
        return f"corroborate: {c} critical, {w} warning, {i} info"

    def __str__(self) -> str:
        if not self.findings:
            return "corroborate: answer is grounded."
        body = "\n".join(str(f) for f in self.sorted())
        return f"{self.summary()}\n{body}"

    def __bool__(self) -> bool:
        return bool(self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "critical": len(self.critical),
                "warning": len(self.warnings),
                "info": len(self.infos),
            },
            "findings": [
                {
                    "code": f.code,
                    "severity": f.severity.value,
                    "message": f.message,
                    "fix": f.fix,
                    "spans": list(f.spans),
                    "details": f.details,
                }
                for f in self.sorted()
            ],
        }


# ---------------------------------------------------------------------------
# Number extraction & normalization
# ---------------------------------------------------------------------------


# Match: percentages (50%, 50.5 %), money ($1,000), plain numbers (1,234.5),
# years (1945), and ordinals ($1.2 billion handled separately).
_NUMBER_RE = re.compile(
    r"""
    (?<![A-Za-z\w/.])           # not in middle of a word/path
    (
        \$?\d{1,3}(?:,\d{3})+(?:\.\d+)?     # 1,234 or 1,234.56 or $1,234
        | \$?\d+\.\d+                       # 12.5 or $12.5
        | \$?\d+                            # 42 or $42
    )
    (?:\s?%)?                   # optional percent
    """,
    re.VERBOSE,
)

# Common scale words that follow a number ("$1.2 billion", "5 million")
_SCALE_WORDS = {
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
    "trillion": 1_000_000_000_000,
    "k": 1_000,
    "m": 1_000_000,
    "bn": 1_000_000_000,
}

# Numbers we don't care about flagging (they appear constantly and never
# represent factual claims that can be hallucinated meaningfully).
_TRIVIAL_NUMBERS = {"0", "1", "2"}


def _norm_number_str(raw: str) -> str:
    """Normalize a number string for substring matching: strip $, commas, %, spaces."""
    s = raw.strip().rstrip("%").lstrip("$").replace(",", "").strip()
    return s


def _extract_numbers(text: str) -> list[tuple[str, str]]:
    """Return list of (raw_span, normalized_str) for every number in text.

    Filters out trivial numbers (0, 1, 2) which appear too often to flag.
    """
    out: list[tuple[str, str]] = []
    for m in _NUMBER_RE.finditer(text):
        raw = m.group(0).strip()
        norm = _norm_number_str(raw)
        if norm in _TRIVIAL_NUMBERS:
            continue
        # Strip trailing decimal zeros for stable matching: "1945.0" -> "1945"
        if "." in norm:
            try:
                f = float(norm)
                if f.is_integer():
                    norm = str(int(f))
            except ValueError:
                pass
        out.append((raw, norm))
    return out


def _number_in_text(norm: str, text: str) -> bool:
    """Check whether a normalized number is present in `text`.

    We re-extract numbers from `text` and compare normalized forms, so
    "$1,945" in the answer matches "1945" in the chunk.
    """
    chunk_nums = {n for _, n in _extract_numbers(text)}
    if norm in chunk_nums:
        return True
    # Also try float-equivalence (handles "1945" vs "1945.0" both ways).
    try:
        target = float(norm)
        for cn in chunk_nums:
            try:
                if abs(float(cn) - target) < 1e-9:
                    return True
            except ValueError:
                continue
    except ValueError:
        pass
    return False


# ---------------------------------------------------------------------------
# Date extraction & normalization
# ---------------------------------------------------------------------------


_MONTH_NAMES = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}


# Year: 4 digits 1500-2099 (avoids matching part numbers / quantities).
_YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
_MONTH_DAY_YEAR_RE = re.compile(
    r"\b(?P<month>january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(?P<year>\d{4}))?\b",
    re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_NUMERIC_DATE_RE = re.compile(r"\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})\b")


@dataclass(frozen=True)
class _ParsedDate:
    """Canonical representation: (year, month, day). Components may be None.

    Two _ParsedDate are equivalent if their non-None components agree.
    """
    year: int | None
    month: int | None
    day: int | None
    raw: str

    def matches(self, other: "_ParsedDate") -> bool:
        if self.year is not None and other.year is not None and self.year != other.year:
            return False
        if self.month is not None and other.month is not None and self.month != other.month:
            return False
        if self.day is not None and other.day is not None and self.day != other.day:
            return False
        # Require at least the year or month to match concretely.
        if self.year is None and other.year is None and self.month is None and other.month is None:
            return False
        return True


def _extract_dates(text: str) -> list[_ParsedDate]:
    out: list[_ParsedDate] = []
    seen_spans: set[tuple[int, int]] = set()

    # ISO dates first (most specific).
    for m in _ISO_DATE_RE.finditer(text):
        seen_spans.add(m.span())
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            out.append(_ParsedDate(y, mo, d, m.group(0)))

    # Numeric dates (mm/dd/yyyy). Ambiguous about US vs EU but for grounding
    # we accept both readings and let the chunk match either.
    for m in _NUMERIC_DATE_RE.finditer(text):
        if any(s[0] <= m.start() < s[1] for s in seen_spans):
            continue
        seen_spans.add(m.span())
        a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
        year = c if c >= 100 else (1900 + c if c >= 50 else 2000 + c)
        # Try both readings (mm/dd and dd/mm). Push both -- match() will
        # accept whichever interpretation the chunk uses.
        if 1 <= a <= 12 and 1 <= b <= 31:
            out.append(_ParsedDate(year, a, b, m.group(0)))
        if 1 <= b <= 12 and 1 <= a <= 31 and a != b:
            out.append(_ParsedDate(year, b, a, m.group(0)))

    # Month-day(-year) phrases.
    for m in _MONTH_DAY_YEAR_RE.finditer(text):
        if any(s[0] <= m.start() < s[1] for s in seen_spans):
            continue
        seen_spans.add(m.span())
        mo = _MONTH_NAMES[m.group("month").lower()]
        d = int(m.group("day"))
        y = int(m.group("year")) if m.group("year") else None
        if 1 <= d <= 31:
            out.append(_ParsedDate(y, mo, d, m.group(0)))

    # Standalone years.
    for m in _YEAR_RE.finditer(text):
        if any(s[0] <= m.start() < s[1] for s in seen_spans):
            continue
        seen_spans.add(m.span())
        out.append(_ParsedDate(int(m.group(0)), None, None, m.group(0)))

    return out


def _date_in_chunks(d: _ParsedDate, chunk_dates: list[list[_ParsedDate]]) -> bool:
    for cdates in chunk_dates:
        for cd in cdates:
            if d.matches(cd):
                return True
    return False


# ---------------------------------------------------------------------------
# Quote extraction (substrings inside double quotes)
# ---------------------------------------------------------------------------


# Match anything between a pair of straight-quote ", curly-quote “ ”,
# or chevron «  » characters. Skip super short and super long matches.
_QUOTE_RE = re.compile(r'"([^"\n]{4,400})"|“([^”\n]{4,400})”|«([^»\n]{4,400})»')


def _extract_quotes(text: str) -> list[str]:
    quotes: list[str] = []
    for m in _QUOTE_RE.finditer(text):
        s = next((g for g in m.groups() if g), "").strip()
        if s and len(s.split()) >= 2:  # require >=2 words to avoid noise
            quotes.append(s)
    return quotes


def _normalize_for_substr(s: str) -> str:
    # Collapse whitespace and lowercase. Curly-quotes were already stripped.
    return re.sub(r"\s+", " ", s.lower()).strip()


def _quote_in_chunks(quote: str, chunks_norm: list[str]) -> bool:
    q = _normalize_for_substr(quote)
    return any(q in c for c in chunks_norm)


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def check_answer(
    query: str,
    retrieved_chunks: Sequence[Any],
    answer: str,
    *,
    mode: str = "loose",
) -> Report:
    """Verify an LLM answer is grounded in the retrieved chunks.

    Parameters
    ----------
    query : str
        The user's question. Currently used only for context in the report;
        future versions may use it to scope checks (e.g. ignore numbers in
        portions of the answer that explicitly disclaim sourcing).
    retrieved_chunks : sequence of str or {"text": str, ...} dicts
        The chunks fed to the LLM as context.
    answer : str
        The model's response.
    mode : "loose" | "strict", default "loose"
        Currently only "loose" is implemented (FG001/002/003). "strict" is
        reserved for v0.2 lexical-overlap checks which can false-fire on
        paraphrased-but-correct answers.

    Returns
    -------
    Report
        report.ok() is True iff every number, date, and quote in `answer`
        is supported by at least one chunk.
    """
    if not isinstance(retrieved_chunks, Sequence) or isinstance(retrieved_chunks, (str, bytes)):
        raise TypeError("retrieved_chunks must be a sequence of strings or dicts")

    chunk_texts: list[str] = []
    for c in retrieved_chunks:
        if isinstance(c, str):
            chunk_texts.append(c)
        elif isinstance(c, dict):
            chunk_texts.append(str(c.get("text", "")))
        else:
            raise TypeError(f"chunks must be str or dict; got {type(c).__name__}")

    chunks_norm = [_normalize_for_substr(t) for t in chunk_texts]
    chunks_concat = "\n".join(chunk_texts)
    chunk_dates = [_extract_dates(t) for t in chunk_texts]

    report = Report()

    # FG001 -- numbers not in chunks (excluding numbers that are dates we'll
    # flag separately under FG002).
    answer_dates = _extract_dates(answer)
    answer_date_year_strs = {str(d.year) for d in answer_dates if d.year is not None}

    answer_numbers = _extract_numbers(answer)
    unsupported_numbers: list[tuple[str, str]] = []
    for raw, norm in answer_numbers:
        # Skip values that are years already covered by date extraction.
        if norm in answer_date_year_strs:
            continue
        if not _number_in_text(norm, chunks_concat):
            unsupported_numbers.append((raw, norm))

    if unsupported_numbers:
        report.add(
            Finding(
                code="FG001",
                severity=Severity.CRITICAL,
                message=(
                    f"{len(unsupported_numbers)} number(s) in answer are not present "
                    f"in any retrieved chunk: "
                    f"{[r for r, _ in unsupported_numbers[:5]]}"
                ),
                fix=(
                    "Either remove the unsupported number, cite a retrieved chunk that "
                    "contains it, or instruct the model to refuse to answer when the "
                    "context lacks the needed value."
                ),
                spans=tuple(r for r, _ in unsupported_numbers),
            )
        )

    # FG002 -- dates not in chunks.
    unsupported_dates: list[_ParsedDate] = []
    for d in answer_dates:
        if not _date_in_chunks(d, chunk_dates):
            unsupported_dates.append(d)
    if unsupported_dates:
        report.add(
            Finding(
                code="FG002",
                severity=Severity.CRITICAL,
                message=(
                    f"{len(unsupported_dates)} date(s) in answer are not present in any "
                    f"retrieved chunk: {[d.raw for d in unsupported_dates[:5]]}"
                ),
                fix=(
                    "Models hallucinate adjacent dates particularly often. Verify the "
                    "exact year/month/day appears in your context, or remove the date "
                    "from the answer."
                ),
                spans=tuple(d.raw for d in unsupported_dates),
            )
        )

    # FG003 -- quoted strings not in chunks.
    answer_quotes = _extract_quotes(answer)
    unsupported_quotes: list[str] = []
    for q in answer_quotes:
        if not _quote_in_chunks(q, chunks_norm):
            unsupported_quotes.append(q)
    if unsupported_quotes:
        report.add(
            Finding(
                code="FG003",
                severity=Severity.CRITICAL,
                message=(
                    f"{len(unsupported_quotes)} quoted span(s) in answer are not "
                    f"verbatim in any retrieved chunk: "
                    f"{[q[:60] for q in unsupported_quotes[:5]]}"
                ),
                fix=(
                    "If the model is paraphrasing, drop the quotation marks. If it's "
                    "supposed to be a direct quote, the chunk should contain the exact "
                    "string -- otherwise the citation is fabricated."
                ),
                spans=tuple(unsupported_quotes),
            )
        )

    return report
