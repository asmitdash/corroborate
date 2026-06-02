"""corroborate HARD test -- 80 adversarial cases, mixed signal types.

Setup
-----
80 (chunks, answer, label) examples covering all three v0.1 codes plus
adversarial paraphrases that *should* pass:

  20 faithful number answers with formatting variation:
      "$1,234,567" in chunk vs "1234567" in answer (must NOT flag)
  20 hallucinated number answers (FG001 should fire)
  10 faithful date answers in different formats (must NOT flag):
      chunk: "September 2, 1945", answer: "1945-09-02"
  10 hallucinated date answers (FG002 should fire)
  10 faithful direct quotes (must NOT flag)
  10 fabricated direct quotes (FG003 should fire)

Plus latency: average milliseconds per check_answer call.

Hypothesis
----------
Without corroborate: F1=0 across all 80.
With corroborate: F1 >= 0.85, false-positive rate <= 0.15 (the strict bar
on faithful paraphrases is what makes this 'hard'), latency < 100ms/call.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import corroborate


# ---------- 20 faithful + 20 hallucinated number cases ----------------------
NUMBER_CASES_FAITHFUL = [
    (["The Eiffel Tower stands $1,234,567,000 in maintenance budget annually."],
     "Maintenance for the Eiffel Tower runs 1234567000 dollars per year."),
    (["The 2023 fiscal year revenue was $42,500,000."],
     "FY 2023 revenue: 42500000."),
    (["Approximately 7.9 billion people lived on Earth in 2021."],
     "About 7.9 billion humans were alive in 2021."),
    (["The drug reduced symptoms by 45.5% in clinical trials."],
     "The drug reduced symptoms by 45.5 percent in trials."),
    (["The marathon distance is 26.2 miles, equivalent to 42.195 kilometers."],
     "A marathon is 26.2 miles or 42.195 km long."),
    (["The Pacific Ocean has an average depth of 4,280 meters."],
     "The Pacific averages 4280 m deep."),
    (["The company employs 12,000 staff across 3 continents."],
     "The company has 12000 employees on 3 continents."),
    (["Light travels at 299,792,458 m/s in vacuum."],
     "Light's vacuum speed is 299792458 m/s."),
    (["The plane cruises at 35,000 feet."],
     "Cruising altitude: 35000 feet."),
    (["The bottle holds 750 mL of wine."],
     "A wine bottle holds 750 ml."),
    (["Annual rainfall averages 1,200 mm in this region."],
     "Average rainfall is 1200 mm per year."),
    (["The CPU runs at 3.6 GHz under load."],
     "The processor reaches 3.6 GHz."),
    (["The earthquake registered 7.8 on the Richter scale."],
     "Magnitude was 7.8."),
    (["The race was won in a time of 9.58 seconds."],
     "Winning time was 9.58 s."),
    (["Approximately 60% of adults are overweight in this study."],
     "About 60 percent of adults were overweight."),
    (["The exam was scored out of 150 points."],
     "Test was out of 150 points."),
    (["The car costs $35,499 with all options."],
     "Price: 35499 dollars fully loaded."),
    (["The vaccine showed 94.5% efficacy in trials."],
     "Efficacy was 94.5%."),
    (["The bridge spans 4.3 km across the strait."],
     "Bridge length: 4.3 km."),
    (["The dataset contains 1,500,000 labeled examples."],
     "Dataset has 1500000 labels."),
]

NUMBER_CASES_HALLUCINATED = [
    (["The Eiffel Tower stands 330 meters tall."],
     "The Eiffel Tower is 425 meters tall."),
    (["The 2023 fiscal year revenue was $42,500,000."],
     "FY 2023 revenue was $50,000,000."),
    (["Approximately 7.9 billion people lived on Earth in 2021."],
     "About 9.5 billion people were alive in 2021."),
    (["The drug reduced symptoms by 45.5% in clinical trials."],
     "The drug reduced symptoms by 80% in trials."),
    (["A marathon is 42.195 kilometers."],
     "A marathon is 50 kilometers."),
    (["The Pacific Ocean has an average depth of 4,280 meters."],
     "The Pacific averages 6000 m deep."),
    (["The company employs 12,000 staff."],
     "The company has 50000 staff."),
    (["Light travels at 299,792,458 m/s."],
     "Light moves at 350,000,000 m/s."),
    (["The plane cruises at 35,000 feet."],
     "The plane cruises at 60,000 feet."),
    (["The bottle holds 750 mL."],
     "The bottle holds 1500 mL."),
    (["Annual rainfall averages 1,200 mm."],
     "Annual rainfall is 3000 mm."),
    (["The CPU runs at 3.6 GHz."],
     "The CPU runs at 5.2 GHz."),
    (["The earthquake was 7.8 on Richter."],
     "The earthquake was 9.5 on Richter."),
    (["The race was won in 9.58 seconds."],
     "The winner ran 7.2 seconds."),
    (["About 60% of adults were overweight."],
     "About 85% of adults were overweight."),
    (["The exam was out of 150 points."],
     "The exam was out of 500 points."),
    (["The car costs $35,499."],
     "The car costs $90,000."),
    (["The vaccine showed 94.5% efficacy."],
     "The vaccine showed 99.9% efficacy."),
    (["The bridge spans 4.3 km."],
     "The bridge spans 12 km."),
    (["The dataset contains 1,500,000 examples."],
     "The dataset contains 5000000 examples."),
]


# ---------- 10 faithful + 10 hallucinated date cases ------------------------
DATE_CASES_FAITHFUL = [
    (["Japan formally surrendered on September 2, 1945."],
     "Japan surrendered on 1945-09-02."),
    (["The Berlin Wall fell on November 9, 1989."],
     "The Berlin Wall fell on 9 November 1989."),
    (["The first Apollo moon landing took place on July 20, 1969."],
     "Apollo 11 landed on the moon in July 1969."),
    (["The Treaty of Versailles was signed on June 28, 1919."],
     "The Versailles Treaty: June 1919."),
    (["The French Revolution began in 1789."],
     "The French Revolution began in 1789."),
    (["The Battle of Waterloo took place on June 18, 1815."],
     "Waterloo was fought June 18, 1815."),
    (["Pearl Harbor was attacked on December 7, 1941."],
     "Pearl Harbor: December 7 1941."),
    (["The Magna Carta was sealed in 1215."],
     "The Magna Carta dates from 1215."),
    (["The first iPhone was released on June 29, 2007."],
     "The iPhone launched in 2007."),
    (["The Soviet Union dissolved on December 25, 1991."],
     "The USSR dissolved in December 1991."),
]

DATE_CASES_HALLUCINATED = [
    (["Japan formally surrendered on September 2, 1945."],
     "Japan surrendered in 1944."),
    (["The Berlin Wall fell on November 9, 1989."],
     "The Berlin Wall fell in 1991."),
    (["The first Apollo moon landing took place on July 20, 1969."],
     "Apollo 11 landed on the moon in 1971."),
    (["The Treaty of Versailles was signed on June 28, 1919."],
     "The Treaty of Versailles was signed in 1920."),
    (["The French Revolution began in 1789."],
     "The French Revolution began in 1792."),
    (["The Battle of Waterloo took place on June 18, 1815."],
     "Waterloo was fought in 1812."),
    (["Pearl Harbor was attacked on December 7, 1941."],
     "Pearl Harbor was attacked in 1942."),
    (["The Magna Carta was sealed in 1215."],
     "The Magna Carta was sealed in 1066."),
    (["The first iPhone was released on June 29, 2007."],
     "The first iPhone was released in 2010."),
    (["The Soviet Union dissolved on December 25, 1991."],
     "The Soviet Union dissolved in 1989."),
]


# ---------- 10 faithful + 10 hallucinated quote cases -----------------------
QUOTE_CASES_FAITHFUL = [
    (["Steve Jobs said \"stay hungry, stay foolish\" in his 2005 Stanford speech."],
     "Jobs told graduates \"stay hungry, stay foolish\" in 2005."),
    (["Einstein once remarked \"imagination is more important than knowledge\"."],
     "Einstein said \"imagination is more important than knowledge\"."),
    (["The signers of the declaration pledged \"our lives, our fortunes, and our sacred honor\"."],
     "They pledged \"our lives, our fortunes, and our sacred honor\"."),
    (["Hamlet asks \"to be or not to be, that is the question\"."],
     "Hamlet asks \"to be or not to be, that is the question\"."),
    (["JFK said \"ask not what your country can do for you\" in his inaugural."],
     "Kennedy declared \"ask not what your country can do for you\"."),
    (["Churchill promised \"blood, toil, tears and sweat\"."],
     "Churchill said \"blood, toil, tears and sweat\"."),
    (["MLK said \"I have a dream\" during the 1963 march."],
     "King's \"I have a dream\" line came in 1963."),
    (["Descartes wrote \"I think, therefore I am\"."],
     "Descartes wrote \"I think, therefore I am\"."),
    (["Neil Armstrong said \"that's one small step for man\"."],
     "Armstrong said \"that's one small step for man\"."),
    (["The advert claimed \"just do it\" was a winning slogan."],
     "The ad's tagline was \"just do it\"."),
]

QUOTE_CASES_HALLUCINATED = [
    (["Steve Jobs said \"stay hungry, stay foolish\" in his Stanford speech."],
     "Jobs said \"think different and change everything you see\"."),
    (["Einstein remarked imagination matters more than knowledge."],
     "Einstein said \"the universe is mostly empty space\"."),
    (["The declaration pledged lives, fortunes, and sacred honor."],
     "The signers said \"freedom is never granted, only taken\"."),
    (["Hamlet famously asks the question of being."],
     "Hamlet says \"a coward dies a thousand deaths\"."),
    (["JFK gave a famous inaugural address."],
     "JFK said \"the only thing we have to fear is corruption itself\"."),
    (["Churchill rallied Britain through the darkest days."],
     "Churchill said \"we shall surrender on the beaches\"."),
    (["MLK led the civil rights movement."],
     "King said \"violence is the only language they understand\"."),
    (["Descartes was a French philosopher."],
     "Descartes wrote \"all knowledge derives from doubt alone\"."),
    (["Armstrong walked on the moon in 1969."],
     "Armstrong said \"the moon is more beautiful than I imagined\"."),
    (["The brand uses a memorable slogan."],
     "The slogan was \"because you are worth it\"."),
]


def detect_naive(answer: str) -> bool:
    return False


def detect_corroborate(query: str, chunks, answer: str) -> bool:
    return not corroborate.check_answer(query, chunks, answer).ok()


def metrics(predictions: list[bool], labels: list[bool]) -> dict[str, float]:
    tp = sum(1 for p, y in zip(predictions, labels) if p and y)
    tn = sum(1 for p, y in zip(predictions, labels) if not p and not y)
    fp = sum(1 for p, y in zip(predictions, labels) if p and not y)
    fn = sum(1 for p, y in zip(predictions, labels) if not p and y)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / max(1, len(predictions))
    fpr = fp / (fp + tn) if fp + tn else 0.0
    fnr = fn / (fn + tp) if fn + tp else 0.0
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1,
            "accuracy": acc, "fpr": fpr, "fnr": fnr}


def run() -> dict:
    examples: list[tuple[str, list[str], str, bool, str]] = []
    for chunks, ans in NUMBER_CASES_FAITHFUL:
        examples.append(("number-faithful", chunks, ans, False, "FG001"))
    for chunks, ans in NUMBER_CASES_HALLUCINATED:
        examples.append(("number-hallucinated", chunks, ans, True, "FG001"))
    for chunks, ans in DATE_CASES_FAITHFUL:
        examples.append(("date-faithful", chunks, ans, False, "FG002"))
    for chunks, ans in DATE_CASES_HALLUCINATED:
        examples.append(("date-hallucinated", chunks, ans, True, "FG002"))
    for chunks, ans in QUOTE_CASES_FAITHFUL:
        examples.append(("quote-faithful", chunks, ans, False, "FG003"))
    for chunks, ans in QUOTE_CASES_HALLUCINATED:
        examples.append(("quote-hallucinated", chunks, ans, True, "FG003"))

    labels = [y for *_, y, _ in examples]

    naive_preds = [detect_naive(a) for *_, a, _, _ in examples]

    t0 = time.perf_counter()
    corr_preds = [detect_corroborate(q, c, a) for q, c, a, _, _ in examples]
    elapsed = time.perf_counter() - t0
    avg_ms = 1000 * elapsed / len(examples)

    # Per-bucket breakdown
    buckets: dict[str, list[bool]] = {}
    bucket_labels: dict[str, list[bool]] = {}
    for (kind, *_, label, _), pred in zip(examples, corr_preds):
        buckets.setdefault(kind, []).append(pred)
        bucket_labels.setdefault(kind, []).append(label)

    bucket_metrics = {k: metrics(buckets[k], bucket_labels[k]) for k in buckets}

    return {
        "n": len(examples),
        "naive": metrics(naive_preds, labels),
        "corroborate": metrics(corr_preds, labels),
        "avg_ms_per_check": avg_ms,
        "bucket_metrics": bucket_metrics,
    }


def test_hard():
    r = run()
    print("\n=== corroborate HARD test (80 mixed cases incl. faithful paraphrases) ===")
    print(f"Total: {r['n']}, avg latency: {r['avg_ms_per_check']:.2f} ms/call")
    print(f"\n  WITHOUT corroborate (no detector): F1={r['naive']['f1']:.2f}  acc={r['naive']['accuracy']:.2f}")
    c = r["corroborate"]
    print(f"\n  WITH corroborate:")
    print(f"    precision={c['precision']:.3f}  recall={c['recall']:.3f}  F1={c['f1']:.3f}  acc={c['accuracy']:.3f}")
    print(f"    FPR={c['fpr']:.3f}  FNR={c['fnr']:.3f}")
    print(f"    confusion: TP={c['tp']} TN={c['tn']} FP={c['fp']} FN={c['fn']}")
    print(f"\n  Per-bucket recall (corroborate flagged):")
    for k, m in r["bucket_metrics"].items():
        flagged = m["tp"] + m["fp"]
        total = m["tp"] + m["fn"] + m["fp"] + m["tn"]
        print(f"    {k:24s}  flagged {flagged}/{total}  recall={m['recall']:.2f} fpr={m['fpr']:.2f}")
    assert r["corroborate"]["f1"] >= 0.85, f"F1 below threshold: {r['corroborate']['f1']:.2f}"
    assert r["corroborate"]["fpr"] <= 0.15, f"FPR above threshold: {r['corroborate']['fpr']:.2f}"
    assert r["avg_ms_per_check"] < 100, f"latency above 100ms: {r['avg_ms_per_check']:.2f}"


if __name__ == "__main__":
    test_hard()
    print("HARD TEST PASSED")
