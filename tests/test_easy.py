"""corroborate EASY test -- detect hallucinated numbers in 50 answers.

Setup
-----
50 (query, chunks, answer) examples. 25 are faithful (the answer's numbers
all appear in the chunks). 25 are hallucinated: a number was mutated to a
plausible-but-wrong value.

Without corroborate: a "naive" detector says everything is fine (this is
the status quo when teams have no grounding check at all). Precision = 1.0
on the faithful set, recall = 0.0 on the hallucinated set; F1 = 0.

With corroborate: FG001 fires on hallucinated numbers and stays silent on
faithful answers.

Metric
------
Precision, recall, F1, accuracy on the binary task "is this answer
hallucinated?", with vs without corroborate.

Hypothesis
----------
Without: F1 = 0 (no detector flags anything).
With:    F1 close to 1.0 with negligible false-positive rate.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import corroborate


# (chunks, faithful_answer, hallucinated_answer)  -- numeric facts
CASES = [
    (["Mount Everest stands 8,849 meters above sea level."],
     "Mount Everest is 8,849 meters tall.",
     "Mount Everest is 9,200 meters tall."),
    (["The Eiffel Tower is 330 meters tall, including its broadcast antennas."],
     "The Eiffel Tower is 330 meters tall.",
     "The Eiffel Tower is 312 meters tall."),
    (["Light travels at approximately 299,792,458 meters per second in vacuum."],
     "Light travels at 299,792,458 m/s in a vacuum.",
     "Light travels at 300,000,000 m/s in a vacuum."),
    (["The marathon distance was standardized at 42.195 kilometers in 1921."],
     "A marathon is 42.195 kilometers long.",
     "A marathon is 26 kilometers long."),
    (["The average human body contains about 60% water."],
     "The human body is about 60% water.",
     "The human body is about 75% water."),
    (["The Moon is on average 384,400 kilometers from Earth."],
     "The Moon is 384,400 km away from Earth on average.",
     "The Moon is 350,000 km away from Earth on average."),
    (["The Pacific Ocean covers approximately 165 million square kilometers."],
     "The Pacific Ocean covers 165 million sq km.",
     "The Pacific Ocean covers 200 million sq km."),
    (["Mount Kilimanjaro reaches 5,895 meters above sea level."],
     "Kilimanjaro is 5,895 meters tall.",
     "Kilimanjaro is 6,200 meters tall."),
    (["The Sahara Desert spans 9.2 million square kilometers."],
     "The Sahara is 9.2 million sq km.",
     "The Sahara is 12 million sq km."),
    (["The human heart beats roughly 100,000 times per day."],
     "The heart beats about 100,000 times daily.",
     "The heart beats about 250,000 times daily."),
    (["The boiling point of water at sea level is 100 degrees Celsius."],
     "Water boils at 100 C at sea level.",
     "Water boils at 110 C at sea level."),
    (["The world population reached 8 billion in November 2022."],
     "The world population reached 8 billion in November 2022.",
     "The world population reached 9 billion in November 2022."),
    (["Greenland's ice sheet is about 1.7 million square kilometers in area."],
     "Greenland's ice sheet covers 1.7 million sq km.",
     "Greenland's ice sheet covers 3 million sq km."),
    (["The Great Wall of China stretches for over 21,196 kilometers."],
     "The Great Wall is over 21,196 km long.",
     "The Great Wall is over 30,000 km long."),
    (["Antarctica holds about 70% of the world's fresh water."],
     "Antarctica holds about 70% of fresh water.",
     "Antarctica holds about 90% of fresh water."),
    (["The Nile River is approximately 6,650 kilometers long."],
     "The Nile is 6,650 km long.",
     "The Nile is 7,200 km long."),
    (["Saturn has at least 146 confirmed moons as of 2024."],
     "Saturn has at least 146 moons.",
     "Saturn has at least 200 moons."),
    (["The Burj Khalifa rises 828 meters into the Dubai sky."],
     "Burj Khalifa is 828 m tall.",
     "Burj Khalifa is 1,000 m tall."),
    (["The speed of sound in dry air at 20 C is roughly 343 m/s."],
     "The speed of sound is about 343 m/s in dry air.",
     "The speed of sound is about 500 m/s in dry air."),
    (["The Mariana Trench reaches a depth of 10,994 meters."],
     "The Mariana Trench is 10,994 m deep.",
     "The Mariana Trench is 12,500 m deep."),
    (["Mount Fuji has an elevation of 3,776 meters."],
     "Mount Fuji is 3,776 m tall.",
     "Mount Fuji is 4,000 m tall."),
    (["The Amazon River discharges roughly 209,000 cubic meters per second."],
     "The Amazon discharges 209,000 m3/s.",
     "The Amazon discharges 300,000 m3/s."),
    (["The asteroid belt's total mass is about 4% of the Moon's mass."],
     "The asteroid belt is about 4% of the Moon's mass.",
     "The asteroid belt is about 25% of the Moon's mass."),
    (["The Gobi Desert covers around 1.3 million square kilometers."],
     "The Gobi covers 1.3 million sq km.",
     "The Gobi covers 2.5 million sq km."),
    (["Earth's atmosphere is roughly 78% nitrogen and 21% oxygen."],
     "Earth's air is 78% nitrogen.",
     "Earth's air is 90% nitrogen."),
]


def detect_naive(answer: str) -> bool:
    """Status quo: no detector. Always returns 'not hallucinated'."""
    return False


def detect_corroborate(query: str, chunks, answer: str) -> bool:
    return not corroborate.check_answer(query, chunks, answer).ok()


def metrics(predictions: list[bool], labels: list[bool]) -> dict[str, float]:
    """Standard binary classification metrics. Positive class = 'hallucinated'."""
    tp = sum(1 for p, y in zip(predictions, labels) if p and y)
    tn = sum(1 for p, y in zip(predictions, labels) if not p and not y)
    fp = sum(1 for p, y in zip(predictions, labels) if p and not y)
    fn = sum(1 for p, y in zip(predictions, labels) if not p and y)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / max(1, len(predictions))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1, "accuracy": acc}


def run() -> dict:
    examples: list[tuple[str, list[str], str, bool]] = []
    for chunks, faithful_a, hall_a in CASES:
        examples.append(("question", chunks, faithful_a, False))  # not hallucinated
        examples.append(("question", chunks, hall_a, True))       # hallucinated
    labels = [y for *_, y in examples]
    naive_preds = [detect_naive(a) for *_, a, _ in examples]
    corr_preds = [detect_corroborate(q, c, a) for q, c, a, _ in examples]
    return {
        "n": len(examples),
        "n_hallucinated": sum(labels),
        "naive": metrics(naive_preds, labels),
        "corroborate": metrics(corr_preds, labels),
    }


def test_easy():
    r = run()
    print("\n=== corroborate EASY test (50 number-grounding cases) ===")
    print(f"Total: {r['n']}, hallucinated: {r['n_hallucinated']}")
    print(f"\n  WITHOUT corroborate (no detector):")
    print(f"    precision={r['naive']['precision']:.2f}  recall={r['naive']['recall']:.2f}  F1={r['naive']['f1']:.2f}  accuracy={r['naive']['accuracy']:.2f}")
    print(f"\n  WITH corroborate:")
    print(f"    precision={r['corroborate']['precision']:.2f}  recall={r['corroborate']['recall']:.2f}  F1={r['corroborate']['f1']:.2f}  accuracy={r['corroborate']['accuracy']:.2f}")
    print(f"    confusion: TP={r['corroborate']['tp']} TN={r['corroborate']['tn']} FP={r['corroborate']['fp']} FN={r['corroborate']['fn']}")
    assert r["corroborate"]["f1"] > r["naive"]["f1"], "corroborate must beat the no-detector baseline"
    assert r["corroborate"]["recall"] >= 0.8, "corroborate should catch most number hallucinations"
    assert r["corroborate"]["precision"] >= 0.85, "corroborate should not produce excessive false positives"


if __name__ == "__main__":
    test_easy()
    print("EASY TEST PASSED")
