# Exp04: Margin as Oracle Confidence Score

**Status:** COMPLETE

**Question:** Does log-probability margin reliably predict whether the oracle answer is correct?
Can it serve as a calibrated confidence score for the STEM truth oracle?

**Method:** Score all 40 bias-pattern examples (baseline + mixed adapter) and the full 97-fact
STEM benchmark. Compute per-example margins, quintile accuracy, and adapter outcome breakdown.

**Results:**

**Perfect calibration on 40-fact bias benchmark:**
- Negative margin → correct: **0/19 (0%)**
- Positive margin → correct: **21/21 (100%)**
- Margin is a perfect binary classifier — zero false positives, zero false negatives

**Quintile breakdown:** Q1-Q2 (most negative): 0% accuracy. Q4-Q5 (most positive): 100%.
Q3 (-0.20..+1.24): 62% -- the uncertain zone around the decision boundary.

**Mixed adapter outcome (n=40):**
- Wrong -> Correct: 10 (delta margin +5.33 avg)
- Right -> Right:   20 (delta margin +1.60 avg)
- Right -> Wrong:    1 (delta margin -2.42)
- Wrong -> Wrong:    9 (delta margin -0.78 -- stubborn cases)

**Full STEM benchmark (Qwen3-4B-Base, n=97):**
- Statistics:     60%, mean margin -1.151 (hardest domain)
- Linear algebra: 58%, mean margin +0.492
- Physics:        85%, mean margin +2.722 (easiest domain)
- 23 negative-margin facts = oracle predicted failures

**Connection to Paper 4 (Confidence Cartography):** That paper showed rho=0.652 between model
confidence and human false-belief prevalence. This is the STEM analogue: margin sign perfectly
predicts correctness, and margin magnitude tracks domain difficulty.

**Key files:** run_exp04.py, results.json
