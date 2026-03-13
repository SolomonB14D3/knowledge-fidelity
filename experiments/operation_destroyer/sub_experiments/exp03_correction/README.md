# Exp03: Adapter Correction of Systematic Biases

**Status:** PENDING

**Question:** Can a logit-space adapter trained on ~80 contrastive STEM pairs (targeting the 4 bias types) flip the systematic failures without regressing on passing facts?

**Secondary question:** Does fixing one bias type transfer to others? (positivity bias in calculus → positivity bias in physics?)

**Design:**
- Build 20 contrastive pairs per pattern type = 80 total training examples
- Train snap-on adapter with MC hinge loss (same as train_mc.py)
- Eval: per-pattern accuracy before/after, cross-pattern transfer matrix
- Control: train only on positivity-bias examples, measure change on linearity-bias facts

**Connection:** STEM analog of syco→bias cross-transfer (Paper 3).
