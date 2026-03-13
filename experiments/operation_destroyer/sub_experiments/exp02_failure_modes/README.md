# Exp02: Systematic Failure Taxonomy

**Question:** Why do all 6 models fail identically on 10 specific facts?

**Method:** Manual analysis + inline `get_completion_logprob` calls to identify the winning wrong answer for each 100%-miss fact.

**Result:** 4 failure patterns — all directional, all scale-invariant:
1. Positivity Bias: model prefers +sin(x) over -sin(x)
2. Linearity Bias: model prefers v over v^2, r over r^2
3. Missing-Constant Bias: model drops proportionality constants (k, v)
4. Simplicity/Truncation Bias: model prefers shorter symbolic forms (sp vs sp3)

**Key finding:** These are the "surprising truths" in STEM — facts where truth conflicts with training-distribution priors. Every model fails in the same direction.

**Next:** Exp03 tests whether a logit adapter can selectively correct these patterns.
