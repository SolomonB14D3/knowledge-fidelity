# NLI Contradiction Pairs Smoke Test

## Hypothesis
Reasoning injection fails with calculator and subitizing pairs because they lack
*inferential* contrast. NLI contradiction pairs teach the model to recognize when
one statement contradicts another through meaning, not just through a surface token
change. This is closer to actual reasoning: "does A follow from B?"

If the symmetry insight is right, the most effective NLI pairs will be those with
high token overlap between premise+entailment and premise+contradiction, where
the semantic contrast is concentrated in minimal surface differences.

## Dataset
Stanford NLI (SNLI): 570K human-written sentence pairs
- Labels: entailment, contradiction, neutral
- License: CC-BY-SA-4.0
- Source: https://huggingface.co/datasets/stanfordnlp/snli

## Pair Construction

### Step 1: Filter SNLI for contradiction pairs
From the training set, extract all pairs labeled "contradiction."
Approximately 190K pairs available.

### Step 2: Compute token overlap
For each contradiction pair, tokenize both premise and hypothesis.
Compute overlap ratio: (shared tokens) / (total unique tokens).
Rank pairs by overlap ratio.

### Step 3: Select high-overlap subset
Take the top 300 pairs by token overlap ratio.
These are pairs where the premise and hypothesis are nearly identical
on the surface but contradict each other semantically.

Example of what we want (high overlap):
  Premise: "A man is playing a guitar."
  Contradiction: "A man is not playing a guitar."
  (differs by one token: "not")

Example of what we don't want (low overlap):
  Premise: "Two women are hugging each other."
  Contradiction: "The women are boxing."
  (completely different surface form)

### Step 4: Format as contrastive training pairs
For each contradiction pair:
  Positive: premise text (the true description)
  Negative: contradiction text (the false description)

Tokenize, pack into training blocks same as bias/sycophancy pairs.

### Alternative: Entailment vs Contradiction pairs
Instead of premise as positive, use the ENTAILMENT hypothesis as positive
and CONTRADICTION hypothesis as negative for the same premise.

For a given premise, find:
  Entailment: "Some men are playing a sport." (follows from premise)
  Contradiction: "The men are sleeping." (contradicts premise)

The pair becomes:
  Positive: [premise] + [entailment hypothesis]
  Negative: [premise] + [contradiction hypothesis]

This format is closer to your existing bias/sycophancy pair structure
where the same prompt has two different completions.

**Use this format.** It matches your existing pipeline and the contrast
is between two responses to the same context, not between two standalone
statements.

## Experiment

### Run 1: 3.3M, NLI-contradiction at 5% injection
- ~10-15 min compute
- Measure all 8 behavioral dimensions
- Primary target: reasoning rho
- Secondary: does NLI injection affect any other dimension?

### Run 2 (if reasoning moves): 7M, NLI-contradiction at 5%
- Confirm scaling
- Check for cross-transfer to bias/sycophancy
- ~22 min compute

### Run 3 (if reasoning still 0): High-overlap only vs random NLI
- Take top 150 pairs by token overlap + 150 random contradiction pairs
- Run both at 3.3M
- If high-overlap outperforms random, symmetry matters for NLI too
- If both are zero, reasoning at 3.3M is confirmed capacity-limited

## What we're comparing

| Pair type        | Token overlap | Semantic depth | Reasoning result |
|-----------------|---------------|----------------|-----------------|
| Calculator       | ~60%          | None (lookup)  | 0.000           |
| Subitizing       | ~91%          | Minimal        | 0.000           |
| NLI contradiction| Variable      | High (inference)| ???             |

If NLI works where the others didn't, the missing ingredient is inferential
depth, not surface symmetry. The model needs to see pairs where understanding
*why* something is wrong requires processing meaning, not just noticing a
token changed.

If NLI also fails at 3.3M but works at 7M (4 layers), that confirms reasoning
needs architectural depth for any signal type.

## Implementation notes

- Use the datasets library to load SNLI: `load_dataset("stanfordnlp/snli")`
- Filter for label == 2 (contradiction)
- For the entailment-vs-contradiction format, you need to match premises
  that have both an entailment and contradiction hypothesis
- Keep the same disjoint train-eval protocol
- Log token overlap statistics for the selected pairs
