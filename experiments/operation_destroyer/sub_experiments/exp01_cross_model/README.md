# Exp01: Cross-Model STEM Accuracy

**Question:** How accurately do frozen base models rank the correct STEM answer above 4 distractors, using log-probability comparison?

**Method:** `eval_stem_crossmodel.py` on 97 ASCII-formatted STEM facts, 6 models.

**Result:** GPT-2=15.5% (random), SmolLM2-360M=61.9%, Qwen3-4B=76.3%.
Clear scaling signal. Statistics is hardest (0–60%). Hard difficulty best discriminates model size.

**Key file:** `results.json` (copy of `results/operation_destroyer/stem_crossmodel/stem_crossmodel.json`)
