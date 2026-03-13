#!/usr/bin/env python3
"""Build extended truth_dict files for new factual domains.

Each domain gets its own truth_dict_{domain}_contrastive.json file,
compatible with eval_mc.py and train_mc.py.

Domains:
  - biology     : cell biology, genetics, evolution facts
  - chemistry   : reactions, properties, periodic table
  - statistics  : probability, distributions, theorems
  - programming : algorithms, data structures, language facts
  - logic       : classical logic, set theory, proof concepts
  - astronomy   : solar system, stars, cosmology
  - medicine    : anatomy, physiology (no sensitive clinical content)

Run: python experiments/operation_destroyer/build_extended_truth_dicts.py
"""

import json
import os

OUT_DIR = "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/experiments/operation_destroyer"

# ─────────────────────────────────────────────────────────────
# Each entry: (context, truth, [distractor1, ..., distractor4])
# "The {context} is {truth}" should be a valid true sentence.
# ─────────────────────────────────────────────────────────────

DOMAINS = {

"biology": [
    ("number of chromosomes in a human cell",      "46",
     ["44", "48", "23", "92"]),
    ("molecule that carries genetic information",   "DNA",
     ["RNA", "ATP", "protein", "lipid"]),
    ("powerhouse of the cell",                      "mitochondria",
     ["nucleus", "ribosome", "lysosome", "Golgi"]),
    ("process by which plants make food",           "photosynthesis",
     ["respiration", "fermentation", "osmosis", "transpiration"]),
    ("building blocks of proteins",                 "amino acids",
     ["nucleotides", "fatty acids", "sugars", "glycerol"]),
    ("organ that produces insulin",                 "pancreas",
     ["liver", "kidney", "spleen", "thyroid"]),
    ("discoverer of the cell",                      "Hooke",
     ["Pasteur", "Leeuwenhoek", "Darwin", "Mendel"]),
    ("father of genetics",                          "Mendel",
     ["Darwin", "Watson", "Crick", "Morgan"]),
    ("number of base pairs in the human genome",    "3 billion",
     ["3 million", "300 million", "30 billion", "3 trillion"]),
    ("type of bond holding DNA strands together",   "hydrogen",
     ["covalent", "ionic", "peptide", "disulfide"]),
    ("phase of mitosis where chromosomes align",    "metaphase",
     ["prophase", "anaphase", "telophase", "interphase"]),
    ("molecule that carries amino acids to ribosome","tRNA",
     ["mRNA", "rRNA", "DNA", "siRNA"]),
    ("process converting glucose to ATP anaerobically","glycolysis",
     ["Krebs cycle", "oxidative phosphorylation", "beta oxidation", "gluconeogenesis"]),
    ("scientist who proposed natural selection",    "Darwin",
     ["Lamarck", "Mendel", "Wallace", "Huxley"]),
    ("longest cell in the human body",              "neuron",
     ["muscle fiber", "osteocyte", "hepatocyte", "red blood cell"]),
],

"chemistry": [
    ("most abundant element in the universe",       "hydrogen",
     ["helium", "oxygen", "carbon", "nitrogen"]),
    ("most abundant element in Earth's crust",      "oxygen",
     ["silicon", "aluminum", "iron", "calcium"]),
    ("pH of pure water",                            "7",
     ["6", "8", "5", "9"]),
    ("number of electrons in a neutral carbon atom","6",
     ["4", "8", "12", "14"]),
    ("melting point of iron in Celsius",            "1538",
     ["1200", "900", "2000", "1100"]),
    ("chemical formula for table salt",             "NaCl",
     ["KCl", "MgCl2", "CaCl2", "NaBr"]),
    ("chemical formula for water",                  "H2O",
     ["H2O2", "HO", "H3O", "H2S"]),
    ("number of elements in the periodic table",    "118",
     ["108", "112", "120", "116"]),
    ("lightest noble gas",                          "helium",
     ["neon", "argon", "krypton", "radon"]),
    ("element with highest electronegativity",      "fluorine",
     ["oxygen", "nitrogen", "chlorine", "bromine"]),
    ("type of bond in NaCl",                        "ionic",
     ["covalent", "metallic", "hydrogen", "van der Waals"]),
    ("number of protons in a gold atom",            "79",
     ["78", "80", "76", "82"]),
    ("gas produced in cellular respiration",        "CO2",
     ["O2", "N2", "H2", "CH4"]),
    ("inventor of the periodic table",              "Mendeleev",
     ["Bohr", "Dalton", "Lavoisier", "Curie"]),
    ("state of matter with no definite shape or volume","gas",
     ["liquid", "solid", "plasma", "gel"]),
],

"statistics": [
    ("mean of a standard normal distribution",      "0",
     ["1", "-1", "0.5", "undefined"]),
    ("standard deviation of a standard normal",     "1",
     ["0", "2", "0.5", "sqrt(2)"]),
    ("probability of any event",                    "between 0 and 1",
     ["between -1 and 1", "between 0 and 100", "greater than 0", "between 0 and 2"]),
    ("theorem relating sample mean to normal dist", "Central Limit Theorem",
     ["Bayes Theorem", "Law of Large Numbers", "Chebyshev's inequality", "Berry-Esseen theorem"]),
    ("expected value of a fair six-sided die",      "3.5",
     ["3", "4", "3.4", "3.6"]),
    ("correlation coefficient range",               "-1 to 1",
     ["0 to 1", "-infinity to infinity", "-2 to 2", "0 to 100"]),
    ("number of parameters in a normal distribution","2",
     ["1", "3", "4", "2.5"]),
    ("statistical test for comparing two means",    "t-test",
     ["chi-squared test", "ANOVA", "F-test", "z-test"]),
    ("distribution of count data",                  "Poisson",
     ["Gaussian", "Binomial", "Exponential", "Uniform"]),
    ("measure of spread around the mean",           "variance",
     ["median", "mode", "range", "skewness"]),
    ("p-value threshold for statistical significance","0.05",
     ["0.01", "0.1", "0.5", "0.001"]),
    ("type of error rejecting a true null hypothesis","Type I",
     ["Type II", "Type III", "alpha error", "beta error"]),
    ("Bayes theorem relates",                       "prior and posterior probability",
     ["mean and variance", "correlation and causation", "samples and populations", "type I and type II error"]),
    ("distribution of sum of squared standard normals","chi-squared",
     ["t-distribution", "F-distribution", "beta", "Poisson"]),
    ("estimator that minimizes mean squared error",  "minimum variance unbiased estimator",
     ["maximum likelihood estimator", "method of moments", "Bayesian estimator", "robust estimator"]),
],

"programming": [
    ("time complexity of binary search",            "O(log n)",
     ["O(n)", "O(n log n)", "O(1)", "O(n^2)"]),
    ("time complexity of bubble sort",              "O(n^2)",
     ["O(n log n)", "O(n)", "O(log n)", "O(n^3)"]),
    ("data structure using LIFO",                   "stack",
     ["queue", "heap", "tree", "graph"]),
    ("data structure using FIFO",                   "queue",
     ["stack", "heap", "deque", "list"]),
    ("HTTP method for creating resources",          "POST",
     ["GET", "PUT", "DELETE", "PATCH"]),
    ("HTTP method for reading resources",           "GET",
     ["POST", "PUT", "FETCH", "READ"]),
    ("language created by Bjarne Stroustrup",       "C++",
     ["Java", "C#", "Rust", "Go"]),
    ("language created by James Gosling",           "Java",
     ["C#", "Kotlin", "Scala", "Groovy"]),
    ("number of bits in a byte",                    "8",
     ["4", "16", "6", "10"]),
    ("base of hexadecimal",                         "16",
     ["8", "2", "10", "12"]),
    ("algorithm for shortest path in a graph",      "Dijkstra",
     ["Bellman-Ford", "Floyd-Warshall", "A-star", "Prim"]),
    ("sorting algorithm with O(n log n) worst case","merge sort",
     ["quicksort", "heapsort", "timsort", "introsort"]),
    ("design pattern for single instance",          "singleton",
     ["factory", "observer", "decorator", "proxy"]),
    ("version control system created by Torvalds",  "Git",
     ["SVN", "Mercurial", "CVS", "Perforce"]),
    ("RFC that defines HTTP/1.1",                   "RFC 2616",
     ["RFC 791", "RFC 2068", "RFC 7230", "RFC 1945"]),
],

"astronomy": [
    ("planet closest to the Sun",                   "Mercury",
     ["Venus", "Mars", "Earth", "Jupiter"]),
    ("largest moon of Saturn",                      "Titan",
     ["Enceladus", "Rhea", "Iapetus", "Mimas"]),
    ("age of the universe in billions of years",    "13.8",
     ["10", "15", "12", "20"]),
    ("distance from Earth to Moon in km",           "384400",
     ["150000", "500000", "100000", "1000000"]),
    ("galaxy containing our solar system",          "Milky Way",
     ["Andromeda", "Triangulum", "Sombrero", "Whirlpool"]),
    ("nearest galaxy to the Milky Way",             "Andromeda",
     ["Triangulum", "Large Magellanic Cloud", "Sombrero", "Pinwheel"]),
    ("planet with the most moons",                  "Saturn",
     ["Jupiter", "Uranus", "Neptune", "Mars"]),
    ("temperature of the Sun's surface in Kelvin",  "5778",
     ["6000", "5000", "10000", "3000"]),
    ("speed of light in km per second",             "299792",
     ["300000", "186000", "150000", "250000"]),
    ("phenomenon when light bends around mass",     "gravitational lensing",
     ["redshift", "blueshift", "parallax", "aberration"]),
    ("type of star our Sun is",                     "G-type main sequence",
     ["K-type", "F-type", "M-type", "A-type"]),
    ("first person to walk on the Moon",            "Armstrong",
     ["Aldrin", "Glenn", "Shepard", "Collins"]),
    ("year of the first Moon landing",              "1969",
     ["1967", "1971", "1965", "1973"]),
    ("planet known as the Red Planet",              "Mars",
     ["Venus", "Mercury", "Jupiter", "Saturn"]),
    ("Hubble Space Telescope launched in year",     "1990",
     ["1985", "1995", "1988", "2000"]),
],

"logic_math": [
    ("number of sides in a hexagon",                "6",
     ["5", "7", "8", "4"]),
    ("sum of angles in a triangle in degrees",      "180",
     ["90", "270", "360", "120"]),
    ("Pythagorean theorem states",                  "a squared plus b squared equals c squared",
     ["a plus b equals c", "a times b equals c", "a cubed plus b cubed equals c cubed", "2a plus 2b equals c"]),
    ("number of prime numbers between 1 and 10",    "4",
     ["3", "5", "6", "2"]),
    ("logarithm base 10 of 1000",                   "3",
     ["2", "4", "10", "100"]),
    ("derivative of sin(x)",                        "cos(x)",
     ["tan(x)", "-cos(x)", "sin(x)", "-sin(x)"]),
    ("integral of 1/x",                             "ln(x)",
     ["1/x^2", "x", "log10(x)", "e^x"]),
    ("number of elements in the empty set",         "0",
     ["1", "undefined", "infinity", "-1"]),
    ("factorial of 0",                              "1",
     ["0", "undefined", "-1", "infinity"]),
    ("number of vertices in a cube",                "8",
     ["6", "12", "4", "10"]),
    ("two to the power of ten",                     "1024",
     ["1000", "512", "2048", "100"]),
    ("Euler's identity involves",                   "e, i, pi, 1, and 0",
     ["e, pi, and sqrt(2)", "e, i, and ln(2)", "pi, phi, and sqrt(5)", "i, e, and gamma"]),
    ("theorem stating every even integer greater than 2 is sum of two primes","Goldbach conjecture",
     ["Riemann hypothesis", "Fermat's last theorem", "twin prime conjecture", "Collatz conjecture"]),
    ("number of faces on a regular tetrahedron",    "4",
     ["6", "8", "12", "20"]),
    ("formula for area of a circle",                "pi r squared",
     ["2 pi r", "pi d", "4 pi r squared", "pi r cubed"]),
],

}


def build_contrastive_dict(domain_name, entries):
    truths = []
    for context, truth, distractors in entries:
        truths.append({
            "context":    context,
            "truth":      truth,
            "category":   domain_name,
            "distractors": distractors[:4],
        })
    return {
        "metadata": {
            "version": "1.0",
            "domain": domain_name,
            "description": f"Factual truth anchors for {domain_name} domain",
            "n_entries": len(truths),
        },
        "truths": truths,
    }


if __name__ == "__main__":
    all_truths = []
    for domain, entries in DOMAINS.items():
        data = build_contrastive_dict(domain, entries)
        path = os.path.join(OUT_DIR, f"truth_dict_{domain}_contrastive.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  {domain:15s}: {len(entries):3d} facts → {path}")
        all_truths.extend(data["truths"])

    # Combined file — all domains together
    combined = {
        "metadata": {
            "version": "1.0",
            "description": "All domains combined: core + biology + chemistry + statistics + programming + astronomy + logic_math",
            "n_entries": len(all_truths),
            "domains": list(DOMAINS.keys()),
        },
        "truths": all_truths,
    }
    combined_path = os.path.join(OUT_DIR, "truth_dict_all_contrastive.json")
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n  Combined: {len(all_truths)} facts → {combined_path}")

    # Also merge with original core dict
    core_path = os.path.join(OUT_DIR, "truth_dict_contrastive.json")
    if os.path.exists(core_path):
        with open(core_path) as f:
            core = json.load(f)
        mega_truths = core["truths"] + all_truths
        mega = {
            "metadata": {
                "version": "1.0",
                "description": "Core 115 facts + all extended domains",
                "n_entries": len(mega_truths),
            },
            "truths": mega_truths,
        }
        mega_path = os.path.join(OUT_DIR, "truth_dict_mega_contrastive.json")
        with open(mega_path, "w") as f:
            json.dump(mega, f, indent=2)
        print(f"  Mega:     {len(mega_truths)} facts → {mega_path}")
