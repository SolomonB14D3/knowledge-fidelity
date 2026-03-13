#!/usr/bin/env python3
"""Generate distractors for truth_dict.json and save truth_dict_contrastive.json.

For each fact, creates 4 plausible-but-wrong distractors.
Category-aware: capitals steal from each other, dates shift ±years, etc.
"""
import json, random

TRUTH_DICT_PATH = "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/experiments/operation_destroyer/truth_dict.json"
OUT_PATH = "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/experiments/operation_destroyer/truth_dict_contrastive.json"

with open(TRUTH_DICT_PATH) as f:
    data = json.load(f)
facts = data["truths"]

# Index all truths by category for cross-category stealing
by_cat = {}
for f in facts:
    by_cat.setdefault(f["category"], []).append(f["truth"])

# Hardcoded distractors for facts that need specific wrong answers
SPECIFIC = {
    # physics
    "speed of light":        ["186000", "300000", "299792", "3e8"],
    "gravitational constant":["6.371", "9.81", "6.022", "1.381"],
    "Planck constant":       ["6.022", "1.381", "1.602", "9.109"],
    "Boltzmann constant":    ["1.602", "6.674", "9.109", "1.673"],
    "Avogadro":              ["6.674", "1.602", "3.14159", "1.381"],
    "electron mass":         ["1.673", "1.602", "6.674", "9.807"],
    "proton mass":           ["9.109", "1.602", "1.381", "6.674"],
    "elementary charge":     ["1.381", "9.109", "6.674", "2.718"],
    "speed of sound":        ["331", "340", "350", "320"],
    "absolute zero":         ["-270", "-273", "-271.15", "-260"],
    # math
    "value of pi":           ["3.14", "3.14159265", "3.1416", "22/7"],
    "value of e":            ["2.718", "2.72", "2.71", "3.14159"],
    "golden ratio":          ["1.414", "1.732", "1.628", "1.618034"],
    "square root of 2":      ["1.41", "1.732", "1.618", "1.732"],
    "Euler-Mascheroni":      ["0.618", "0.7071", "0.5", "0.693"],
    # science
    "largest planet":        ["Saturn", "Neptune", "Uranus", "Mars"],
    "smallest planet":       ["Mars", "Pluto", "Venus", "Earth"],
    "closest star":          ["Alpha Centauri", "Sirius", "Betelgeuse", "Vega"],
    "largest ocean":         ["Atlantic", "Indian", "Arctic", "Southern"],
    "longest river":         ["Amazon", "Mississippi", "Yangtze", "Congo"],
    "tallest mountain":      ["K2", "Kangchenjunga", "Lhotse", "Makalu"],
    "deepest ocean trench":  ["Puerto Rico", "Tonga", "Philippine", "Java"],
    "boiling point of water":["98", "99", "212", "101"],
    "freezing point of water":["32", "-1", "1", "-0.5"],
    "discoverers of the DNA double helix": ["Mendel and Darwin", "Curie and Bohr", "Einstein and Planck", "Franklin and Wilkins"],
    "proposer of the theory of relativity": ["Newton", "Bohr", "Planck", "Heisenberg"],
    "proposer of the theory of evolution":  ["Lamarck", "Mendel", "Wallace", "Huxley"],
    "discoverer of penicillin":    ["Pasteur", "Koch", "Lister", "Jenner"],
    "inventor of the telephone":   ["Edison", "Morse", "Marconi", "Tesla"],
    "inventor of the light bulb":  ["Tesla", "Bell", "Morse", "Swan"],
    # culture
    "author of Romeo and Juliet":      ["Marlowe", "Chaucer", "Milton", "Dickens"],
    "painter of the Mona Lisa":        ["Michelangelo", "Raphael", "Botticelli", "Titian"],
    "composer of the Ninth Symphony":  ["Mozart", "Bach", "Brahms", "Tchaikovsky"],
    "author of The Odyssey":           ["Virgil", "Sophocles", "Euripides", "Plato"],
    "author of 1984":                  ["Aldous Huxley", "H.G. Wells", "Ray Bradbury", "Arthur Koestler"],
    "painter of Starry Night":         ["Monet", "Gauguin", "Cezanne", "Matisse"],
    "author of The Republic":          ["Aristotle", "Socrates", "Cicero", "Seneca"],
    "author of Don Quixote":           ["Lope de Vega", "Calderon", "Quevedo", "Tirso"],
    # history
    "first president of the United States":  ["John Adams", "Thomas Jefferson", "Benjamin Franklin", "James Madison"],
    "16th president of the United States":   ["Ulysses Grant", "Andrew Jackson", "James Buchanan", "Andrew Johnson"],
    "president during World War II":         ["Wilson", "Truman", "Hoover", "Taft"],
    "first emperor of China":                ["Han Gaozu", "Emperor Wu", "Liu Bang", "Sun Yat-sen"],
    "civilization that built the pyramids":  ["Rome", "Greece", "Mesopotamia", "Nubia"],
    "birthplace of the Renaissance":         ["France", "Germany", "Spain", "Greece"],
    "century of the Industrial Revolution":  ["17th century", "19th century", "16th century", "20th century"],
    # geography
    "number of continents":      ["6", "8", "5", "9"],
    "number of oceans":          ["4", "6", "3", "7"],
    "most populous country":     ["China", "United States", "Indonesia", "Pakistan"],
    "largest country by area":   ["Canada", "China", "United States", "Brazil"],
    "smallest country":          ["Monaco", "San Marino", "Liechtenstein", "Nauru"],
    "continent of the Sahara Desert":       ["Asia", "Middle East", "South America", "Australia"],
    "continent of the Amazon rainforest":   ["Africa", "Central America", "Southeast Asia", "North America"],
    "country of the Great Wall of China":   ["Mongolia", "Japan", "Korea", "Tibet"],
    # computing
    "binary for 10":             ["1001", "1100", "0110", "1011"],
    "hexadecimal for 255":       ["FE", "EF", "F0", "0F"],
    "HTTP status 404":           ["Forbidden", "Unauthorized", "Bad Request", "Server Error"],
    "HTTP status 200":           ["Created", "Accepted", "No Content", "Found"],
    "TCP port for HTTP":         ["8080", "443", "8000", "21"],
    "TCP port for HTTPS":        ["80", "8443", "8080", "22"],
    "inventor of the World Wide Web": ["Bill Gates", "Steve Jobs", "Vint Cerf", "Alan Turing"],
    "creator of Linux":          ["Richard Stallman", "Dennis Ritchie", "Ken Thompson", "Andrew Tanenbaum"],
    "creator of Python":         ["James Gosling", "Bjarne Stroustrup", "Larry Wall", "Yukihiro Matsumoto"],
    "founder of Apple":          ["Bill Gates", "Steve Wozniak", "Jony Ive", "Tim Cook"],
    "founder of Microsoft":      ["Steve Jobs", "Larry Ellison", "Mark Zuckerberg", "Jeff Bezos"],
}

# Date shifting for date category
def date_distractors(truth):
    try:
        yr = int(truth)
        shifts = [-15, -8, +7, +12]
        return [str(yr + s) for s in shifts]
    except:
        return []

# Chemical element distractors
ELEMENT_DISTRACTORS = {
    "chemical symbol for gold":    ["Go", "Gd", "Ag", "Gl"],
    "chemical symbol for silver":  ["Si", "Sr", "Sv", "Sl"],
    "chemical symbol for iron":    ["Ir", "In", "Fr", "Fo"],
    "chemical symbol for copper":  ["Co", "Cp", "Cr", "Cm"],
    "chemical symbol for sodium":  ["So", "Sd", "Sn", "Nm"],
    "chemical symbol for potassium":["Po", "Pt", "Pm", "Ks"],
    "atomic number of hydrogen":   ["2", "3", "0", "4"],
    "atomic number of carbon":     ["7", "5", "8", "4"],
    "atomic number of oxygen":     ["6", "9", "7", "10"],
    "atomic number of nitrogen":   ["6", "8", "5", "9"],
    "atomic number of helium":     ["1", "3", "4", "0"],
}

def get_distractors(fact):
    ctx = fact["context"]
    truth = fact["truth"]
    cat = fact["category"]

    # Check specific overrides first
    if ctx in SPECIFIC:
        return SPECIFIC[ctx]
    if ctx in ELEMENT_DISTRACTORS:
        return ELEMENT_DISTRACTORS[ctx]

    # Dates: shift years
    if cat == "dates":
        d = date_distractors(truth)
        if d:
            return d

    # Capitals: steal 4 other capitals
    if cat == "capitals":
        others = [t for t in by_cat["capitals"] if t != truth]
        random.seed(hash(ctx))
        return random.sample(others, min(4, len(others)))

    # Fallback: steal from same category
    others = [t for t in by_cat.get(cat, []) if t != truth]
    if len(others) >= 4:
        random.seed(hash(ctx))
        return random.sample(others, 4)

    # Last resort: steal from all categories
    all_truths = [f["truth"] for f in facts if f["truth"] != truth]
    random.seed(hash(ctx))
    return random.sample(all_truths, 4)


# Build output
contrastive_facts = []
for fact in facts:
    distractors = get_distractors(fact)
    contrastive_facts.append({
        "context":    fact["context"],
        "truth":      fact["truth"],
        "category":   fact["category"],
        "distractors": distractors[:4],
    })

out = {
    "metadata": {
        "version": "2.0",
        "description": "Truth anchors with contrastive distractors (sun + black hole)",
        "n_entries": len(contrastive_facts),
    },
    "truths": contrastive_facts,
}
with open(OUT_PATH, "w") as f:
    json.dump(out, f, indent=2)

print(f"Written {len(contrastive_facts)} facts with distractors to {OUT_PATH}")

# Sanity check
missing = [f for f in contrastive_facts if len(f["distractors"]) < 4]
if missing:
    print(f"WARNING: {len(missing)} facts have <4 distractors:")
    for m in missing:
        print(f"  {m['context']}: {m['distractors']}")
else:
    print("All facts have 4 distractors. ✓")

# Show a sample
import random as r
r.seed(42)
samples = r.sample(contrastive_facts, 5)
print("\nSamples:")
for s in samples:
    print(f"  [{s['category']}] {s['context']!r}")
    print(f"    SUN:   {s['truth']!r}")
    print(f"    HOLES: {s['distractors']}")
