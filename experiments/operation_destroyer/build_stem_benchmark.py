#!/usr/bin/env python3
"""Build a STEM truth benchmark for log-prob MC evaluation.

ASCII-only notation throughout — no unicode minus, superscripts, fractions, or Greek.
Every truth and distractor tokenizes cleanly on ALL models including GPT-2.

Substitutions: - (not minus-sign), ^2 (not superscript), sqrt() (not sqrt-symbol),
               (1/2) (not half-fraction), lambda/mu/sigma/pi (not Greek letters)

Usage:
    python experiments/operation_destroyer/build_stem_benchmark.py
"""

import json, os

OUT_DIR = "/Volumes/4TB SD/ClaudeCode/knowledge-fidelity/experiments/operation_destroyer"

CALCULUS = [
    # Derivatives
    ("derivative of sin(x)",          "cos(x)",            ["-sin(x)", "-cos(x)", "tan(x)", "sin(x)"],                      "easy"),
    ("derivative of cos(x)",          "-sin(x)",           ["sin(x)", "-cos(x)", "cos(x)", "tan(x)"],                       "easy"),
    ("derivative of e^x",             "e^x",               ["x*e^x", "e^(x-1)", "ln(x)", "e^(x+1)"],                       "easy"),
    ("derivative of ln(x)",           "1/x",               ["ln(x)/x", "x*ln(x)", "1/x^2", "e^x"],                         "easy"),
    ("derivative of x^n",             "n*x^(n-1)",         ["(n-1)*x^n", "n*x^n", "x^(n+1)/(n+1)", "n*x^(n+1)"],          "easy"),
    ("derivative of tan(x)",          "sec^2(x)",          ["cos^2(x)", "1/cos(x)", "-csc^2(x)", "tan^2(x)"],              "medium"),
    ("derivative of arctan(x)",       "1/(1+x^2)",         ["1/(1-x^2)", "arctan(x)/x", "1/sqrt(1-x^2)", "-1/(1+x^2)"],   "medium"),
    ("derivative of arcsin(x)",       "1/sqrt(1-x^2)",     ["1/sqrt(1+x^2)", "-1/sqrt(1-x^2)", "cos(x)", "1/(1-x^2)"],    "medium"),
    ("second derivative of sin(x)",   "-sin(x)",           ["sin(x)", "cos(x)", "-cos(x)", "0"],                            "easy"),
    ("derivative of x*ln(x)",         "ln(x)+1",           ["ln(x)", "1/x+ln(x)", "x/ln(x)", "x+ln(x)"],                  "medium"),
    # Integrals
    ("integral of cos(x)",            "sin(x)+C",          ["-sin(x)+C", "cos(x)+C", "tan(x)+C", "sin^2(x)+C"],            "easy"),
    ("integral of sin(x)",            "-cos(x)+C",         ["cos(x)+C", "-sin(x)+C", "sin^2(x)+C", "sin(x)+C"],            "easy"),
    ("integral of e^x",               "e^x+C",             ["x*e^x+C", "e^x/x+C", "ln(x)+C", "e^(x+1)+C"],               "easy"),
    ("integral of 1/x",               "ln|x|+C",           ["1/x^2+C", "x*ln(x)+C", "log(x)+C", "-1/x^2+C"],              "easy"),
    ("integral of x^n where n!=-1",   "x^(n+1)/(n+1)+C",  ["n*x^(n-1)+C", "x^n/n+C", "x^(n+1)+C", "(n+1)*x^n+C"],       "easy"),
    ("integral of sec^2(x)",          "tan(x)+C",          ["sec(x)*tan(x)+C", "sec^3(x)/3+C", "cot(x)+C", "-cot(x)+C"],  "medium"),
    ("integral of 1/(1+x^2)",         "arctan(x)+C",       ["arcsin(x)+C", "ln(1+x^2)+C", "1/(2*x^2)+C", "arctan^2(x)/2+C"], "medium"),
    # Limits
    ("limit of sin(x)/x as x approaches 0", "1",          ["0", "inf", "pi", "undefined"],                                 "medium"),
    ("limit of (1+1/n)^n as n approaches inf", "e",        ["1", "2", "pi", "inf"],                                        "medium"),
    ("limit of (e^x-1)/x as x approaches 0", "1",         ["0", "e", "inf", "1/e"],                                       "medium"),
    # Theorems
    ("fundamental theorem of calculus (derivative of integral)", "f(x)", ["F(x)", "f'(x)", "integral f(x)dx", "F'(x)"],   "medium"),
    ("Taylor series first two terms of e^x around 0", "1+x", ["x+x^2", "e+e*x", "1+x+x^2/2", "1-x"],                    "medium"),
]

PHYSICS = [
    ("Newton's second law (force)",       "F=m*a",             ["F=m*v", "F=m*v^2", "F=m/a", "F=m*a^2"],                  "easy"),
    ("kinetic energy formula",            "(1/2)*m*v^2",       ["m*v^2", "m*v", "(1/2)*m*v", "2*m*v^2"],                  "easy"),
    ("gravitational potential energy",    "m*g*h",             ["m*g/h", "(1/2)*m*g*h", "m*g*h^2", "m^2*g*h"],            "easy"),
    ("momentum formula",                  "m*v",               ["m*v^2", "(1/2)*m*v^2", "m*a", "m/v"],                    "easy"),
    ("work-energy theorem",               "W=delta(KE)",       ["W=delta(PE)", "W=F*t", "W=m*a*t", "W=F/d"],              "medium"),
    ("centripetal acceleration",          "v^2/r",             ["v*r", "v/r", "v^2*r", "2*v/r"],                           "medium"),
    ("period of a simple pendulum",       "2*pi*sqrt(L/g)",    ["2*pi*sqrt(g/L)", "2*pi*L/g", "pi*sqrt(L/g)", "2*pi*sqrt(m/k)"], "hard"),
    ("escape velocity formula",           "sqrt(2*G*M/r)",     ["sqrt(G*M/r)", "2*G*M/r^2", "sqrt(G*M/(2*r))", "G*M/r^2"], "hard"),
    ("Coulomb's law force",               "k*q1*q2/r^2",       ["k*q1*q2/r", "k*q1*q2*r^2", "q1*q2/r^2", "k/(q1*q2*r^2)"], "medium"),
    ("Ohm's law",                         "V=I*R",             ["V=I/R", "V=I*R^2", "V=I+R", "I=V*R"],                    "easy"),
    ("electric power formula",            "P=I*V",             ["P=I^2/V", "P=V/I", "P=I+V", "P=I*V^2"],                  "easy"),
    ("capacitance definition",            "Q/V",               ["V/Q", "Q*V", "Q/V^2", "Q^2/V"],                          "medium"),
    ("energy stored in capacitor",        "(1/2)*C*V^2",       ["C*V^2", "(1/2)*Q*V^2", "C*V^2/(2*Q)", "Q^2/(2*C)"],     "hard"),
    ("first law of thermodynamics",       "delta(U)=Q-W",      ["delta(U)=Q+W", "delta(U)=W-Q", "Q=delta(U)+W", "delta(U)=Q*W"], "medium"),
    ("ideal gas law",                     "P*V=n*R*T",         ["P*V=R*T", "P*V=N*k*T", "P=n*R*T/V", "P*V=n*k*T"],       "easy"),
    ("entropy change (reversible)",       "delta(S)=Q/T",      ["delta(S)=Q*T", "delta(S)=Q*T^2", "delta(S)=T/Q", "delta(S)=Q/T^2"], "medium"),
    ("wave speed formula",                "v=f*lambda",        ["v=f/lambda", "v=f+lambda", "v=lambda/f", "v=f^2*lambda"], "easy"),
    ("photon energy formula",             "E=h*f",             ["E=h*lambda", "E=h/f", "E=h*f^2", "E=f/h"],              "easy"),
    ("de Broglie wavelength",             "h/(m*v)",           ["m*v/h", "h*m*v", "h/(m^2*v)", "m*v/h^2"],               "hard"),
    ("relativistic rest energy formula",  "m*c^2",             ["m*v^2", "(1/2)*m*c^2", "m*c^2/2", "2*m*c^2"],           "medium"),
]

CHEMISTRY = [
    ("atomic number of hydrogen",               "1",                   ["2", "0", "4", "3"],                                "easy"),
    ("atomic number of carbon",                 "6",                   ["4", "8", "12", "14"],                              "easy"),
    ("atomic number of oxygen",                 "8",                   ["6", "10", "16", "12"],                             "easy"),
    ("atomic number of gold",                   "79",                  ["78", "80", "47", "82"],                            "medium"),
    ("most electronegative element",            "fluorine",            ["oxygen", "chlorine", "nitrogen", "neon"],          "medium"),
    ("noble gas in period 2",                   "neon",                ["argon", "helium", "krypton", "xenon"],             "easy"),
    ("number of electrons in a full p subshell","6",                   ["2", "3", "8", "10"],                               "easy"),
    ("Gibbs free energy formula",               "G=H-T*S",             ["G=H+T*S", "G=H*T*S", "G=T*S-H", "G=H/(T*S)"],   "medium"),
    ("Arrhenius equation for rate constant",    "k=A*e^(-Ea/(R*T))",   ["k=A*e^(Ea/(R*T))", "k=Ea/(R*T)", "k=A/e^(Ea/(R*T))", "k=R*T/Ea"], "hard"),
    ("Henderson-Hasselbalch equation",          "pH=pKa+log([A-]/[HA])", ["pH=pKa-log([A-]/[HA])", "pH=pKa+log([HA]/[A-])", "pH=Ka+log([A-]/[HA])", "pH=pKa*log([A-]/[HA])"], "hard"),
    ("Avogadro's number",                       "6.022e23",            ["6.022e22", "6.022e24", "3.011e23", "6.022e19"],   "easy"),
    ("molarity definition",                     "moles per liter",     ["grams per liter", "moles per kilogram", "moles per milliliter", "molecules per liter"], "easy"),
    ("functional group of alcohols",            "hydroxyl",            ["carbonyl", "carboxyl", "amino", "ester"],          "easy"),
    ("functional group of carboxylic acids",    "carboxyl",            ["hydroxyl", "carbonyl", "ester", "aldehyde"],       "easy"),
    ("hybridization of carbon in methane",      "sp3",                 ["sp2", "sp", "sp3d", "sp2d"],                       "medium"),
    ("hybridization of carbon in ethylene",     "sp2",                 ["sp3", "sp", "sp3d", "sp2d"],                       "medium"),
]

LINEAR_ALGEBRA = [
    ("determinant of 2x2 matrix [[a,b],[c,d]]",          "a*d-b*c",          ["a*b-c*d", "a*d+b*c", "a*c-b*d", "a*b+c*d"],   "easy"),
    ("rank of identity matrix In",                        "n",                ["n-1", "n^2", "1", "0"],                         "easy"),
    ("trace of a matrix (definition)",                    "sum of diagonal entries", ["sum of all entries", "product of eigenvalues", "sum of off-diagonal entries", "determinant"], "easy"),
    ("eigenvalue equation",                               "A*v=lambda*v",     ["A*v=lambda", "A=lambda*v", "A*v=v/lambda", "A*lambda=v"], "easy"),
    ("dot product of orthogonal vectors",                 "0",                ["1", "-1", "|u|*|v|", "undefined"],              "easy"),
    ("number of linearly independent columns (definition)","rank",            ["nullity", "dimension", "trace", "determinant"], "medium"),
    ("rank-nullity theorem",                              "rank+nullity=n",   ["rank*nullity=n", "rank-nullity=n", "rank+nullity=n^2", "rank=nullity"], "medium"),
    ("scalar factor in inverse of 2x2 matrix [[a,b],[c,d]]", "1/(a*d-b*c)", ["1/(a*d+b*c)", "a*d-b*c", "1/(a*b-c*d)", "(a*d-b*c)"], "medium"),
    ("Cauchy-Schwarz inequality",                         "|u*v| <= |u|*|v|", ["|u*v| >= |u|*|v|", "|u*v| = |u|*|v|", "|u*v| < |u|+|v|", "|u*v| <= |u|+|v|"], "medium"),
    ("characteristic polynomial (definition)",            "det(A-lambda*I)",  ["det(A+lambda*I)", "tr(A-lambda*I)", "det(A)-lambda", "det(lambda*I-A)"], "medium"),
    ("spectral theorem applies to",                       "symmetric matrices", ["invertible matrices", "diagonal matrices", "upper triangular matrices", "square matrices"], "hard"),
    ("matrix multiplication is",                          "associative",      ["commutative", "distributive only", "neither associative nor commutative", "always invertible"], "medium"),
]

STATISTICS = [
    ("Bayes theorem formula",                    "P(A|B)=P(B|A)*P(A)/P(B)", ["P(A|B)=P(A)*P(B|A)", "P(A|B)=P(B|A)/P(A)", "P(A|B)=P(A and B)/P(A)", "P(A|B)=P(B|A)*P(B)/P(A)"], "medium"),
    ("mean of normal distribution N(mu,sigma^2)","mu",                      ["sigma^2", "sigma", "mu+sigma", "0"],               "easy"),
    ("variance of normal distribution N(mu,sigma^2)", "sigma^2",            ["sigma", "mu", "mu^2", "sqrt(sigma)"],              "easy"),
    ("central limit theorem result",             "sample mean is approximately normal", ["population is normal", "sample variance equals sigma^2/n", "all samples are identical", "sample median equals mean"], "medium"),
    ("law of large numbers result",              "sample mean converges to population mean", ["sample variance goes to 0", "all samples become equal", "standard deviation increases", "mean equals median"], "medium"),
    ("standard error of the mean",               "sigma/sqrt(n)",            ["sigma/n", "sigma^2/n", "sigma*sqrt(n)", "sigma/n^2"], "medium"),
    ("chi-squared distribution is used for",     "testing categorical data independence", ["comparing two means", "linear regression slope", "testing normality only", "confidence intervals for means"], "hard"),
    ("p-value definition",                       "probability of observing result if null hypothesis is true", ["probability the null hypothesis is true", "probability of type I error", "probability the alternative is true", "confidence level"], "hard"),
    ("maximum likelihood estimation finds",      "parameter that maximizes P(data|parameter)", ["parameter closest to prior mean", "parameter with minimum variance", "parameter that minimizes MSE", "posterior mode"], "hard"),
    ("covariance of independent variables",      "0",                        ["1", "sigma1*sigma2", "undefined", "-1"],           "medium"),
]

CONSTANTS = [
    ("value of pi (first 5 digits)",                    "3.14159",   ["3.14259", "3.14152", "3.14169", "3.14195"], "easy"),
    ("Euler's number e (first 5 digits)",               "2.71828",   ["2.71823", "2.71818", "2.71829", "2.71726"], "medium"),
    ("golden ratio phi (first 5 digits)",               "1.61803",   ["1.61813", "1.61903", "1.61808", "1.61703"], "medium"),
    ("Euler-Mascheroni constant gamma (first 4 digits)","0.5772",    ["0.5182", "0.6772", "0.5872", "0.5072"],     "hard"),
    ("sqrt(2) first 5 digits",                          "1.41421",   ["1.41412", "1.41521", "1.41431", "1.41321"], "easy"),
    ("sqrt(3) first 4 digits",                          "1.7320",    ["1.7230", "1.7420", "1.7310", "1.7730"],    "medium"),
    ("Euler's identity e^(i*pi)+1 equals",              "0",         ["i", "-1", "1", "2*pi*i"],                   "medium"),
    ("sin(pi/6)",                                       "1/2",       ["sqrt(3)/2", "sqrt(2)/2", "1/sqrt(3)", "sqrt(3)/3"], "easy"),
    ("cos(pi/3)",                                       "1/2",       ["sqrt(3)/2", "sqrt(2)/2", "1", "0"],          "easy"),
    ("sin(pi/4)",                                       "sqrt(2)/2", ["1/2", "sqrt(3)/2", "1", "1/sqrt(2)"],       "easy"),
    ("cos(pi/6)",                                       "sqrt(3)/2", ["1/2", "sqrt(2)/2", "sqrt(3)/3", "1"],       "easy"),
    ("tan(pi/4)",                                       "1",         ["sqrt(3)", "1/sqrt(3)", "0", "sqrt(2)"],      "easy"),
    ("ln(1)",                                           "0",         ["1", "e", "-1", "undefined"],                  "easy"),
    ("ln(e)",                                           "1",         ["e", "0", "2", "1/e"],                        "easy"),
    ("log base 10 of 1000",                             "3",         ["2", "4", "10", "100"],                       "easy"),
    ("sum of angles in a triangle (degrees)",           "180",       ["360", "90", "270", "120"],                   "easy"),
    ("Pythagorean theorem",                             "a^2+b^2=c^2", ["a+b=c", "a^2+b=c^2", "a^2*b^2=c^2", "a^2-b^2=c^2"], "easy"),
]

DOMAINS = {
    "calculus":       CALCULUS,
    "physics":        PHYSICS,
    "chemistry":      CHEMISTRY,
    "linear_algebra": LINEAR_ALGEBRA,
    "statistics":     STATISTICS,
    "constants":      CONSTANTS,
}

def make_fact(context, truth, distractors, difficulty, domain):
    assert len(distractors) == 4, f"Need exactly 4 distractors for '{context}'"
    return {"context": context, "truth": truth, "distractors": distractors,
            "domain": domain, "difficulty": difficulty}

all_facts = []
per_domain = {}

for domain, rows in DOMAINS.items():
    facts = [make_fact(*row, domain) for row in rows]
    per_domain[domain] = facts
    all_facts.extend(facts)
    path = os.path.join(OUT_DIR, f"stem_bench_{domain}.json")
    with open(path, "w") as f:
        json.dump({"domain": domain, "truths": facts}, f, indent=2)
    print(f"  {domain:<18}: {len(facts):>3} facts -> {path}")

combined_path = os.path.join(OUT_DIR, "stem_bench_all.json")
with open(combined_path, "w") as f:
    json.dump({"domain": "all", "truths": all_facts}, f, indent=2)

diff_counts = {}
for f in all_facts:
    diff_counts[f["difficulty"]] = diff_counts.get(f["difficulty"], 0) + 1

print(f"\n  Combined: {len(all_facts)} facts -> {combined_path}")
print(f"  Difficulty: easy={diff_counts.get('easy',0)}, medium={diff_counts.get('medium',0)}, hard={diff_counts.get('hard',0)}")
print(f"  Domains: {', '.join(f'{d}({len(v)})' for d,v in per_domain.items())}")
