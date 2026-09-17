#!/usr/bin/env python3
"""Deterministic synthetic FASTA for the e2e tests. No real data is used.

Emits records built from one base string (so the near-duplicate collapse stage
has clusters to reduce) plus fully-independent random strings (clear outliers
the frequency filter should remove). Seeded, so the "golden" promoted set is
reproducible across runs and machines.

    python3 gen_fixture.py [N] [OUT]
"""
import random
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 600
OUT = sys.argv[2] if len(sys.argv) > 2 else "fixture.fasta"

random.seed(1234)
base = "".join(random.choice("ACGT") for _ in range(600))
recs = []
for i in range(N):
    if i % 40 == 0:
        seq = "".join(random.choice("ACGT") for _ in range(600))  # outlier
    elif i % 20 == 0:
        seq = base                                                # exact duplicate
    else:
        s = list(base)
        for _ in range(random.randint(1, 25)):
            s[random.randrange(len(s))] = random.choice("ACGT")
        seq = "".join(s)
    recs.append(f">rec{i}|2021-{(i % 12) + 1:02d}-15|loc\n{seq}\n")

with open(OUT, "w") as fh:
    fh.write("".join(recs))
print(f"wrote {OUT}: {len(recs)} records", file=sys.stderr)
