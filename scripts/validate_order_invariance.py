#!/usr/bin/env python3
"""
Validate the ORDER-INVARIANCE claim for the novelty gate (PLAN §2.x).

Claim: the promoted set is a set-COVER property — different input orderings yield
different-but-EQUIVALENT covers (every cluster of near-duplicates gets a
representative; only WHICH representative, and boundary counts, differ).

Test: run the greedy novelty gate on the SAME admitted genomes in N different
random orders. Measure:
  (1) promoted COUNT per order (drift = max-min / mean)
  (2) MUTUAL COVERAGE: for orders A vs B, what fraction of A's promoted genomes are
      within threshold of SOME B-promoted genome (and vice versa)? ~100% = the covers
      are equivalent (the claim holds); low = order genuinely changes what's covered.

Reuses the prototype's exact sketch (canonical k-mer bottom-m MinHash, Jaccard).
"""
import argparse, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from l1_trigger_prototype import canonical_kmer_hashes, jaccard_est, stream_fasta

def build_sketches(path, k, m, max_n, max_n_frac, min_len):
    """Sketch admitted (quality-passing) genomes. Returns list of (idx, sketch)."""
    rng = np.random.default_rng(42)
    salts = (rng.integers(0, 2**63, size=m, dtype=np.uint64) | np.uint64(1))
    sketches = []
    n = 0
    for hdr, seq in stream_fasta(path):
        n += 1
        # admission quality gate (same spirit as compressor filter's outlier removal, simplified)
        L = len(seq)
        if L < min_len:
            continue
        arr = np.frombuffer(seq, dtype=np.uint8)
        acgt = np.isin(arr, np.frombuffer(b"ACGT", dtype=np.uint8)).sum()
        if 1.0 - acgt / max(1, L) > max_n_frac:
            continue
        sk = canonical_kmer_hashes(seq, k, salts)
        if sk is not None:
            sketches.append(sk)
        if max_n and len(sketches) >= max_n:
            break
    return sketches

def greedy_novelty(sketches, order, threshold):
    """Greedy: iterate in `order`, promote if novelty (1-maxJaccard vs promoted) >= threshold."""
    promoted = []          # list of sketches
    promoted_idx = []      # original indices
    for i in order:
        sk = sketches[i]
        best = 0.0
        for ps in promoted:
            j = jaccard_est(sk, ps)
            if j > best:
                best = j
                if best >= 1.0 - threshold:   # early exit: already too similar
                    break
        if (1.0 - best) >= threshold:
            promoted.append(sk)
            promoted_idx.append(i)
    return promoted_idx

def coverage(promoted_a_idx, promoted_b, sketches, threshold):
    """Fraction of A-promoted genomes within threshold of SOME B-promoted sketch."""
    covered = 0
    for i in promoted_a_idx:
        sk = sketches[i]
        for ps in promoted_b:
            if (1.0 - jaccard_est(sk, ps)) < threshold:   # within threshold = covered
                covered += 1
                break
    return covered / max(1, len(promoted_a_idx))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--k', type=int, default=15)
    ap.add_argument('--sketch', type=int, default=64)
    ap.add_argument('--threshold', type=float, default=0.15)
    ap.add_argument('--max-n', type=int, default=5000, help='cap #sketched genomes (O(P^2) cost)')
    ap.add_argument('--max-n-frac', type=float, default=0.01)
    ap.add_argument('--min-len', type=int, default=25000)
    ap.add_argument('--orders', type=int, default=4, help='# random orderings to compare')
    args = ap.parse_args()

    print(f"building sketches (cap {args.max_n})...", flush=True)
    sketches = build_sketches(args.input, args.k, args.sketch, args.max_n, args.max_n_frac, args.min_len)
    N = len(sketches)
    print(f"admitted+sketched genomes: {N}", flush=True)

    rng = np.random.default_rng(0)
    orders = [np.arange(N)]                              # order 0 = file order
    for o in range(args.orders - 1):
        perm = np.arange(N); rng.shuffle(perm); orders.append(perm)

    promoted_sets = []
    for oi, order in enumerate(orders):
        pid = greedy_novelty(sketches, order, args.threshold)
        promoted_sets.append(pid)
        print(f"order {oi} ({'file' if oi==0 else 'shuffle'+str(oi)}): promoted {len(pid)}", flush=True)

    counts = [len(p) for p in promoted_sets]
    print(f"\n=== COUNT DRIFT ===")
    print(f"promoted counts: {counts}")
    print(f"min {min(counts)}  max {max(counts)}  mean {np.mean(counts):.1f}  "
          f"drift (max-min)/mean = {100*(max(counts)-min(counts))/np.mean(counts):.1f}%")

    print(f"\n=== MUTUAL COVERAGE (order i's promoted covered by order j's) ===")
    covs = []
    for i in range(len(orders)):
        for j in range(len(orders)):
            if i == j: continue
            c = coverage(promoted_sets[i], [sketches[x] for x in promoted_sets[j]], sketches, args.threshold)
            covs.append(c)
            print(f"  cover(order{i} by order{j}) = {100*c:.1f}%")
    print(f"\nmean mutual coverage: {100*np.mean(covs):.1f}%   min {100*min(covs):.1f}%")
    print("\nVERDICT:", "ORDER-INVARIANT (covers equivalent) — two-phase decomposition VALID"
          if np.mean(covs) > 0.98 and (max(counts)-min(counts))/np.mean(counts) < 0.10
          else "ORDER-SENSITIVE — reconsider decomposition")

if __name__ == '__main__':
    main()
