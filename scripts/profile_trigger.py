#!/usr/bin/env python3
"""
Profile the L1 trigger's phases to find the TRUE bottleneck before optimizing.
Times, separately, on a real FASTA slice:
  (A) I/O + FASTA parse (stream_fasta)
  (B) k-mer extraction + hashing (genome_kmers -- numpy vectorized)
  (C) np.unique per genome
  (D) dict-lookup frequency scan (the suspected killer: Python loop + int() boxing)
  (E) dict update (corpus count maintenance)
Reports per-phase wall-time, % of total, and per-genome cost.
Also runs cProfile on the whole for a function-level view.
"""
import argparse, time, cProfile, pstats, io as _io
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from l1_trigger_compressor import genome_kmers, stream_fasta, binomial_cutoff

def profile_phases(path, k, limit):
    t = {'io':0.0,'kmer':0.0,'uniq':0.0,'lookup':0.0,'update':0.0}
    kmer_df = {}
    n = 0
    total0 = time.perf_counter()
    it = stream_fasta(path)
    while True:
        t0 = time.perf_counter()
        try:
            hdr, seq = next(it)
        except StopIteration:
            break
        t1 = time.perf_counter(); t['io'] += t1 - t0
        km = genome_kmers(seq, k)
        t2 = time.perf_counter(); t['kmer'] += t2 - t1
        if km.size == 0:
            continue
        uniq = np.unique(km)
        t3 = time.perf_counter(); t['uniq'] += t3 - t2
        # (D) the suspected hot path
        dfs = np.fromiter((kmer_df.get(int(x), 0) for x in uniq), dtype=np.int64, count=uniq.size)
        t4 = time.perf_counter(); t['lookup'] += t4 - t3
        for x in uniq:
            xi = int(x); kmer_df[xi] = kmer_df.get(xi, 0) + 1
        t5 = time.perf_counter(); t['update'] += t5 - t4
        n += 1
        if limit and n >= limit:
            break
    total = time.perf_counter() - total0
    print(f"\n=== PHASE PROFILE ({n} genomes, k={k}) ===")
    print(f"total wall: {total:.2f}s   throughput {n/total:.1f} seq/s   distinct kmers {len(kmer_df)}")
    for name, lbl in [('io','I/O+parse'),('kmer','kmer+hash(numpy)'),('uniq','np.unique'),
                      ('lookup','dict-LOOKUP(py loop)'),('update','dict-UPDATE(py loop)')]:
        print(f"  {lbl:24s} {t[name]:7.2f}s  {100*t[name]/total:5.1f}%   {1000*t[name]/n:.2f} ms/genome")
    accounted = sum(t.values())
    print(f"  {'(unaccounted)':24s} {total-accounted:7.2f}s  {100*(total-accounted)/total:5.1f}%")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--k', type=int, default=15)
    ap.add_argument('--limit', type=int, default=3000)
    ap.add_argument('--cprofile', action='store_true')
    args = ap.parse_args()
    if args.cprofile:
        pr = cProfile.Profile(); pr.enable()
    profile_phases(args.input, args.k, args.limit)
    if args.cprofile:
        pr.disable()
        s = _io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(20)
        print("\n=== cProfile (top 20 cumulative) ===\n" + s.getvalue())

if __name__ == '__main__':
    main()
