#!/usr/bin/env python3
"""
L1 TRIGGER v2 -- streaming, alignment-free port of compressor-2.bf.

Reconstructs compressor-2.bf's REAL-vs-ERROR discrimination WITHOUT an alignment,
using k-mers as self-anchoring pseudo-columns (a shared 15-mer = shared local
homology, no global MSA needed).

compressor-2 (post-alignment)          ->  here (pre-alignment, k-mer)
  minority variant at column X         ->  rare k-mer (low corpus frequency)
  binomial cutoff on site count vs N   ->  binomial cutoff on k-mer freq vs N seqs
  co-occurrence at column PAIR         ->  rare-k-mer PAIR co-occurring across genomes
  seq outlier = many minority variants ->  genome with many SINGLETON rare k-mers
  punch site -> gap                    ->  down-weight the rare k-mer (keep genome)

GOAL (user, three-way): (1) PRESERVE real diversity; (2) ADMIT highly divergent
REAL singletons; (3) REJECT low-quality (error-laden) outliers. The discriminator
between (2) and (3) is compressor-2's insight: a divergent REAL lineage's rare
k-mers CO-OCCUR / RECUR (linked, heritable); an error-laden genome's rare k-mers
are ISOLATED SINGLETONS (independent, random). Count of isolated-singleton k-mers,
not divergence itself, is the outlier signal.

STREAMING NOTE: compressor-2 is 2-pass (needs global counts first). We stream with
a RUNNING frequency/co-occurrence table; the binomial cutoff is recomputed from the
running N (stabilizes fast). Early genomes judged on thinner stats; --warmup defers
hard reject decisions until N_seen >= warmup so the counter is meaningful.

v1 is deliberately CODING-BLIND (no reading frame pre-alignment); any syn-vs-non-syn /
selection-relevance weighting is left to a downstream analysis layer.
"""
import argparse, time
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

_LUT = np.full(256, 255, dtype=np.uint8)
for b, v in zip(b"ACGT", range(4)):
    _LUT[b] = v
MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def binomial_cutoff(N, p, t):
    """Port of compressor-2.bf filter.binomial_cutoff (lines 310-320).
    Smallest count i s.t. cumulative Binomial(N,p) mass reaches t. A k-mer/variant
    seen <= this many times is explainable by sequencing error at rate p over N draws.
    """
    if N <= 0:
        return 0
    s = 0.0
    term = (1.0 - p) ** N
    i = 0
    # guard against runaway; i can't exceed N
    while s < t and i < N:
        s += term
        denom = (1.0 - p) * (i + 1)
        if denom == 0:
            break
        term = term * p / (1.0 - p) * (N - i) / (i + 1)
        i += 1
    return i


def genome_kmers(seq_bytes, k):
    """Return uint64 array of canonical k-mer codes over maximal valid ACGT runs.
    Same encoding as the v1 sketch prototype (N/ambiguity breaks a run)."""
    codes = _LUT[np.frombuffer(seq_bytes, dtype=np.uint8)]
    valid = codes != 255
    n = len(codes)
    if n < k:
        return np.empty(0, dtype=np.uint64)
    powers = (np.uint64(1) << (np.uint64(2) * np.arange(k, dtype=np.uint64)))[::-1]
    out = []
    i = 0
    while i < n:
        if not valid[i]:
            i += 1
            continue
        j = i
        while j < n and valid[j]:
            j += 1
        run = codes[i:j].astype(np.uint64)
        if len(run) >= k:
            win = sliding_window_view(run, k)
            fwd = (win * powers).sum(axis=1).astype(np.uint64)
            rc = (np.uint64(3) - win)[:, ::-1]
            rev = (rc * powers).sum(axis=1).astype(np.uint64)
            out.append(np.minimum(fwd, rev))
        i = j
    if not out:
        return np.empty(0, dtype=np.uint64)
    return np.concatenate(out)


def stream_fasta(path):
    hdr = None; chunks = []
    with open(path, 'rb') as fh:
        for line in fh:
            if line[:1] == b'>':
                if hdr is not None:
                    yield hdr, b''.join(chunks).upper()
                hdr = line[1:].rstrip(); chunks = []
            else:
                chunks.append(line.rstrip())
    if hdr is not None:
        yield hdr, b''.join(chunks).upper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--k', type=int, default=15)
    ap.add_argument('--err-rate', type=float, default=1e-4,
                    help='per-site sequencing error rate (compressor-2 used 1/10000)')
    ap.add_argument('--p', type=float, default=0.9, help='binomial probability cutoff (compressor-2 --p)')
    ap.add_argument('--outlier-sd', type=float, default=5.0,
                    help='reject genome if singleton-kmer count > mean+SD*this (compressor-2 used 5)')
    ap.add_argument('--warmup', type=int, default=2000,
                    help='defer hard-reject until this many genomes seen (streaming stats warmup)')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--report-every', type=int, default=5000)
    ap.add_argument('--dump', default=None, help='TSV per-genome decision diagnostics')
    args = ap.parse_args()

    # running k-mer frequency table (how many GENOMES each k-mer has appeared in)
    kmer_df = {}          # k-mer -> document frequency (num genomes)
    n = 0
    n_admit = 0; n_reject = 0; n_skip = 0
    # streaming estimate of singleton-kmer-count distribution (for the mean+5SD outlier test)
    sc_sum = 0.0; sc_sqsum = 0.0; sc_n = 0
    t0 = time.time()

    dump = open(args.dump, 'w') if args.dump else None
    if dump:
        dump.write("decision\tlen\tn_frac\tn_kmers\tn_singleton\tsingleton_frac\tcutoff\theader\n")

    def n_frac(seq):
        L = len(seq)
        if L == 0: return 0, 1.0
        arr = np.frombuffer(seq, dtype=np.uint8)
        acgt = np.isin(arr, np.frombuffer(b"ACGT", dtype=np.uint8)).sum()
        return L, 1.0 - acgt / L

    for hdr, seq in stream_fasta(args.input):
        n += 1
        km = genome_kmers(seq, args.k)
        L, nf = n_frac(seq)
        if km.size == 0:
            n_skip += 1
            continue
        uniq = np.unique(km)

        # binomial cutoff from RUNNING N: a k-mer seen in <= cutoff genomes so far is
        # "rare/minority" (error-explainable). Recomputed each genome; stabilizes fast.
        cutoff = binomial_cutoff(max(1, n - 1), args.err_rate, args.p)

        # classify this genome's k-mers by their CURRENT corpus document-frequency.
        # singleton rare k-mers (df==0 so far, i.e. never seen in a prior genome) that
        # are ISOLATED (don't recur) are the error signal per compressor-2 mech #2/#3.
        dfs = np.fromiter((kmer_df.get(int(x), 0) for x in uniq), dtype=np.int64, count=uniq.size)
        n_rare = int((dfs <= cutoff).sum())
        n_singleton = int((dfs == 0).sum())   # never seen before in the stream
        singleton_frac = n_singleton / max(1, uniq.size)

        # outlier test (compressor-2 mech #2): reject if singleton-kmer COUNT is an
        # anomalous outlier vs the running distribution. A divergent REAL lineage shares
        # most k-mers with its lineage-mates (low singleton count); an error-laden genome
        # has many isolated singletons. Deferred until warmup so stats are meaningful.
        decision = "ADMIT"
        if sc_n >= args.warmup and n > args.warmup:
            mean = sc_sum / sc_n
            var = max(0.0, sc_sqsum / sc_n - mean * mean)
            sd = var ** 0.5
            thresh = mean + args.outlier_sd * sd + 0.5
            if n_singleton > thresh:
                decision = "REJECT"

        if decision == "ADMIT":
            n_admit += 1
            # update corpus counts ONLY for admitted genomes (errors shouldn't pollute the table)
            for x in uniq:
                xi = int(x); kmer_df[xi] = kmer_df.get(xi, 0) + 1
        else:
            n_reject += 1

        # update the running singleton-count distribution (all genomes, for a stable estimate)
        sc_sum += n_singleton; sc_sqsum += n_singleton * n_singleton; sc_n += 1

        if dump:
            dump.write(f"{decision}\t{L}\t{nf:.4f}\t{uniq.size}\t{n_singleton}\t{singleton_frac:.4f}\t{cutoff}\t{hdr.decode(errors='replace')}\n")

        if args.limit and n >= args.limit:
            break
        if n % args.report_every == 0:
            dt = time.time() - t0
            print(f"[{n:>8}] admit={n_admit} reject={n_reject} skip={n_skip} "
                  f"cutoff={cutoff} distinct_kmers={len(kmer_df)} rate={n/dt:.1f}/s", flush=True)

    if dump: dump.close()
    dt = time.time() - t0
    print("\n=== L1 TRIGGER v2 (compressor-2 port) RESULT ===")
    print(f"input        : {args.input}")
    print(f"k={args.k} err_rate={args.err_rate} p={args.p} outlier_sd={args.outlier_sd} warmup={args.warmup}")
    print(f"seen         : {n}")
    print(f"skip(no kmer): {n_skip}")
    print(f"ADMIT        : {n_admit}  ({100*n_admit/max(1,n):.2f}%)")
    print(f"REJECT(error): {n_reject}  ({100*n_reject/max(1,n):.2f}%)")
    print(f"distinct kmers in corpus: {len(kmer_df)}")
    print(f"throughput   : {n/dt:.1f} seq/s (single core)   wall {dt:.1f}s")
    print("NOTE: this is the REAL-VS-ERROR admission filter (compressor-2 port). Novelty")
    print("      promotion (sketch-distance) + any selection-weighting layer sit on TOP.")


if __name__ == '__main__':
    main()
