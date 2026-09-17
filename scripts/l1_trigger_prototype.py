#!/usr/bin/env python3
"""
L1 TRIGGER PROTOTYPE -- alignment-free streaming novelty gate (reference oracle).

Detector-physics framing: this is the "L1 trigger" that runs at ingest rate and
decides, per genome, PROMOTE (novel enough -> goes to L2 reconstruction) vs DROP
(near-duplicate of something already promoted -> just increment its counter).

Signal = SKETCH-DISTANCE NOVELTY (user decision 2026-09-08):
  - MinHash sketch of each genome's k-mer set (bottom-m via multiply-shift hash).
  - novelty = 1 - max Jaccard-estimate over already-promoted sketches.
  - promote iff novelty >= --threshold ; else drop (increment nearest's count).

This measures the two numbers that decide whether "near-real-time" is feasible:
  (1) THROUGHPUT   -- genomes/sec a single core sustains (streaming, no alignment).
  (2) REJECTION    -- fraction dropped = how much the L1 shields the L2 farm.

Nearest-neighbour search here is the honest-but-naive O(P) scan over the promoted
set (P grows). That's the WORST case; production would use an LSH index over the
MinHash bands (O(1) expected). We report both raw throughput and the P-scan cost
so the LSH speedup is quantifiable later.
"""
import argparse, sys, time
import numpy as np

# ---- multiply-shift 64-bit hashing of k-mers (vectorized, no external deps) ----
# Map A,C,G,T -> 2-bit codes; other bases (N, gaps) break the k-mer (skipped).
_LUT = np.full(256, 255, dtype=np.uint8)
for b, v in zip(b"ACGT", range(4)):
    _LUT[b] = v
_LUT[ord('a'):ord('a')+0]  # noqa (keep uppercase only; input is uppercased)

# Two odd 64-bit multipliers for a pair of independent hash functions -> we only
# need ONE permutation for a bottom-m MinHash, but keep a salt array for m hashes.
MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)

def canonical_kmer_hashes(seq_bytes, k, salts):
    """Return an (m,) uint64 min-hash sketch for one sequence.
    Rolling 2-bit encoding of all valid k-mers (canonical = min(fwd,rev-comp)),
    hashed by m multiply-shift functions; bottom value per function = sketch.
    """
    codes = _LUT[np.frombuffer(seq_bytes, dtype=np.uint8)]
    valid = codes != 255
    n = len(codes)
    if n < k:
        return None
    # rolling forward + reverse-complement 2-bit packing over maximal valid runs
    m = len(salts)
    sketch = np.full(m, np.iinfo(np.uint64).max, dtype=np.uint64)
    kmers = []
    i = 0
    while i < n:
        if not valid[i]:
            i += 1
            continue
        j = i
        while j < n and valid[j]:
            j += 1
        run = codes[i:j].astype(np.uint64)  # length L, all in 0..3
        L = len(run)
        if L >= k:
            # forward k-mer integer via sliding window (base-4)
            # build (L-k+1, ) fwd codes
            powers = (np.uint64(1) << (np.uint64(2) * np.arange(k, dtype=np.uint64)))[::-1]
            # vectorized windowed dot: use stride trick
            from numpy.lib.stride_tricks import sliding_window_view
            win = sliding_window_view(run, k)           # (L-k+1, k)
            fwd = (win * powers).sum(axis=1).astype(np.uint64)
            # reverse complement: comp = 3-code, reversed order
            rc_win = (np.uint64(3) - win)[:, ::-1]
            rev = (rc_win * powers).sum(axis=1).astype(np.uint64)
            canon = np.minimum(fwd, rev)
            kmers.append(canon)
        i = j
    if not kmers:
        return None
    allk = np.concatenate(kmers)
    # m independent multiply-shift hashes; bottom value each = MinHash
    for hi, salt in enumerate(salts):
        h = (allk * salt) & MASK64
        h ^= (h >> np.uint64(29))
        sketch[hi] = h.min()
    return sketch

def jaccard_est(a, b):
    """MinHash Jaccard estimate = fraction of matching bottom-hashes."""
    return np.mean(a == b)

def stream_fasta(path):
    """Yield (header, seq_bytes_uppercased) streaming, record-aware, low-mem."""
    hdr = None
    chunks = []
    with open(path, 'rb') as fh:
        for line in fh:
            if line[:1] == b'>':
                if hdr is not None:
                    yield hdr, b''.join(chunks).upper()
                hdr = line[1:].rstrip()
                chunks = []
            else:
                chunks.append(line.rstrip())
    if hdr is not None:
        yield hdr, b''.join(chunks).upper()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--k', type=int, default=15)
    ap.add_argument('--sketch', type=int, default=64, help='MinHash size m')
    ap.add_argument('--threshold', type=float, default=0.15,
                    help='promote if novelty (1-maxJaccard) >= threshold')
    ap.add_argument('--limit', type=int, default=0, help='cap #seqs (0=all)')
    ap.add_argument('--report-every', type=int, default=10000)
    ap.add_argument('--dump-promoted', default=None,
                    help='TSV: per-promoted-genome composition (len,N-frac,novelty) + sampled drops')
    # QUALITY GATE (fix for §2.8: junk-driven novelty). A genome must pass these to
    # be PROMOTABLE at all; failing genomes are dropped as low-quality (not promoted).
    ap.add_argument('--max-n-frac', type=float, default=0.01,
                    help='max non-ACGT fraction to be promotion-eligible (legacy used 0.005-0.01)')
    ap.add_argument('--min-len', type=int, default=25000,
                    help='min sequence length to be promotion-eligible')
    args = ap.parse_args()

    rng = np.random.default_rng(42)
    # odd salts for multiply-shift
    salts = (rng.integers(0, 2**63, size=args.sketch, dtype=np.uint64) | np.uint64(1))

    promoted = []       # list of (m,) uint64 sketches
    promoted_counts = []
    n = 0
    n_promoted = 0
    n_skipped = 0       # too short / all-N (no valid k-mers)
    n_lowqual = 0       # failed quality gate (high-N / short) -> dropped, NOT promoted
    t0 = time.time()
    scan_ops = 0

    # --dump-promoted diagnostics: per-genome composition so we can check WHETHER
    # promotion tracks real biological novelty vs. junk (high-N / short / artifact).
    dump = None
    if args.dump_promoted:
        dump = open(args.dump_promoted, 'w')
        dump.write("decision\tlen\tn_frac\tnovelty\theader\n")

    def seq_stats(seq_bytes):
        L = len(seq_bytes)
        if L == 0:
            return 0, 1.0
        arr = np.frombuffer(seq_bytes, dtype=np.uint8)
        acgt = np.isin(arr, np.frombuffer(b"ACGT", dtype=np.uint8)).sum()
        return L, 1.0 - acgt / L   # (length, non-ACGT fraction ~ N/ambiguity/gap)

    for hdr, seq in stream_fasta(args.input):
        n += 1
        # QUALITY GATE (§2.8 fix): reject low-quality genomes BEFORE novelty scoring,
        # so missing-data artifacts can't masquerade as novelty and get promoted.
        L_q, nf_q = seq_stats(seq)
        if L_q < args.min_len or nf_q > args.max_n_frac:
            n_lowqual += 1
            if dump and n % 50 == 0:
                dump.write(f"LOWQUAL\t{L_q}\t{nf_q:.4f}\t0.0000\t{hdr.decode(errors='replace')}\n")
            continue
        sk = canonical_kmer_hashes(seq, args.k, salts)
        if sk is None:
            n_skipped += 1
            continue
        if not promoted:
            promoted.append(sk); promoted_counts.append(1); n_promoted += 1
            if dump:
                L, nf = seq_stats(seq); dump.write(f"PROMOTE\t{L}\t{nf:.4f}\t1.0000\t{hdr.decode(errors='replace')}\n")
        else:
            # naive O(P) nearest-neighbour scan (worst case; LSH later)
            best = 0.0; best_i = -1
            for i, ps in enumerate(promoted):
                js = jaccard_est(sk, ps)
                if js > best:
                    best = js; best_i = i
            scan_ops += len(promoted)
            novelty = 1.0 - best
            if novelty >= args.threshold:
                promoted.append(sk); promoted_counts.append(1); n_promoted += 1
                if dump:
                    L, nf = seq_stats(seq); dump.write(f"PROMOTE\t{L}\t{nf:.4f}\t{novelty:.4f}\t{hdr.decode(errors='replace')}\n")
            else:
                promoted_counts[best_i] += 1
                # sample dropped genomes (every 50th) for the comparison baseline
                if dump and n % 50 == 0:
                    L, nf = seq_stats(seq); dump.write(f"DROP\t{L}\t{nf:.4f}\t{novelty:.4f}\t{hdr.decode(errors='replace')}\n")
        if args.limit and n >= args.limit:
            break
        if n % args.report_every == 0:
            dt = time.time() - t0
            print(f"[{n:>8}] promoted={n_promoted:>7} ({100*n_promoted/n:5.1f}%) "
                  f"rate={n/dt:8.1f} seq/s  P={len(promoted)}  "
                  f"scan_ops={scan_ops/1e6:.1f}M", flush=True)

    if dump:
        dump.close()
    dt = time.time() - t0
    print("\n=== L1 TRIGGER PROTOTYPE RESULT ===")
    print(f"input             : {args.input}")
    print(f"k={args.k} sketch_m={args.sketch} threshold={args.threshold}")
    print(f"sequences seen    : {n}")
    print(f"quality gate      : max_n_frac={args.max_n_frac} min_len={args.min_len}")
    print(f"LOW-QUALITY drop  : {n_lowqual}  ({100*n_lowqual/max(1,n):.2f}%)  [high-N / short]")
    print(f"skipped (no kmers): {n_skipped}")
    print(f"PROMOTED (to L2)  : {n_promoted}  ({100*n_promoted/max(1,n):.2f}%)")
    print(f"DROPPED (dup)     : {n - n_promoted - n_skipped - n_lowqual}  "
          f"({100*(n-n_promoted-n_skipped-n_lowqual)/max(1,n):.2f}%)")
    print(f"REJECTION FACTOR  : {n/max(1,n_promoted):.1f}x  (L1 shields L2 by this)")
    print(f"throughput        : {n/dt:.1f} seq/s (single core)")
    print(f"wall              : {dt:.1f}s")
    print(f"naive scan_ops    : {scan_ops/1e6:.1f}M pairwise-sketch cmps "
          f"(LSH would cut this to ~O(N))")
    # project to 17.6M at this single-core rate + naive scan
    rate = n/dt
    print(f"PROJECT 17.6M @ this rate (single core, naive scan): "
          f"{17.6e6/rate/3600:.2f} h  ({17.6e6/rate/60:.0f} min)")
    print(f"  (naive scan cost grows with P; LSH index removes the O(P) term)")

if __name__ == '__main__':
    main()
