# kmer-diversity-sieve

[![CI](https://github.com/stevenweaver/kmer-diversity-sieve/actions/workflows/ci.yml/badge.svg)](https://github.com/stevenweaver/kmer-diversity-sieve/actions/workflows/ci.yml)

An **alignment-free, streaming k-mer sieve** for archive-scale FASTA. In a
single **C + OpenMPI** pass it reduces tens of millions of nucleotide sequences
to a small, quality-controlled, **diversity-preserving** subset — dropping
partial records, sequencing-error records (Count-Min-sketch admission), and
near-duplicates (MinHash/LSH novelty) — so an expensive downstream analysis only
ever processes an informative subset instead of the whole archive.

The core is **organism-agnostic**: it works on k-mers over arbitrary nucleotide
FASTA and assumes no particular reference, genome length, or taxon — those are
just tunable flags. Its first validated application was a large SARS-CoV-2
consensus-genome archive (see the
[case study](#case-study-sars-cov-2-archive)), but any collection of sequences
works.

> **On the design (and the "trigger"/"admission"/"novelty" vocabulary).** The
> staged reduce-cheapest-first design was **inspired by a particle-physics L1
> data-acquisition (DAQ) trigger** — a cheap hardware filter that discards
> uninteresting events before the costly analysis runs. That's an analogy for
> the *shape* of the pipeline, not a claim to be DAQ hardware or software: this
> is read-only k-mer statistics over a *static* archive of sequence files.
> Nothing runs live.

---

## Contents

- [How it works: the three-stage funnel](#how-it-works-the-three-stage-funnel)
- [Build & run](#build--run)
- [CLI reference](#cli-reference)
- [Running on a single node — read this first](#running-on-a-single-node--read-this-first)
- [Output & the L2 layer](#output--the-l2-layer)
- [Internals](#internals)
- [Case study: SARS-CoV-2 archive](#case-study-sars-cov-2-archive)
- [Repository layout](#repository-layout)
- [Licensing / data](#licensing--data)

---

## How it works: the three-stage funnel

Every sequence streams through, in this order (cheapest reject first):

| Stage | Question | Mechanism | Drops |
|---|---|---|---|
| **1. Completeness gate** | Complete enough? | length ≥ `--min-len` **and** N-fraction ≤ `--max-n-frac` | partial / gappy records |
| **2. Admission** | Is the variation *real*, not error? | Count-Min-Sketch k-mer frequency table + singleton-outlier test (`--outlier-sd`); too many singleton k-mers ⇒ error record | isolated-error records |
| **3. Novelty** | Is it *new*, not redundant? | MinHash sketch (m=`--sketch`) + LSH band index; promote iff `1 − maxJaccard ≥ --threshold` vs the already-promoted set | near-duplicate / redundant records |

k-mers are extracted **once** per record and reused across all three stages —
no double parse.

**Optional partitioned novelty.** The LSH band key can mix in a partition
bucket parsed from the record header (in the SARS-CoV-2 application: the
sample's `year*12+month`; undated → shared bucket `-1`), so a record is only
compared against promoted records **in the same bucket**. This keeps
under-represented strata (e.g. sparse months) from being starved by dense ones,
flattening whatever distribution a downstream temporal analysis depends on.

The pipeline is **np-invariant**: a two-phase merge (local sketches → MPI
reduce → global decision) makes admit/reject/promote reproducible regardless of
rank count.

## Build & run

The pipeline is a **single C source** (`scripts/l1_pipeline.c`) that builds into
**three interchangeable backends**, chosen at compile time. Only the **mpi**
backend needs an MPI toolchain; **serial** and **threads** compile with nothing
but a stock C compiler (`cc` — clang or gcc) and a math library.

```sh
make serial    # cc -O3, no MPI, no threads     -> l1_pipeline_serial
make threads   # cc -pthread (NO OpenMP)         -> l1_pipeline_threads
make mpi       # mpicc -DUSE_MPI                 -> l1_pipeline
make all       # build all three
```

Then run whichever you built:

```sh
./l1_pipeline_serial  --input FILE.fasta [--emit-promoted PREFIX] [opts]
./l1_pipeline_threads --input FILE.fasta --threads N [--emit-promoted PREFIX] [opts]
mpirun -np N ./l1_pipeline --input FILE.fasta [--emit-promoted PREFIX] [opts]
```

Smoke test (serial, capped input — zero dependencies):

```sh
./l1_pipeline_serial --input FILE.fasta --limit 5000
```

### Build variants

One source, one algorithm, three ways to shard the input file. **The admission
and novelty math is identical in all three** — they differ only in *who owns
which byte-range of the input and how the per-shard promoted sets are merged.*

| Backend | Build | Parallelism | When to use |
|---|---|---|---|
| **serial** | `make serial` (`cc -O3`, no MPI/threads) | none — one process reads the whole file | **zero dependencies**; laptop, CI, smoke tests. The **drift-free reference oracle** (see below). |
| **threads** | `make threads` (`cc -pthread`, **no OpenMP**) | `--threads N` pthreads, each owns a byte-range, merged in-process | **one big multi-core box.** Only needs `-pthread` — no MPI toolchain. |
| **mpi** | `make mpi` (`mpicc -DUSE_MPI`) | `mpirun -np N` ranks, byte-range sharded, MPI-gathered merge | **multi-node cluster.** Scale across nodes (see the case study). |

- **serial is the drift-free reference.** It uses **one** shard = the whole file
  in genome order: a single Count-Min sketch and a single warm-up ramp that
  every record contributes to, so the phase-2 cross-shard merge is a genuine
  **no-op** — there is no cross-shard boundary and therefore **zero merge
  drift**. Use it to define ground-truth admit/reject/promote decisions.
- **threads** gives each pthread its own Count-Min sketch + promoted set +
  warm-up stats over its own byte-range (**no shared/atomic state** — sharing one
  sketch would change the numerics), then runs the **same** greedy,
  partition-aware phase-2 merge the MPI rank-0 does. With `N` threads it
  reproduces `mpirun -np N` **bit-for-bit** on the same input.
- `--threads N` selects the thread count for the **threads** backend and
  defaults to the number of online CPUs (`sysconf(_SC_NPROCESSORS_ONLN)`). It is
  accepted-and-ignored by the serial and mpi backends (which take their width
  from the process count / `-np`).
- **mpi** is the original backend and is **unchanged** — same byte-range
  sharding, same two-phase gather-and-merge, same output filenames.

> **Arch note.** `-march=native` is x86-only; clang on Apple Silicon rejects it.
> The `Makefile` detects the architecture and uses `-mcpu=native` on
> `arm64`/`aarch64` instead. If you build by hand on Apple Silicon, use plain
> `-O3` (or `-mcpu=native`) rather than `-march=native`.

## CLI reference

| Flag | Default | Meaning |
|---|---|---|
| `--input` | *(required)* | archive FASTA (byte-range sharded across ranks) |
| `--emit-promoted` | *(off)* | write promoted records → `PREFIX.rank*.fasta` |
| `--min-len` | 27000 | completeness gate: minimum length (nt) |
| `--max-n-frac` | 0.05 | completeness gate: maximum N fraction |
| `--k` | 15 | k-mer size |
| `--sketch` | 64 | MinHash sketch size (m); LSH = 16 bands × (m/16) rows |
| `--threshold` | 0.15 | novelty promotion threshold (τ) |
| `--err-rate` | 1e-4 | admission Count-Min error rate |
| `--p` | 0.9 | admission binomial cutoff p |
| `--outlier-sd` | 5.0 | admission singleton-outlier SD threshold |
| `--warmup` | 2000 | records to warm the Count-Min sketch before deciding |
| `--limit` | 0 (all) | cap records (smoke tests) |
| `--threads` | online CPUs | **threads backend only:** number of pthreads (each owns a byte-range) |

The `27000` / `0.05` defaults are sized for coronavirus-length genomes —
**override `--min-len` and `--max-n-frac` for any other organism.**

## Running on a single node — read this first

**On a single node, prefer the `threads` backend over `mpirun` on localhost.**
Both shard the same file the same way and produce identical decisions, but the
threaded build needs only `-pthread` (no MPI toolchain, no launcher, no per-rank
process overhead), shares one address space, and is the simpler thing to reason
about on one box:

```sh
make threads
./l1_pipeline_threads --input FILE.fasta --threads 16 [--emit-promoted PREFIX]
```

Reserve **mpi** for when you are actually spanning **multiple nodes** — that is
the only regime where `mpirun` earns its overhead.

The single biggest lesson from building this: **at archive scale you are
bound by how fast one file can be read, not by CPU.** Each shard (pthread or MPI
rank) `fseek`s to its own contiguous byte-range of the input and streams from
there, so all shards hammer the *same* file on the *same* storage.

- **Admission throughput scales to ~16–32 shards per node, then plateaus** —
  that plateau is the shared single-file read bandwidth, not a CPU limit.
  Adding more shards *on the same node* (more `--threads`, or more ranks) buys
  nothing past that point and can make things worse (I/O contention + memory).
- **Go wider with more *nodes*, not more shards per node.** Novelty (the
  compute-bound stage) scales near-linearly; ingestion does not. On one node,
  expect to saturate at that per-node plateau — a single node is fine for
  millions of records, but do not expect linear speedup from `--threads 128`
  (or `-np 128`) on one box.
- **Memory:** the Count-Min sketch is **64 MB/shard**. At 128 shards/node that is
  ~8 GB just for sketches — an earlier 256 MB/shard sketch OOM-killed nodes at
  high shard counts. Size your `shards × 64 MB` against node RAM.
- **Fastest storage wins.** Put the input on the fastest local/parallel
  filesystem you have; a single file on slow shared storage is the worst case.
- **One shard = no parallelism.** `./l1_pipeline_serial` (or `--threads 1`, or
  `-np 1`) reads the whole file sequentially. Serial is the right mode for smoke
  tests (`--limit`) and as the drift-free reference oracle.

Rule of thumb on one box: `./l1_pipeline_threads --threads <16–32>`. To go
bigger, `mpirun -np <16–32 per node> --bind-to core` scaled out across as many
nodes as you can give it.

## Output & the L2 layer

`--emit-promoted P` writes one FASTA per rank, `P.rank0.fasta`, `P.rank1.fasta`,
… Concatenate them to get the promoted subset:

```sh
cat P.rank*.fasta > promoted.fasta
```

That `promoted.fasta` is the handoff to **L2** — whatever analysis you want to
run on the reduced set. L1 does not prescribe L2; it just guarantees L2 gets a
clean, small, diversity-preserving input. Typical L2 layers:

- **Per-gene / per-region extraction.** Map the promoted genomes to a reference
  with `minimap2 -a --secondary=no -x asm20`, then carve out each region of
  interest from the CIGAR (frame-preserving for codon analyses). This is the
  usual first step before any alignment-based method.
- **Selection / dN/dS analysis.** e.g. HyPhy MEME/FEL/SLAC on the extracted
  codon alignments, or a neural surrogate as a fast pre-screen.
- **Molecular-clock dating / phylodynamics.** Root-to-tip regression or a
  dated-tree method on the promoted, diversity-flattened set.
- **Anything downstream that is too expensive to run on the full archive.** The
  whole point of L1 is that L2 can cost seconds-per-record because it only ever
  sees thousands, not tens of millions.

> The specific L2 tooling for the SARS-CoV-2 application (per-gene minimap2
> extraction, viral gene lists, UShER tree checks, and the selection/dating
> handoff) is organism-specific and lives outside this general-purpose repo.

## Internals

- **File read:** `fseeko` byte-range sharding — each rank reads only its own
  contiguous slice (no redundant reads), snapping slice boundaries to the next
  `>` header so no record is split.
- **Count-Min sketch:** 2²² × 4 rows × 4 B = **64 MB/rank**.
- **np-invariance:** two-phase merge (local shard-decisions → `Gatherv`
  promoted sketches → rank-0 greedy global merge), so results do not depend on
  rank count.
- **Reference oracle:** the C trigger is validated to *exactly* match the Python
  prototype (`scripts/l1_trigger_prototype.py` /
  `scripts/l1_trigger_compressor.py`) — identical admit/reject decisions.

## Case study: SARS-CoV-2 archive

The trigger was validated at full scale on a ~489 GiB, ~17.6M-genome SARS-CoV-2
consensus-genome archive (128 MPI ranks = 8 nodes × 16, `--bind-to core`):

| Metric | Value |
|---|---|
| Input | 489 GiB, ~17.6M sequences |
| Completeness-gate drop | 1,589,433 (9.02%) partial/gappy |
| Admission reject (error) | 1,161 (0.007%) |
| Admitted | 16,026,932 (90.97%) |
| **Promoted** | **1,443** (0.0082% of input) |
| **Overall reduction** | **12,209× (input/promoted)** |
| Throughput | 14,688 seq/s aggregate |
| **Wall time** | **1,199 s ≈ 20 min** |

For scale: an L2 costing ~1 s/record would need ~36 node-days to touch all
17.6M; L1 clears the archive ~600× faster and hands L2 ~1.4k records instead.
The promoted set was cross-checked against a public UShER global tree.

## Repository layout

- `scripts/l1_pipeline.c` — production L1 trigger (C + OpenMPI). **Primary binary.**
- `scripts/l1_novelty.c`, `scripts/l1_trigger.c` — earlier / component sources.
- `scripts/l1_trigger_prototype.py`, `scripts/l1_trigger_compressor.py` —
  Python reference oracle the C trigger is validated against.
- `scripts/validate_order_invariance.py`, `scripts/profile_trigger.py` —
  validation / profiling helpers.
- `scripts/bench_scaling.sbatch`, `scripts/bench_novelty_scaling.sbatch` —
  SLURM scaling benchmarks.

## Licensing / data

The code is released under the **MIT License** — see [`LICENSE`](LICENSE).

This repository is **code only** — no sequences, alignments, promoted subsets,
per-record tables, or model weights are committed, and the `.gitignore` blocks
those file types as a guardrail. The MIT license covers this code; it does
**not** cover any data you run it on. If you point the sieve at your own data,
mind whatever license or data-use agreement governs that data.
