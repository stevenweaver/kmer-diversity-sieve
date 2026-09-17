#!/usr/bin/env bash
# e2e.sh -- end-to-end + backend-invariant tests for kmer-diversity-sieve.
#
# Builds all available backends and asserts the properties we care about on a
# deterministic synthetic fixture (no real data):
#   1. every backend runs and emits a non-empty promoted subset
#   2. the frequency filter FIRES on the single-process pass (>=1 removed)
#   3. serial is the DRIFT-FREE reference: its promoted set is stable run-to-run
#   4. threads(N) == mpi(N) BIT-FOR-BIT (identical promoted-header set) -- the
#      core cross-backend invariant, checked only when both are buildable
#   5. serial admit/reject counts are deterministic (golden regression)
#
# MPI is optional: if mpicc/mpirun are absent, the MPI lane is SKIPPED (not failed)
# so the suite still runs on a laptop with just a C compiler.
#
# Usage: tests/e2e.sh   (run from the repo root, or anywhere -- it cd's to repo root)
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
SRC="$ROOT/scripts/l1_pipeline.c"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# arch-appropriate tuning flag (clang on arm64 rejects -march=native)
case "$(uname -m)" in
  arm64|aarch64) ARCH="-mcpu=native" ;;
  *)             ARCH="-march=native" ;;
esac
# -Wno-misleading-indentation: the source is intentionally dense (multiple
# statements per line); the warning is pure style and not what these tests check.
CFLAGS="-O2 $ARCH -D_FILE_OFFSET_BITS=64 -Wall -Wno-misleading-indentation"

FAIL=0
ok()   { echo "  PASS: $*"; }
bad()  { echo "  FAIL: $*"; FAIL=1; }
skip() { echo "  SKIP: $*"; }

RUNOPTS="--min-len 200 --max-n-frac 0.5 --warmup 200"

echo "== generating deterministic fixture =="
python3 "$HERE/gen_fixture.py" 600 "$WORK/fixture.fasta"
FASTA="$WORK/fixture.fasta"

# --- build serial + threads (always) ---
echo "== building backends =="
cc $CFLAGS -o "$WORK/serial"  "$SRC" -lm            || { echo "serial build failed"; exit 1; }
cc $CFLAGS -pthread -DUSE_THREADS -o "$WORK/threads" "$SRC" -lm || { echo "threads build failed"; exit 1; }
HAVE_MPI=0
if command -v mpicc >/dev/null 2>&1 && command -v mpirun >/dev/null 2>&1; then
  if mpicc $CFLAGS -DUSE_MPI -o "$WORK/mpi" "$SRC" -lm 2>/dev/null; then HAVE_MPI=1; fi
fi

promoted_set() { cat "$1".rank*.fasta 2>/dev/null | grep '^>' | sort; }
counter() { grep -Eo "$2"'[[:space:]]*:[[:space:]]*[0-9]+' "$1" | grep -Eo '[0-9]+$' | head -1; }

echo "== 1. each backend emits a non-empty promoted subset =="
"$WORK/serial"  --input "$FASTA" $RUNOPTS --emit-promoted "$WORK/ser" >"$WORK/ser.log" 2>&1
"$WORK/threads" --input "$FASTA" $RUNOPTS --threads 4 --emit-promoted "$WORK/thr" >"$WORK/thr.log" 2>&1
ns=$(promoted_set "$WORK/ser" | wc -l | tr -d ' ')
nt=$(promoted_set "$WORK/thr" | wc -l | tr -d ' ')
[ "$ns" -ge 1 ] && ok "serial promoted $ns"  || bad "serial promoted 0"
[ "$nt" -ge 1 ] && ok "threads promoted $nt" || bad "threads promoted 0"

echo "== 2. frequency filter fires on the single-process (serial) pass =="
rej=$(counter "$WORK/ser.log" 'REJECT \(error\)')
[ "${rej:-0}" -ge 1 ] && ok "serial filter removed ${rej}" || bad "serial filter removed 0 (should flag outliers)"

echo "== 3. serial is drift-free: same fixture -> same promoted set on a re-run =="
"$WORK/serial" --input "$FASTA" $RUNOPTS --emit-promoted "$WORK/ser2" >/dev/null 2>&1
if diff -q <(promoted_set "$WORK/ser") <(promoted_set "$WORK/ser2") >/dev/null; then
  ok "serial promoted set is reproducible"
else
  bad "serial promoted set differs across identical runs"
fi

echo "== 4. threads(4) == mpi(4) bit-for-bit =="
if [ "$HAVE_MPI" = 1 ]; then
  mpirun --oversubscribe -np 4 "$WORK/mpi" --input "$FASTA" $RUNOPTS --emit-promoted "$WORK/mpi_o" >/dev/null 2>&1
  if diff -q <(promoted_set "$WORK/thr") <(promoted_set "$WORK/mpi_o") >/dev/null; then
    ok "threads(4) and mpi(4) produce an identical promoted-header set"
  else
    bad "threads(4) != mpi(4) -- backend divergence"
    diff <(promoted_set "$WORK/thr") <(promoted_set "$WORK/mpi_o") | head
  fi
else
  skip "MPI toolchain not found -- threads==mpi invariant not checked"
fi

echo "== 5. serial admit/reject is deterministic (golden regression) =="
adm=$(counter "$WORK/ser.log" 'admitted')
# Golden values are the measured, deterministic serial output for this exact
# fixture (N=600, seed 1234) at warmup 200. If you change the fixture or the
# admission math, re-measure and update these.
GOLD_ADM=594 GOLD_REJ=6
[ "${adm:-0}" = "$GOLD_ADM" ] && ok "serial admitted == $GOLD_ADM (golden)" \
                               || bad "serial admitted ${adm:-?} != golden $GOLD_ADM"
[ "${rej:-0}" = "$GOLD_REJ" ] && ok "serial removed == $GOLD_REJ (golden)" \
                               || bad "serial removed ${rej:-?} != golden $GOLD_REJ"

echo
if [ "$FAIL" = 0 ]; then echo "e2e: ALL CHECKS PASSED"; else echo "e2e: FAILURES ABOVE"; fi
exit $FAIL
