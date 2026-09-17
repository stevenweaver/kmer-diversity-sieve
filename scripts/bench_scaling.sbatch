#!/bin/bash
#SBATCH --job-name=l1-scaling
#SBATCH --partition=YOUR_PARTITION
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=l1_scaling_%j.out

set -euo pipefail
# Point these at your build + a benchmark FASTA of your own.
BIN=${BIN:-./l1_trigger}
FA=${FA:?set FA=/path/to/benchmark.fasta}

echo "== $(date) host=$(hostname) cores=$(nproc) =="
echo "exclusive node scaling sweep on 100k, byte-range sharding"

for NP in 1 4 8 16 32 64; do
  echo "---- np=$NP ----"
  mpirun -np $NP --bind-to core $BIN --input $FA --warmup 2000 --report-every 1000000 2>&1 \
    | grep -E 'processed|ADMIT|REJECT|throughput'
done
echo "== DONE $(date) =="
