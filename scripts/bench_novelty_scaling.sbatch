#!/bin/bash
#SBATCH --job-name=nov-scaling
#SBATCH --partition=YOUR_PARTITION
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=nov_scaling_%j.out

set -euo pipefail
# Point these at your build + a benchmark FASTA of your own.
BIN=${BIN:-./l1_novelty}
FA=${FA:?set FA=/path/to/benchmark.fasta}

echo "== $(date) host=$(hostname) cores=$(nproc) =="
echo "NOVELTY scaling sweep on 100k, two-phase, exclusive node"
for NP in 1 4 8 16 32 64; do
  echo "---- np=$NP ----"
  mpirun -np $NP --bind-to core $BIN --input $FA 2>&1 \
    | grep -E 'PROMOTED|REJECTION|throughput|candidates'
done
echo "== DONE $(date) =="
