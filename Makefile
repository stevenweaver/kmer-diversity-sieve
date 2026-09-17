# Makefile for l1_pipeline -- ONE source, THREE build backends.
#
#   make serial   (default) -> l1_pipeline_serial   : cc -O3, no MPI. DRIFT-FREE reference oracle.
#   make threads            -> l1_pipeline_threads   : cc -pthread, N pthreads, in-process merge.
#   make mpi                -> l1_pipeline           : mpicc -DUSE_MPI, current MPI behavior (UNCHANGED).
#   make all                -> all three.
#   make clean              -> remove built binaries.
#
# Dependency-light: serial/threads compile with just stock 'cc' (clang or gcc). No OpenMP.
# arm64-safe: clang on Apple Silicon rejects -march=native, so we detect uname -m and use
# -mcpu=native on aarch64/arm64, -march=native on x86.

CC      ?= cc
MPICC   ?= mpicc
SRC     := scripts/l1_pipeline.c

# --- arch-appropriate tuning flag (arm64 clang rejects -march=native) ---
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),arm64)
  ARCHFLAG := -mcpu=native
else ifeq ($(UNAME_M),aarch64)
  ARCHFLAG := -mcpu=native
else
  ARCHFLAG := -march=native
endif

# common flags for every backend
CFLAGS  := -O3 $(ARCHFLAG) -D_FILE_OFFSET_BITS=64
LDLIBS  := -lm

BIN_SERIAL  := l1_pipeline_serial
BIN_THREADS := l1_pipeline_threads
BIN_MPI     := l1_pipeline

.PHONY: all serial threads mpi clean
.DEFAULT_GOAL := serial

serial: $(BIN_SERIAL)
$(BIN_SERIAL): $(SRC)
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDLIBS)

threads: $(BIN_THREADS)
$(BIN_THREADS): $(SRC)
	$(CC) $(CFLAGS) -DUSE_THREADS -pthread -o $@ $(SRC) $(LDLIBS)

mpi: $(BIN_MPI)
$(BIN_MPI): $(SRC)
	$(MPICC) $(CFLAGS) -DUSE_MPI -o $@ $(SRC) $(LDLIBS)

all: serial threads mpi

clean:
	rm -f $(BIN_SERIAL) $(BIN_THREADS) $(BIN_MPI)
