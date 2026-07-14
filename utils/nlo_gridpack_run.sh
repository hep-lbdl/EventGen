#!/bin/bash
# Per-chunk driver packed into an NLO gridpack tarball (see
# tasks.py:_create_nlo_gridpack). Usage: run.sh NEVENTS SEED
set -e
NEVENTS=$1
SEED=$2
OUTDIR="$PWD"

# Update nevents and iseed in run_card.dat. The seed must go into the
# run_card: a non-zero baked-in iseed (from the warmup) overrides
# SubProcesses/randinit, which would give every chunk identical events.
python3 - "$NEVENTS" "$SEED" "Cards/run_card.dat" <<'PYEOF'
import sys, re
nevents, seed, path = sys.argv[1:]
with open(path) as f:
    card = f.read()
card = re.sub(r'\b\d+\b(\s+= nevents)', lambda m: nevents + m.group(1), card)
card = re.sub(r'\b\d+\b(\s+= iseed)', lambda m: seed + m.group(1), card)
with open(path, 'w') as f:
    f.write(card)
PYEOF

# Pin MG5 parallelism to the Slurm allocation; without this it auto-detects
# all node cores (256 on Perlmutter) and OOMs the job's cgroup.
echo "run_mode = 2" >> Cards/amcatnlo_configuration.txt
echo "nb_core = ${SLURM_CPUS_PER_TASK:-4}" >> Cards/amcatnlo_configuration.txt

# Events/ was excluded from the tarball to keep it small; recreate it.
mkdir -p Events
# -f: no questions; -o/-x: reuse the baked integration grids and compiled code.
./bin/generate_events -f -o -x

cp "Events/run_01/events.lhe.gz" "$OUTDIR/events.lhe.gz"
