#!/bin/bash
# Per-chunk driver packed into an NLO gridpack tarball (see
# tasks.py:_create_nlo_gridpack). Usage: run.sh NEVENTS SEED
set -e
NEVENTS=$1
SEED=$2
OUTDIR="$PWD"

# Update nevents in run_card.dat
python3 - "$NEVENTS" "Cards/run_card.dat" <<'PYEOF'
import sys, re
nevents, path = sys.argv[1:]
with open(path) as f:
    card = f.read()
card = re.sub(r'\b\d+\b(\s+= nevents)', lambda m: nevents + m.group(1), card)
with open(path, 'w') as f:
    f.write(card)
PYEOF

# Seed via randinit (simpler than editing run_card.dat iseed)
echo "r=${SEED}" > SubProcesses/randinit

# Events/ was excluded from the tarball to keep it small; recreate it.
mkdir -p Events
# -f: no questions, use existing compiled code and integration grids.
./bin/generate_events -f

cp "Events/run_01/events.lhe.gz" "$OUTDIR/events.lhe.gz"
