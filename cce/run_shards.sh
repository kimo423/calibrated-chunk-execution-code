#!/bin/bash
# Launch one CCE benchmark sharded over plants.  Usage:
#   ./cce/run_shards.sh <replay|tracking> <uid> <nshards> <tag|-> [extra args...]
# Outputs: data/cce/parts/<bench><tag>_<uid>_<shard>.json
set -eu
ROOT=/opt/cce
cd "$ROOT"
BENCH=$1; ROBOT=$2; NSH=$3; TAG=$4; shift 4
[ "$TAG" = "-" ] && TAG=""
NAME=${BENCH}${TAG}
mkdir -p data/cce/parts data/logs
rm -f data/cce/parts/${NAME}_${ROBOT}_*.json data/logs/${NAME}_${ROBOT}_*.log data/logs/${NAME}_${ROBOT}_*.exit

if [ "$BENCH" = "replay" ] && [ ! -f "data/cce/intent_${ROBOT}.npz" ]; then
  ./cce/env.sh cce/bench_replay.py --uid "$ROBOT" --prep-only \
      --out data/cce/parts/prep_${ROBOT}.json > data/logs/prep_${ROBOT}.log 2>&1
fi

i=0
while [ "$i" -lt "$NSH" ]; do
  LOG=data/logs/${NAME}_${ROBOT}_${i}.log
  OUT=data/cce/parts/${NAME}_${ROBOT}_${i}.json
  nohup bash -c "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./cce/env.sh cce/bench_${BENCH}.py --uid $ROBOT --shard $i --nshards $NSH --out $OUT $* > $LOG 2>&1; echo \$? > data/logs/${NAME}_${ROBOT}_${i}.exit" >/dev/null 2>&1 &
  i=$((i+1))
done
echo "launched $NSH shards for $NAME/$ROBOT"
