#!/bin/bash

INPUT=$1
ITER=${2:-500}

if [ -z "$INPUT" ]; then
  echo "Usage: ./runtime_cmp.sh input.png [iterations]"
  exit 1
fi

ALPHA=0.2
OFFSET=0.01
GAMMA=2.2

IMPLEMENTATIONS=(
  mantiuk_cpu
  mantiuk_naive
  mantiuk_shared
)

printf "%-20s %-15s\n" "Implementation" "Time"
echo "-----------------------------------------"

for IMPL in "${IMPLEMENTATIONS[@]}"; do
  if [[ -x "./$IMPL" ]]; then
    RUNTIME=$( (/usr/bin/time -f "%e" ./$IMPL $INPUT ./outputs/out_${IMPL}.png $ALPHA $OFFSET $GAMMA $ITER) 2>&1 1>/dev/null )
    printf "%-20s %-15s\n" "$IMPL" "$RUNTIME s"
  else
    printf "%-20s %-15s\n" "$IMPL" "[binary missing]"
  fi
  sleep 0.2
done