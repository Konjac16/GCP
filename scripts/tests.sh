#!/usr/bin/env bash
set -euo pipefail

BIN_SOLVER=${1:? "need solver path, e.g., bin/gcp"}
BIN_VALID=${2:? "need validator path, e.g., bin/validator"}
CASELIST=${3:-"tests/cases.list"}

mkdir -p logs
mkdir -p logs/result
mkdir -p logs/report

total=0
ok=0
solved=0

while read -r LINE; do
  [[ -z "$LINE" || "$LINE" =~ ^# ]] && continue
  inst=$(echo "$LINE" | awk '{print $1}')
  tl=$(echo "$LINE"   | awk '{print $2}')
  seed=$(echo "$LINE" | awk '{print $3}')
  name=$(echo "$LINE" | awk '{print $4}')
  if [[ -z "$name" ]]; then
    name=$(basename "$inst")
    name="${name%.*}"
  fi

  out="logs/result/${name}.out"
  rep="logs/report/${name}.report"

  echo "==> Running ${name} (TL=${tl}s, seed=${seed})"

  start=$(date +%s)
  rc=0
    "${BIN_SOLVER}" "${tl}" "${seed}" < "${inst}" > "${out}" || rc=$?
  end=$(date +%s)
  elapsed=$((end - start))

  N=""; E=""; Cref=""; colors=""; conflicts=""

  if [[ -f "$out" ]]; then
    "${BIN_VALID}" "${inst}" "${out}" | tee "${rep}" >/dev/null
    first=$(head -n1 "${rep}")
    N=$(echo "$first" | awk '{for(i=1;i<=NF;i++) if($i~"^N="){split($i,a,"="); print a[2]}}')
    E=$(echo "$first" | awk '{for(i=1;i<=NF;i++) if($i~"^E="){split($i,a,"="); print a[2]}}')
    Cref=$(echo "$first" | awk '{for(i=1;i<=NF;i++) if($i~"^Cref="){split($i,a,"="); print a[2]}}')
    colors=$(echo "$first" | awk '{for(i=1;i<=NF;i++) if($i~"^colors="){split($i,a,"="); print a[2]}}')
    conflicts=$(echo "$first" | awk '{for(i=1;i<=NF;i++) if($i~"^conflicts="){split($i,a,"="); print a[2]}}')
  fi
  echo
done < "$CASELIST"

