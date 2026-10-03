#!/bin/bash
# =============================================================================
# Smoke test: run each method × each model with N=1, max-steps=5
# Confirms the whole pipeline works before kicking off the full sweep.
#
# For Vertex AI we explicitly UNSET GOOGLE_API_KEY so the SDK falls back to
# Application Default Credentials (otherwise it tries to use the GLA key as a
# Vertex API key and gets 401).
# =============================================================================

set -u
cd "$(dirname "$0")/.."

PROBLEM="${PROBLEM:-bouncing_ball}"
MAX_STEPS=1
OUT_BASE="/tmp/beagle_smoke"
PY="${PYTHON:-python}"

# Models to test
MODELS=(
    "google-vertex:gemini-2.5-flash"
    "google-gla:gemini-2.5-flash"
)

# Methods: name | runner | extra-flags
# Runner is one of: beagle, baseline:<kind>
METHODS=(
    "beagle           | beagle               |"
    "vanilla          | baseline:vanilla     |"
    "vanilla_metacog  | baseline:vanilla     | --enable-metacog"
    "cot              | baseline:cot         |"
    "cot_metacog      | baseline:cot         | --enable-metacog"
    "fewshot          | baseline:fewshot     |"
    "fewshot_metacog  | baseline:fewshot     | --enable-metacog"
    "llmss            | baseline:llmss       |"
    "simstudent       | baseline:simstudent  |"
    "coderagent       | baseline:coderagent  |"
)

rm -rf "$OUT_BASE"
mkdir -p "$OUT_BASE"

# Load .env so GOOGLE_API_KEY is available for the AI Studio provider.
[ -f .env ] && set -a && . ./.env && set +a

echo "============================================="
echo "Smoke test: every method × every model"
echo "Problem: $PROBLEM | N=1 | max-steps=$MAX_STEPS"
echo "============================================="

declare -a RESULTS=()

run_one() {
    local method=$1 runner=$2 extra=$3 model=$4
    local model_tag=${model//[:.]/_}
    local out="$OUT_BASE/${method}_${model_tag}"
    local label="$method × ${model#google-}"

    rm -rf "$out"

    # For Vertex, unset the AI Studio key so ADC is used.
    local env_setup=""
    if [[ "$model" == google-vertex:* ]]; then
        env_setup="unset GOOGLE_API_KEY;"
    fi

    local cmd
    if [[ "$runner" == beagle ]]; then
        cmd="$PY evaluation/run_batch_simulations.py \
            --n-per-level 1 --levels low \
            --problem $PROBLEM \
            --model $model \
            --max-steps $MAX_STEPS \
            --output-dir $out"
    else
        local kind=${runner#baseline:}
        cmd="$PY evaluation/run_baseline_batch.py \
            --baseline $kind \
            --n-per-level 1 --levels low \
            --problem $PROBLEM \
            --model $model \
            --max-steps $MAX_STEPS \
            --output-dir $out $extra"
    fi

    echo
    echo ">>> $label"
    local t0=$SECONDS
    bash -c "$env_setup $cmd" > "$out.log" 2>&1
    local rc=$?
    local dt=$((SECONDS - t0))

    # Inspect outcome
    local status total_steps="?"
    if [ "$rc" -eq 0 ] && [ -f "$out/summary.json" ]; then
        total_steps=$($PY -c "
import json, glob
files = sorted(glob.glob('$out/runs/*.json'))
if not files:
    print('no_runs')
else:
    r = json.load(open(files[0]))
    print(r.get('total_steps', 0))
" 2>/dev/null || echo "?")
        if [ "$total_steps" = "0" ]; then
            status="ZERO_STEPS"
        elif [ "$total_steps" = "?" ] || [ "$total_steps" = "no_runs" ]; then
            status="NO_RUNS"
        else
            status="OK"
        fi
    else
        status="EXIT_$rc"
    fi

    printf "  %-9s %-3s steps  %ds  log=%s\n" "$status" "$total_steps" "$dt" "$out.log"
    RESULTS+=("$label | $status | steps=$total_steps | ${dt}s")
}

for model in "${MODELS[@]}"; do
    echo
    echo "============================================="
    echo "MODEL: $model"
    echo "============================================="
    for entry in "${METHODS[@]}"; do
        IFS='|' read -r method runner extra <<< "$entry"
        method=$(echo "$method" | xargs)
        runner=$(echo "$runner" | xargs)
        extra=$(echo "$extra" | xargs)
        run_one "$method" "$runner" "$extra" "$model"
    done
done

echo
echo "============================================="
echo "SUMMARY"
echo "============================================="
printf '%s\n' "${RESULTS[@]}"
