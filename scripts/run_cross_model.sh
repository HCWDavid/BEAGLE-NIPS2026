#!/bin/bash
# =============================================================================
# Cross-model BEAGLE generalization sweep (Table 3 of the paper).
#
# Runs BEAGLE on particle_simulator with multiple LLM backbones to show the
# framework is model-agnostic. Output folders are named to match the MODELS_X
# config in evaluation/compute_comprehensive_metrics.py so the table populates
# automatically.
#
# Run order is OpenAI -> Anthropic -> Gemini, so the heavy Gemini calls happen
# AFTER the current Gemini cross-task sweep (run_new_tasks.sh) finishes — no
# rate-limit collision.
#
# Usage:
#   ./run_cross_model.sh                  # all non-Gemini first, then Gemini
#   ./run_cross_model.sh --openai-only    # just GPT-4.1-mini + GPT-5-mini
#   ./run_cross_model.sh --anthropic-only # just Claude Haiku 4.5
#   ./run_cross_model.sh --gemini-only    # just Gemini 3 Flash (preview)
#   ./run_cross_model.sh --no-gemini      # skip Gemini (run while sweep is alive)
#   ./run_cross_model.sh --skip-judge     # sims only; judge later via Phase 2
# =============================================================================

set -u
cd "$(dirname "$0")/.."

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
N_PER_LEVEL=25                      # 25 high + 25 low = N=50 (cross-model generalizability)
MAX_STEPS=30
PROBLEM="particle_simulator"        # canonical anchor — has deepest legacy data
JUDGE_MODEL="google-gla:gemini-2.5-pro"
SAMPLE_SIZE=50
DATE=$(date +%Y-%m-%d)

PY="${PYTHON:-python}"

# Each entry: <folder_tag>|<pydantic_ai_model_string>|<provider_kind>
# folder_tag must match the path in MODELS_X (compute_comprehensive_metrics.py)
MODELS_NON_GEMINI=(
    "gpt4_1mini|openai:gpt-4.1-mini|openai"
    # "gpt5mini|openai:gpt-5-mini|openai"  # disabled: thinking model, ~113s/step in smoke test
    "claudehaiku45|anthropic:claude-haiku-4-5|anthropic"
)

MODELS_GEMINI=(
    "gemini3flash|google-vertex:gemini-3-flash-preview|gemini-vertex-global"
)

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
SKIP_JUDGE=0
ONLY=""

for arg in "$@"; do
    case "$arg" in
        --skip-judge)     SKIP_JUDGE=1 ;;
        --openai-only)    ONLY="openai" ;;
        --anthropic-only) ONLY="anthropic" ;;
        --gemini-only)    ONLY="gemini" ;;
        --no-gemini)      ONLY="non-gemini" ;;
        --*)              echo "Unknown flag: $arg" >&2; exit 1 ;;
    esac
done

# Build the run list based on filter.
RUN_LIST=()
case "$ONLY" in
    openai)
        for m in "${MODELS_NON_GEMINI[@]}"; do
            [[ "$m" == *"|openai" ]] && RUN_LIST+=("$m")
        done
        ;;
    anthropic)
        for m in "${MODELS_NON_GEMINI[@]}"; do
            [[ "$m" == *"|anthropic" ]] && RUN_LIST+=("$m")
        done
        ;;
    gemini)
        RUN_LIST=("${MODELS_GEMINI[@]}")
        ;;
    non-gemini)
        RUN_LIST=("${MODELS_NON_GEMINI[@]}")
        ;;
    *)
        # Default: non-Gemini first, then Gemini
        RUN_LIST=("${MODELS_NON_GEMINI[@]}" "${MODELS_GEMINI[@]}")
        ;;
esac

# -----------------------------------------------------------------------------
# Load env
# -----------------------------------------------------------------------------
[ -f .env ] && set -a && . ./.env && set +a

echo "============================================="
echo "BEAGLE Cross-Model Sweep (Table 3)"
echo "============================================="
echo "Date:        $DATE"
echo "Problem:     $PROBLEM"
echo "N per level: $N_PER_LEVEL  (total per model: $((N_PER_LEVEL * 2)))"
echo "Skip judge:  $SKIP_JUDGE"
echo "Models:      ${#RUN_LIST[@]} in queue"
for m in "${RUN_LIST[@]}"; do
    echo "  - ${m%%|*}"
done
echo

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
out_dir() {
    # out_dir <folder_tag>
    echo "results/${DATE}_beagle_$1_${PROBLEM}"
}

already_done() {
    [ -f "$1/summary.json" ]
}

run_judge() {
    local out=$1
    [ "$SKIP_JUDGE" -eq 1 ] && { echo "  (skipping LLM judge)"; return; }
    if [ -f "$out/llm_judgments.json" ]; then
        local judged
        judged=$($PY -c "import json; print(len(json.load(open('$out/llm_judgments.json'))))" 2>/dev/null || echo 0)
        if [ "$judged" -ge "$SAMPLE_SIZE" ]; then
            echo "  (already judged: $judged judgments)"
            return
        fi
    fi
    echo "  -> LLM judge ($SAMPLE_SIZE samples)"
    $PY evaluation/evaluator.py \
        --folder "$out" \
        --llm-judge --llm-model "$JUDGE_MODEL" \
        --sample-size "$SAMPLE_SIZE" --sample-strategy balanced \
        --export "$out/evaluation.json"
}

run_one_model() {
    local tag=$1 model=$2 kind=$3
    local out
    out=$(out_dir "$tag")

    echo
    echo "============================================="
    echo ">>> BEAGLE on $PROBLEM with $model"
    echo "    output: $out"
    echo "    kind:   $kind"
    echo "============================================="

    if already_done "$out"; then
        echo "  (skip: summary.json already exists)"
    else
        # Per-provider env setup. The llm_client patch only pops GOOGLE_API_KEY
        # for google-vertex models, so OpenAI/Anthropic just need their key
        # already in env (.env handles that).
        local extra_env=""
        if [[ "$kind" == "gemini-vertex-global" ]]; then
            # Gemini 3.x is only available in the 'global' Vertex region.
            extra_env="GOOGLE_CLOUD_LOCATION=global"
        fi

        bash -c "$extra_env $PY evaluation/run_batch_simulations.py \
            --n-per-level $N_PER_LEVEL --levels both \
            --problem $PROBLEM \
            --model '$model' \
            --max-steps $MAX_STEPS \
            --output-dir '$out'"
    fi

    run_judge "$out"
}

# -----------------------------------------------------------------------------
# Execute
# -----------------------------------------------------------------------------
for entry in "${RUN_LIST[@]}"; do
    IFS='|' read -r tag model kind <<< "$entry"
    run_one_model "$tag" "$model" "$kind"
done

echo
echo "============================================="
echo "Cross-model sweep complete."
echo "============================================="
echo
echo "Now generate the table:"
echo "  $PY evaluation/compute_comprehensive_metrics.py --table cross_model"
