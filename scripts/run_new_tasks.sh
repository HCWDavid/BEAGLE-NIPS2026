#!/bin/bash
# =============================================================================
# Run BEAGLE + all baselines on additional C2STEM topologies (and any
# pure-programming task), to populate the cross-task generalization table.
#
# Methods (in order — BEAGLE runs first):
#   1. BEAGLE (Flash 2.0)
#   2. Vanilla, Vanilla+M, CoT, CoT+M, Few-Shot, FewShot+M
#   3. LLMSS, SimStudent, CoderAgent
#
# For each (method, problem) pair we:
#   - Skip if `summary.json` already exists in the output dir
#   - Run the simulation batch
#   - Run LLM-as-Judge in batch mode
#
# Usage:
#   ./run_new_tasks.sh                              # default: bouncing_ball + inclined_plane
#   ./run_new_tasks.sh bouncing_ball                # one problem
#   ./run_new_tasks.sh bouncing_ball inclined_plane gradient_descent
#
#   ./run_new_tasks.sh --skip-judge bouncing_ball   # skip LLM judge step
#   ./run_new_tasks.sh --beagle-only bouncing_ball  # BEAGLE only, no baselines
#   ./run_new_tasks.sh --baselines-only bouncing_ball
#   ./run_new_tasks.sh --lean bouncing_ball         # BEAGLE + cot_metacog + coderagent only
# =============================================================================

set -e
cd "$(dirname "$0")/.."

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
N_PER_LEVEL=15                                # 15 high + 15 low = N=30 — cross-task table sizing
MAX_STEPS=30
# Both routes use the same work account, just different products:
#   - Student loop -> Vertex AI 2.5 Flash via ADC (paper-quality realism ≈ 2.68)
#   - Judge        -> AI Studio 2.5 Pro via GOOGLE_API_KEY (same work key)
# AI Studio is used for the judge because Vertex's Batch API requires
# gcs_uri/bigquery_uri inputs and silently rejects evaluator.py's InlinedRequest
# format -> falls back to sync. AI Studio accepts inline requests, so we get
# the real 50% batch cost savings + parallel judging.
LLM_MODEL="google-vertex:gemini-2.5-flash"
LLM_TAG="gemini2.5flash"
JUDGE_MODEL="google-gla:gemini-2.5-pro"
SAMPLE_SIZE=30                                # judges every run since N=30
DATE=$(date +%Y-%m-%d)

# Load .env for GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION; the llm_client patch
# will pop GOOGLE_API_KEY when the model uses google-vertex: so ADC is used.
[ -f .env ] && set -a && . ./.env && set +a

# Mamba env wrapper
PY="${PYTHON:-python}"

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
SKIP_JUDGE=0
BEAGLE_ONLY=0
BASELINES_ONLY=0
LEAN=0
PROBLEMS=()

for arg in "$@"; do
    case "$arg" in
        --skip-judge)     SKIP_JUDGE=1 ;;
        --beagle-only)    BEAGLE_ONLY=1 ;;
        --baselines-only) BASELINES_ONLY=1 ;;
        --lean)           LEAN=1 ;;
        --*)              echo "Unknown flag: $arg" >&2; exit 1 ;;
        *)                PROBLEMS+=("$arg") ;;
    esac
done

if [ "${#PROBLEMS[@]}" -eq 0 ]; then
    PROBLEMS=(bouncing_ball inclined_plane)
fi

echo "============================================="
echo "BEAGLE + Baselines Cross-Task Evaluation"
echo "============================================="
echo "Date:        $DATE"
echo "Problems:    ${PROBLEMS[*]}"
echo "Model:       $LLM_MODEL"
echo "N per level: $N_PER_LEVEL  (total per method: $((N_PER_LEVEL * 2)))"
echo "Skip judge:  $SKIP_JUDGE"
echo "BEAGLE only: $BEAGLE_ONLY"
echo "Baselines only: $BASELINES_ONLY"
echo "Lean mode:   $LEAN  (BEAGLE + cot_metacog + coderagent only)"
echo

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
out_dir() {
    # out_dir <method_tag> <problem>
    echo "results/${DATE}_$1_${LLM_TAG}_$2"
}

already_done() {
    # already_done <out_dir>
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
    echo "  -> LLM judge ($SAMPLE_SIZE samples, batch mode)"
    $PY evaluation/evaluator.py \
        --folder "$out" \
        --llm-judge --llm-model "$JUDGE_MODEL" \
        --sample-size "$SAMPLE_SIZE" --sample-strategy balanced \
        --export "$out/evaluation.json"
}

run_beagle() {
    local problem=$1
    local out
    out=$(out_dir "beagle" "$problem")
    echo
    echo ">>> BEAGLE on $problem -> $out"
    if already_done "$out"; then
        echo "  (skip: summary.json already exists)"
    else
        $PY evaluation/run_batch_simulations.py \
            --n-per-level $N_PER_LEVEL --levels both \
            --problem "$problem" \
            --model "$LLM_MODEL" \
            --max-steps $MAX_STEPS \
            --output-dir "$out"
    fi
    run_judge "$out"
}

run_baseline() {
    # run_baseline <baseline_kind> <method_tag> <problem> [--enable-metacog]
    local kind=$1 tag=$2 problem=$3 extra=$4
    local out
    out=$(out_dir "$tag" "$problem")
    echo
    echo ">>> $tag on $problem -> $out"
    if already_done "$out"; then
        echo "  (skip: summary.json already exists)"
    else
        $PY evaluation/run_baseline_batch.py \
            --baseline "$kind" \
            --n-per-level $N_PER_LEVEL --levels both \
            --problem "$problem" \
            --model "$LLM_MODEL" \
            --max-steps $MAX_STEPS \
            --output-dir "$out" \
            $extra
    fi
    run_judge "$out"
}

# -----------------------------------------------------------------------------
# Per-problem sweep
# -----------------------------------------------------------------------------
for problem in "${PROBLEMS[@]}"; do
    echo
    echo "============================================="
    echo "Problem: $problem"
    echo "============================================="

    if [ "$BASELINES_ONLY" -eq 0 ]; then
        run_beagle "$problem"
    fi

    if [ "$BEAGLE_ONLY" -eq 0 ]; then
        if [ "$LEAN" -eq 1 ]; then
            # Lean cross-task: 1 strongest pure-LLM + 1 strongest structured baseline
            run_baseline cot        "cot_metacog" "$problem" --enable-metacog
            run_baseline coderagent "coderagent"  "$problem"
        else
            # Pure LLM baselines
            run_baseline vanilla   "vanilla"          "$problem"
            run_baseline vanilla   "vanilla_metacog"  "$problem" --enable-metacog
            run_baseline cot       "cot"              "$problem"
            run_baseline cot       "cot_metacog"      "$problem" --enable-metacog
            run_baseline fewshot   "fewshot"          "$problem"
            run_baseline fewshot   "fewshot_metacog"  "$problem" --enable-metacog

            # Structured approaches
            run_baseline llmss      "llmss"      "$problem"
            run_baseline simstudent "simstudent" "$problem"
            run_baseline coderagent "coderagent" "$problem"
        fi
    fi
done

echo
echo "============================================="
echo "All requested runs complete."
echo "============================================="
echo
echo "Next: edit CROSS_TASK_TASKS in"
echo "  evaluation/compute_comprehensive_metrics.py (~line 1259)"
echo "to point at the new result folders, then run:"
echo "  $PY evaluation/compute_comprehensive_metrics.py --table cross_task"
