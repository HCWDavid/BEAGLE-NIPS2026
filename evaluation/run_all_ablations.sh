#!/bin/bash
# =============================================================================
# BEAGLE: Run All Ablation Studies for Component Analysis
# =============================================================================
# 
# This script runs ablation experiments that systematically remove BEAGLE
# components to evaluate their individual contribution to simulation fidelity.
#
# Ablation Conditions:
#   1. FULL BEAGLE (control) - All components enabled
#   2. No BKT - Remove knowledge tracking (epistemic fidelity disabled)
#   3. No Semi-Markov - Replace trained transitions with random sampling
#   4. No Interrupts - Disable assistance and off-topic behaviors
#   5. No Γ_S (Strategist Memory) - Disable thought_buffer, episodic_memories
#   6. No Γ_E (Executor Memory) - Disable agent_notes
#
# Usage:
#   ./run_all_ablations.sh [n_per_level] [model_version] [max_steps] [conditions]
#
# Arguments:
#   n_per_level:   Number of runs per performance level (default: 25)
#   model_version: "2.0" for Gemini 2.0 Flash, "2.5" for Gemini 2.5 Flash (default: 2.0)
#   max_steps:     Maximum steps per simulation (default: 30)
#   conditions:    Comma-separated list of conditions to run, e.g., "1,2,3" or "all" (default: all)
#
# Examples:
#   ./run_all_ablations.sh 25 2.0 30 all     # Run all conditions
#   ./run_all_ablations.sh 25 2.0 40 1,2     # Run only Full BEAGLE and No BKT
#   ./run_all_ablations.sh 25 2.0 30 2,3,4   # Run No BKT, No Semi-Markov, No Interrupts
#   ./run_all_ablations.sh 5 2.0 30 5,6      # Quick test of memory ablations
# =============================================================================

set -e

N_PER_LEVEL=${1:-25}
MODEL_VERSION=${2:-2.0}
MAX_STEPS=${3:-30}
CONDITIONS=${4:-all}
PROBLEM="particle_simulator"

# Set model based on version
if [ "$MODEL_VERSION" == "2.5" ]; then
    MODEL="google-gla:gemini-2.5-flash"
    MODEL_SHORT="gemini2.5flash"
else
    MODEL="google-gla:gemini-2.0-flash"
    MODEL_SHORT="gemini2.0flash"
fi

# Get current date for folder naming
DATE=$(date +%Y-%m-%d)

# Common arguments for all runs
COMMON_ARGS="--n-per-level $N_PER_LEVEL --problem $PROBLEM --max-steps $MAX_STEPS --model $MODEL"

# Helper function to check if a condition should be run
should_run() {
    local cond=$1
    if [ "$CONDITIONS" == "all" ]; then
        return 0  # true
    fi
    # Check if condition is in the comma-separated list
    if [[ ",$CONDITIONS," == *",$cond,"* ]]; then
        return 0  # true
    fi
    return 1  # false
}

echo "=============================================="
echo "BEAGLE ABLATION STUDY RUNNER"
echo "=============================================="
echo "Date: $DATE"
echo "Runs per level: $N_PER_LEVEL"
echo "Max steps: $MAX_STEPS"
echo "Problem: $PROBLEM"
echo "Model: $MODEL"
echo "Conditions: $CONDITIONS"
echo "=============================================="
echo ""
echo "Ablation Conditions:"
echo "  1. Full BEAGLE (control)"
echo "  2. No BKT (--no-bkt)"
echo "  3. No Semi-Markov (--disable-markov)"
echo "  4. No Interrupts (--disable-assistance --disable-offtopic)"
echo "  5. No Γ_S (--disable-memory-strategist)"
echo "  6. No Γ_E (--disable-memory-executor)"
echo "=============================================="
echo ""

# Change to project root
cd "$(dirname "$0")/.."

# =============================================================================
# CONDITION 1: FULL BEAGLE (Control)
# =============================================================================

if should_run 1; then
    echo ""
    echo ">>> [1/6] Running FULL BEAGLE (Control)..."
    python evaluation/run_batch_simulations.py \
        --output-dir "results/${DATE}_ablation_full_${MAX_STEPS}s_${MODEL_SHORT}" \
        --llm-judge \
        $COMMON_ARGS
else
    echo ">>> [1/6] SKIPPED: Full BEAGLE"
fi

# =============================================================================
# CONDITION 2: No BKT (Epistemic Fidelity Disabled)
# Tests: Does knowledge tracking improve simulation fidelity?
# =============================================================================

if should_run 2; then
    echo ""
    echo ">>> [2/6] Running ABLATION: No BKT..."
    python evaluation/run_batch_simulations.py \
        --no-bkt \
        --output-dir "results/${DATE}_ablation_no_bkt_${MAX_STEPS}s_${MODEL_SHORT}" \
        --llm-judge \
        $COMMON_ARGS
else
    echo ">>> [2/6] SKIPPED: No BKT"
fi

# =============================================================================
# CONDITION 3: No Semi-Markov (Random State Sampling)
# Tests: Does trained state transitions improve behavioral fidelity?
# =============================================================================

if should_run 3; then
    echo ""
    echo ">>> [3/6] Running ABLATION: No Semi-Markov..."
    python evaluation/run_batch_simulations.py \
        --disable-markov \
        --output-dir "results/${DATE}_ablation_no_markov_${MAX_STEPS}s_${MODEL_SHORT}" \
        --llm-judge \
        $COMMON_ARGS
else
    echo ">>> [3/6] SKIPPED: No Semi-Markov"
fi

# =============================================================================
# CONDITION 4: No Interrupts (No Assistance, No Off-Topic)
# Tests: Do interrupt behaviors add to perceptual fidelity?
# =============================================================================

if should_run 4; then
    echo ""
    echo ">>> [4/6] Running ABLATION: No Interrupts..."
    python evaluation/run_batch_simulations.py \
        --disable-assistance \
        --disable-offtopic \
        --output-dir "results/${DATE}_ablation_no_interrupts_${MAX_STEPS}s_${MODEL_SHORT}" \
        --llm-judge \
        $COMMON_ARGS
else
    echo ">>> [4/6] SKIPPED: No Interrupts"
fi

# =============================================================================
# CONDITION 5: No Γ_S (Strategist Memory Disabled)
# Tests: Does strategist memory (thought_buffer, episodic_memories) help?
# =============================================================================

if should_run 5; then
    echo ""
    echo ">>> [5/6] Running ABLATION: No Strategist Memory (Γ_S)..."
    python evaluation/run_batch_simulations.py \
        --disable-memory-strategist \
        --output-dir "results/${DATE}_ablation_no_mem_strategist_${MAX_STEPS}s_${MODEL_SHORT}" \
        --llm-judge \
        $COMMON_ARGS
else
    echo ">>> [5/6] SKIPPED: No Strategist Memory"
fi

# =============================================================================
# CONDITION 6: No Γ_E (Executor Memory Disabled)
# Tests: Does executor memory (agent_notes) help?
# =============================================================================

if should_run 6; then
    echo ""
    echo ">>> [6/6] Running ABLATION: No Executor Memory (Γ_E)..."
    python evaluation/run_batch_simulations.py \
        --disable-memory-executor \
        --output-dir "results/${DATE}_ablation_no_mem_executor_${MAX_STEPS}s_${MODEL_SHORT}" \
        --llm-judge \
        $COMMON_ARGS
else
    echo ">>> [6/6] SKIPPED: No Executor Memory"
fi

# =============================================================================
# COMPLETION SUMMARY
# =============================================================================

echo ""
echo "=============================================="
echo "ALL ABLATION CONDITIONS COMPLETE!"
echo "=============================================="
echo ""
echo "Results saved to:"
echo ""
echo "  Control:"
echo "    - results/${DATE}_ablation_full_${MODEL_SHORT}/"
echo ""
echo "  Ablation Conditions:"
echo "    - results/${DATE}_ablation_no_bkt_${MODEL_SHORT}/"
echo "    - results/${DATE}_ablation_no_markov_${MODEL_SHORT}/"
echo "    - results/${DATE}_ablation_no_interrupts_${MODEL_SHORT}/"
echo "    - results/${DATE}_ablation_no_mem_strategist_${MODEL_SHORT}/"
echo "    - results/${DATE}_ablation_no_mem_executor_${MODEL_SHORT}/"
echo ""
echo "=============================================="
echo "Next step: Run LLM-as-Judge on each folder:"
echo ""
echo "  for folder in results/${DATE}_ablation_*; do"
echo "    python evaluation/evaluator.py --folder \$folder --llm-judge \\"
echo "      --batch-mode --full-context --sample-size 50"
echo "  done"
echo ""
echo "Or generate comprehensive metrics:"
echo "  python evaluation/compute_comprehensive_metrics.py"
echo "=============================================="
