#!/bin/bash
# =============================================================================
# BEAGLE: Run All Baselines for Comprehensive Evaluation (Reproducibility Script)
# =============================================================================
# 
# This script reproduces ALL baseline experiments from the paper, including:
#   - Pure LLM baselines (Vanilla, CoT, Few-Shot)
#   - Structured approaches (LLMSS, SimStudent)
#   - Metacog-enhanced baselines (+M variants)
#   - BEAGLE framework
#   - LLM-as-Judge evaluation
#
# Usage:
#   ./run_all_baselines.sh [n_per_level] [model_version]
#
# Arguments:
#   n_per_level:   Number of runs per performance level (default: 25)
#   model_version: "2.0" for Gemini 2.0 Flash, "2.5" for Gemini 2.5 Flash (default: 2.0)
#
# Examples:
#   ./run_all_baselines.sh 25 2.0   # Full evaluation with Gemini 2.0 Flash
#   ./run_all_baselines.sh 10 2.5   # Quick test with Gemini 2.5 Flash
#
# =============================================================================

set -e

N_PER_LEVEL=${1:-25}
MODEL_VERSION=${2:-2.0}
MAX_STEPS=30
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

# Common arguments
COMMON_ARGS="--n-per-level $N_PER_LEVEL --problem $PROBLEM --max-steps $MAX_STEPS --model $MODEL --llm-judge"

echo "=============================================="
echo "BEAGLE COMPREHENSIVE BASELINE RUNNER"
echo "=============================================="
echo "Date: $DATE"
echo "Runs per level: $N_PER_LEVEL"
echo "Max steps: $MAX_STEPS"
echo "Problem: $PROBLEM"
echo "Model: $MODEL"
echo "LLM Judge: Enabled (batch-mode + full-context)"
echo "=============================================="
echo ""

# Change to project root
cd "$(dirname "$0")/.."

# =============================================================================
# SECTION 1: Pure LLM Baselines (without metacog)
# =============================================================================

echo ""
echo ">>> [1/10] Running VANILLA BASELINE..."
python evaluation/run_baseline_batch.py \
    --baseline vanilla \
    --output-dir "results/${DATE}_vanilla_${MODEL_SHORT}" \
    $COMMON_ARGS

echo ""
echo ">>> [2/10] Running CHAIN-OF-THOUGHT BASELINE..."
python evaluation/run_baseline_batch.py \
    --baseline cot \
    --output-dir "results/${DATE}_cot_${MODEL_SHORT}" \
    $COMMON_ARGS

echo ""
echo ">>> [3/10] Running FEW-SHOT BASELINE..."
python evaluation/run_baseline_batch.py \
    --baseline fewshot \
    --output-dir "results/${DATE}_fewshot_${MODEL_SHORT}" \
    $COMMON_ARGS

# =============================================================================
# SECTION 2: Structured Approaches
# =============================================================================

echo ""
echo ">>> [4/10] Running LLMSS BASELINE..."
python evaluation/run_baseline_batch.py \
    --baseline llmss \
    --output-dir "results/${DATE}_llmss_${MODEL_SHORT}" \
    $COMMON_ARGS

echo ""
echo ">>> [5/10] Running SIMSTUDENT BASELINE..."
python evaluation/run_baseline_batch.py \
    --baseline simstudent \
    --output-dir "results/${DATE}_simstudent_${MODEL_SHORT}" \
    $COMMON_ARGS

# =============================================================================
# SECTION 3: Metacog-Enhanced Baselines (+M variants)
# These add SRL tracking (Planning, Enacting, Monitoring, Reflecting)
# =============================================================================

echo ""
echo ">>> [6/10] Running VANILLA+METACOG BASELINE..."
python evaluation/run_baseline_batch.py \
    --baseline vanilla \
    --enable-metacog \
    --output-dir "results/${DATE}_vanilla_metacog_${MODEL_SHORT}" \
    $COMMON_ARGS

echo ""
echo ">>> [7/10] Running COT+METACOG BASELINE..."
python evaluation/run_baseline_batch.py \
    --baseline cot \
    --enable-metacog \
    --output-dir "results/${DATE}_cot_metacog_${MODEL_SHORT}" \
    $COMMON_ARGS

echo ""
echo ">>> [8/10] Running FEWSHOT+METACOG BASELINE..."
python evaluation/run_baseline_batch.py \
    --baseline fewshot \
    --enable-metacog \
    --output-dir "results/${DATE}_fewshot_metacog_${MODEL_SHORT}" \
    $COMMON_ARGS

# =============================================================================
# SECTION 4: BEAGLE Framework (Ours)
# =============================================================================

echo ""
echo ">>> [9/10] Running BEAGLE (Full Framework)..."
python evaluation/run_batch_simulations.py \
    --n-per-level $N_PER_LEVEL \
    --max-steps $MAX_STEPS \
    --model $MODEL \
    --output-dir "results/${DATE}_beagle_${MODEL_SHORT}"

# Run LLM Judge for BEAGLE separately
echo ""
echo ">>> [10/10] Running LLM-as-Judge for BEAGLE..."
python evaluation/evaluator.py \
    --folder "results/${DATE}_beagle_${MODEL_SHORT}" \
    --llm-judge \
    --batch-mode \
    --full-context \
    --sample-size 50 \
    --sample-strategy balanced \
    --export "results/${DATE}_beagle_${MODEL_SHORT}/evaluation.json"

# =============================================================================
# COMPLETION SUMMARY
# =============================================================================

echo ""
echo "=============================================="
echo "ALL BASELINES COMPLETE!"
echo "=============================================="
echo ""
echo "Results saved to:"
echo ""
echo "  Pure LLM Baselines:"
echo "    - results/${DATE}_vanilla_${MODEL_SHORT}/"
echo "    - results/${DATE}_cot_${MODEL_SHORT}/"
echo "    - results/${DATE}_fewshot_${MODEL_SHORT}/"
echo ""
echo "  Structured Approaches:"
echo "    - results/${DATE}_llmss_${MODEL_SHORT}/"
echo "    - results/${DATE}_simstudent_${MODEL_SHORT}/"
echo ""
echo "  Metacog-Enhanced (+M):"
echo "    - results/${DATE}_vanilla_metacog_${MODEL_SHORT}/"
echo "    - results/${DATE}_cot_metacog_${MODEL_SHORT}/"
echo "    - results/${DATE}_fewshot_metacog_${MODEL_SHORT}/"
echo ""
echo "  BEAGLE (Ours):"
echo "    - results/${DATE}_beagle_${MODEL_SHORT}/"
echo ""
echo "=============================================="
echo "Next step: Generate LaTeX table with:"
echo "  python evaluation/compute_comprehensive_metrics.py"
echo "=============================================="
