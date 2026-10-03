#!/usr/bin/env python3
"""
Batch Simulation Runner for Baseline Methods

Runs N simulations using baseline methods (e.g., VanillaStudent, CoTStudent) for comparison.
This is separate from run_batch_simulations.py which runs BEAGLE.

Used for RQ2: "Does the Neuro-Symbolic framework mitigate Competence Bias?"

Usage:
    # Vanilla baseline (pure LLM, no Markov control)
    python evaluation/run_baseline_batch.py --baseline vanilla --n-per-level 15 --output-dir results/vanilla_30
    
    # Chain-of-Thought baseline
    python evaluation/run_baseline_batch.py --baseline cot --n-per-level 15 --output-dir results/cot_30
"""

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv()

from beagle.data_generation.baselines.vanilla_student import (
    VanillaStudent, VanillaSimulationResult, run_vanilla_simulation
)
from beagle.data_generation.baselines.cot_student import (
    CoTStudent, CoTSimulationResult, run_cot_simulation
)
from beagle.data_generation.baselines.fewshot_student import (
    FewShotStudent, FewShotSimulationResult, run_fewshot_simulation
)
from beagle.data_generation.baselines.llmss_student import (
    LLMSSStudent, LLMSSSimulationResult, run_llmss_simulation
)
from beagle.data_generation.baselines.simstudent import (
    SimStudent, SimStudentSimulationResult, run_simstudent_simulation
)
from beagle.data_generation.baselines.coderagent_student import (
    CoderAgentStudent, CoderAgentSimulationResult, run_coderagent_simulation
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

# Suppress noisy logs
logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# =============================================================================
# Result Saving (Compatible format with BEAGLE's run_batch_simulations.py)
# =============================================================================


def save_single_run(result, output_dir: Path):
    """Save a single run immediately after completion."""
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_dir = output_dir / "runs"
    runs_dir.mkdir(exist_ok=True)

    run_file = runs_dir / f"run_{result.run_id:03d}_{result.performance_level}.json"
    with open(run_file, 'w') as f:
        json.dump(result.to_dict(), f, indent=2, default=str)

    log.info(f"  Saved run to {run_file}")


def save_results(
    results: List, output_dir: Path,
    baseline_type: str
):
    """Save all results to the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save individual runs (in case not saved incrementally)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(exist_ok=True)

    for result in results:
        run_file = runs_dir / f"run_{result.run_id:03d}_{result.performance_level}.json"
        with open(run_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

    # 2. Save summary (without full history)
    summary_results = []
    for r in results:
        summary = r.to_dict()
        del summary['history']
        summary_results.append(summary)

    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(
            {
                'baseline_type': baseline_type,
                'total_runs': len(results),
                'timestamp': datetime.now().isoformat(),
                'runs': summary_results
            },
            f,
            indent=2
        )

    # 3. Save aggregate statistics
    low_results = [r for r in results if r.performance_level == 'low']
    high_results = [r for r in results if r.performance_level == 'high']

    def compute_stats(group: List, name: str) -> dict:
        if not group:
            return {
                'name': name,
                'n': 0
            }

        n = len(group)
        solved = sum(1 for r in group if r.solved)
        steps = [r.total_steps for r in group]

        # Aggregate action distribution
        action_total = {}
        for r in group:
            for k, v in r.action_counts.items():
                action_total[k] = action_total.get(k, 0) + v

        total_steps = sum(steps)

        return {
            'name': name,
            'n': n,
            'solved_count': solved,
            'solved_rate': solved / n,
            'mean_steps': sum(steps) / n,
            'min_steps': min(steps),
            'max_steps': max(steps),
            'action_distribution': {
                k: v / total_steps
                for k, v in action_total.items()
            } if total_steps > 0 else {},
        }

    stats = {
        'baseline_type': baseline_type,
        'low_performers': compute_stats(low_results, 'low'),
        'high_performers': compute_stats(high_results, 'high'),
        'all': compute_stats(results, 'all'),
    }

    stats_file = output_dir / "statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    log.info(f"Results saved to {output_dir}")
    log.info(f"  - {len(results)} individual runs in {runs_dir}")
    log.info(f"  - Summary: {summary_file}")
    log.info(f"  - Statistics: {stats_file}")


def print_summary(results: List, baseline_type: str):
    """Print a summary table of results."""
    print("\n" + "=" * 80)
    print(f"BASELINE SIMULATION SUMMARY ({baseline_type.upper()})")
    print("=" * 80)

    low_results = [r for r in results if r.performance_level == 'low']
    high_results = [r for r in results if r.performance_level == 'high']

    for name, group in [
        ('LOW PERFORMERS', low_results), ('HIGH PERFORMERS', high_results)
    ]:
        if not group:
            continue

        print(f"\n{name} (n={len(group)})")
        print("-" * 60)

        solved = sum(1 for r in group if r.solved)
        steps = [r.total_steps for r in group]

        print(
            f"  Solved: {solved}/{len(group)} ({solved/len(group)*100:.1f}%)"
        )
        print(
            f"  Steps: mean={sum(steps)/len(group):.1f}, min={min(steps)}, max={max(steps)}"
        )

        # Action distribution
        action_total = {}
        for r in group:
            for k, v in r.action_counts.items():
                action_total[k] = action_total.get(k, 0) + v
        total_actions = sum(action_total.values())

        if total_actions > 0:
            print("  Action Distribution:")
            for action, count in sorted(action_total.items()):
                print(f"    {action}: {count/total_actions*100:.1f}%")

        # Per-run table
        print(f"\n  {'Run':>4} | {'Steps':>5} | {'Solved':>6}")
        print("  " + "-" * 25)
        for r in group:
            print(
                f"  {r.run_id:>4} | {r.total_steps:>5} | {'YES' if r.solved else 'NO':>6}"
            )

    print("\n" + "=" * 80)


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Baseline Batch Simulation Runner for BEAGLE Evaluation"
    )

    parser.add_argument(
        '--baseline',
        type=str,
        default='vanilla',
        choices=['vanilla', 'cot', 'fewshot', 'llmss', 'simstudent', 'coderagent'],
        help='Baseline method to run (default: vanilla)'
    )
    parser.add_argument(
        '--n-per-level',
        type=int,
        default=15,
        help='Number of simulations per performance level (default: 15)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/baseline_vanilla',
        help='Output directory for results'
    )
    parser.add_argument(
        '--problem',
        type=str,
        default='particle_simulator',
        help='Problem ID (default: particle_simulator)'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=30,
        help='Maximum steps per simulation (default: 30)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='google-gla:gemini-2.0-flash',
        help='LLM model to use'
    )
    parser.add_argument(
        '--levels',
        type=str,
        default='both',
        choices=['low', 'high', 'both'],
        help='Which performance levels to run (default: both)'
    )

    # LLM Judge options
    parser.add_argument(
        '--llm-judge',
        action='store_true',
        help='Run LLM-as-Judge evaluation after simulations complete'
    )
    parser.add_argument(
        '--llm-judge-model',
        type=str,
        default='google-gla:gemini-2.5-pro',
        help='LLM model for judge (default: gemini-2.5-pro)'
    )
    parser.add_argument(
        '--llm-judge-sample-size',
        type=int,
        default=50,
        help='Number of traces to judge (default: 50)'
    )
    parser.add_argument(
        '--llm-judge-strategy',
        type=str,
        default='balanced',
        choices=['random', 'balanced'],
        help='How to sample traces for judging (default: balanced)'
    )
    
    # Metacog option (Option C+: Stateful SRL prompting)
    parser.add_argument(
        '--enable-metacog',
        action='store_true',
        help='Enable SRL metacog tracking (Planning, Enacting, Monitoring, Reflecting)'
    )

    args = parser.parse_args()

    # Determine levels
    if args.levels == 'both':
        levels = ['low', 'high']
    else:
        levels = [args.levels]

    total_runs = args.n_per_level * len(levels)

    print(f"\n{'='*60}")
    print(f"BASELINE BATCH SIMULATION")
    print(f"{'='*60}")
    print(f"Baseline: {args.baseline}")
    print(f"Problem: {args.problem}")
    print(f"Model: {args.model}")
    print(f"Runs per level: {args.n_per_level}")
    print(f"Levels: {levels}")
    print(f"Total runs: {total_runs}")
    print(f"Max steps: {args.max_steps}")
    print(f"Enable Metacog: {args.enable_metacog}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}\n")

    results = []
    output_dir = Path(args.output_dir)
    runs_dir = output_dir / 'runs'

    for level in levels:
        # Defensive cleanup: drop any in-progress checkpoint files (no `run_id`
        # field) so their run_ids are re-attempted from step 0. Mirrors the
        # cleanup pass in run_batch_simulations.py — baselines currently write
        # only the final result so this should be a no-op, but it future-proofs
        # against any baseline that adds per-step checkpointing later.
        if runs_dir.exists():
            for f in runs_dir.glob(f"run_*_{level}.json"):
                try:
                    with open(f) as fp:
                        data = json.load(fp)
                except (json.JSONDecodeError, OSError):
                    log.warning(f"Removing unreadable checkpoint: {f.name}")
                    f.unlink(missing_ok=True)
                    continue
                if 'run_id' not in data:
                    log.warning(
                        f"Removing in-progress checkpoint {f.name} "
                        f"so run will restart from step 0"
                    )
                    f.unlink(missing_ok=True)

        # Find max existing run_id
        existing_runs = list(runs_dir.glob(f"run_*_{level}.json")
                             ) if runs_dir.exists() else []
        if existing_runs:
            max_existing = max(
                int(f.stem.split('_')[1]) for f in existing_runs
            )
            run_id = max_existing + 1
        else:
            run_id = 1

        # Target a TOTAL count for this level, not "N more iterations" — without
        # this a resume after partial completion would overshoot.
        target_total = args.n_per_level
        remaining = max(target_total - (run_id - 1), 0)

        print(
            f"\n--- Running {remaining}/{target_total} {args.baseline} simulations for {level.upper()} performers (starting from run {run_id}) ---\n"
        )

        for i in range(remaining):
            if args.baseline == 'vanilla':
                result = run_vanilla_simulation(
                    run_id=run_id,
                    performance_level=level,
                    problem_id=args.problem,
                    max_steps=args.max_steps,
                    llm_model=args.model,
                    enable_metacog=args.enable_metacog
                )
            elif args.baseline == 'cot':
                result = run_cot_simulation(
                    run_id=run_id,
                    performance_level=level,
                    problem_id=args.problem,
                    max_steps=args.max_steps,
                    llm_model=args.model,
                    enable_metacog=args.enable_metacog
                )
            elif args.baseline == 'fewshot':
                result = run_fewshot_simulation(
                    run_id=run_id,
                    performance_level=level,
                    problem_id=args.problem,
                    max_steps=args.max_steps,
                    llm_model=args.model,
                    enable_metacog=args.enable_metacog
                )
            elif args.baseline == 'llmss':
                result = run_llmss_simulation(
                    run_id=run_id,
                    performance_level=level,
                    problem_id=args.problem,
                    max_steps=args.max_steps,
                    llm_model=args.model
                )
            elif args.baseline == 'simstudent':
                result = run_simstudent_simulation(
                    run_id=run_id,
                    performance_level=level,
                    problem_id=args.problem,
                    max_steps=args.max_steps,
                    llm_model=args.model
                )
            elif args.baseline == 'coderagent':
                result = run_coderagent_simulation(
                    run_id=run_id,
                    performance_level=level,
                    problem_id=args.problem,
                    max_steps=args.max_steps,
                    llm_model=args.model
                )
            else:
                raise ValueError(
                    f"Unknown baseline: {args.baseline}. Choose from: vanilla, cot, fewshot, llmss, simstudent, coderagent"
                )

            results.append(result)
            save_single_run(result, output_dir)
            run_id += 1

    # Save final results
    save_results(results, output_dir, args.baseline)
    print_summary(results, args.baseline)

    print(f"\nAll results saved to: {output_dir.absolute()}")

    # Run LLM Judge if requested
    if args.llm_judge:
        print(f"\n{'='*60}")
        print("RUNNING LLM-AS-JUDGE EVALUATION")
        print(f"{'='*60}")
        print(f"Model: {args.llm_judge_model}")
        print(f"Sample size: {args.llm_judge_sample_size}")
        print(f"Strategy: {args.llm_judge_strategy}")

        import subprocess
        judge_cmd = [
            sys.executable,
            str(project_root / 'evaluation' / 'evaluator.py'),
            '--folder', str(output_dir),
            '--llm-judge',
            '--llm-model', args.llm_judge_model,
            '--sample-size', str(args.llm_judge_sample_size),
            '--sample-strategy', args.llm_judge_strategy,
            '--batch-mode',  # Use batch API for efficiency
            '--full-context',  # Include full trace context
        ]
        log.info(f"Running: {' '.join(judge_cmd)}")
        subprocess.run(judge_cmd, check=True)
        print(f"\nLLM Judge evaluation complete. Results in: {output_dir / 'llm_judgments.json'}")


if __name__ == "__main__":
    main()

    """
    python evaluation/run_baseline_batch.py \
    --baseline vanilla \
    --n-per-level 25 \
    --output-dir results/2026-01-01_vanilla_metacog_gemini2.0flash \
    --max-steps 30 \
    --model google-gla:gemini-2.0-flash \
    --enable-metacog

python evaluation/run_baseline_batch.py \
    --baseline cot \
    --n-per-level 25 \
    --max-steps 30 \
    --model google-gla:gemini-2.0-flash \
    --enable-metacog \
    --output-dir results/2026-01-01_cot_metacog_gemini2.0flash \
    --llm-judge
# Few-Shot + Metacog (batch mode + full context)
python evaluation/run_baseline_batch.py \
    --baseline fewshot \
    --n-per-level 25 \
    --max-steps 30 \
    --model google-gla:gemini-2.0-flash \
    --enable-metacog \
    --output-dir results/2026-01-01_fewshot_metacog_gemini2.0flash \
    --llm-judge
    """