#!/usr/bin/env python3
"""
Batch Simulation Runner for BEAGLE Evaluation

Runs N simulations for each performance level and saves all outputs.
Designed for systematic data collection to support evaluation studies.

Usage:
    python evaluation/run_batch_simulations.py --n-per-level 15 --output-dir results/batch_30
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv()

from beagle.data_generation.ide_oracle.ide_oracle import IDEOracle
from beagle.data_generation.studentv2.student import Student
from beagle.tutor import TutorFactory, TutorType

# Prompt variants - currently only baseline is used
# Format: 'variant_name': ('executor_system', 'strategist_system' or None, 'executor_user' or None)
PROMPT_VARIANTS = {
    'baseline': ('executor_system.txt', None, None),
}

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

# EFI pattern detection
EFI_PATTERNS = {
    'KC_C2_MATH_LIBRARY': [
        r'import\s+math',
        r'from\s+math\s+import',
        r'math\.',
    ],
    'KC_P5_UNIT_RADIANS': [
        r'math\.radians',
        r'radians\s*\(',
    ],
    'KC_P6_UNIT_DEGREES': [
        r'math\.degrees',
        r'degrees\s*\(',
    ],
    'KC_C9_CLASS_DEFINITION': [
        r'^\s*class\s+\w+',  # class definition
    ],
}


def check_efi_violation(code: str, efi_kc: str) -> bool:
    """Check if code violates EFI by using the forbidden concept."""
    import re
    if not code or not efi_kc:
        return False
    patterns = EFI_PATTERNS.get(efi_kc, [])
    for pattern in patterns:
        if re.search(pattern, code, re.IGNORECASE | re.MULTILINE):
            return True
    return False


@dataclass
class SimulationResult:
    """Result of a single simulation run."""
    run_id: int
    performance_level: str
    problem_id: str
    seed: Optional[int]

    # Outcome metrics
    solved: bool
    total_steps: int

    # Behavioral metrics
    assistance_count: int
    offtopic_count: int

    # State distributions
    metacog_counts: Dict[str, int]
    action_counts: Dict[str, int]

    # Test progress
    final_tests_passed: int
    final_tests_total: int

    # Timing
    duration_seconds: float
    timestamp: str

    # Full history (for detailed analysis)
    history: List[Dict[str, Any]]

    # EFI-specific fields
    efi_kc: Optional[str] = None
    efi_violation_found: bool = False
    efi_violation_steps: List[int] = None

    def __post_init__(self):
        if self.efi_violation_steps is None:
            self.efi_violation_steps = []

    def to_dict(self) -> dict:
        return asdict(self)


def extract_metrics(
    history: List[Dict[str, Any]],
    efi_kc: Optional[str] = None
) -> Dict[str, Any]:
    """Extract metrics from simulation history."""
    if not history:
        return {
            'solved': False,
            'total_steps': 0,
            'assistance_count': 0,
            'offtopic_count': 0,
            'metacog_counts': {},
            'action_counts': {},
            'final_tests_passed': 0,
            'final_tests_total': 0,
            'efi_violation_found': False,
            'efi_violation_steps': [],
        }

    # Count states
    metacog_counts = {}
    action_counts = {}
    assistance_count = 0
    offtopic_count = 0
    efi_violation_steps = []

    for i, step in enumerate(history):
        # Metacognitive state
        metacog = step.get('metacognitive_state', 'Unknown')
        metacog_counts[metacog] = metacog_counts.get(metacog, 0) + 1

        # Cognitive action
        action = step.get('cognitive_state', 'Unknown')
        action_counts[action] = action_counts.get(action, 0) + 1

        # Count interrupts
        if action == 'ASSISTANCE':
            assistance_count += 1
        elif action == 'OFF_TOPIC':
            offtopic_count += 1

        # Check for EFI violations
        if efi_kc:
            code = step.get('code', '')
            if check_efi_violation(code, efi_kc):
                efi_violation_steps.append(i + 1)

    # Get final state
    last_step = history[-1]
    solved = last_step.get('success', False)

    # Test progress (if available in history)
    final_tests_passed = last_step.get('tests_passed', 0)
    final_tests_total = last_step.get('tests_total', 0)

    return {
        'solved': solved,
        'total_steps': len(history),
        'assistance_count': assistance_count,
        'offtopic_count': offtopic_count,
        'metacog_counts': metacog_counts,
        'action_counts': action_counts,
        'final_tests_passed': final_tests_passed,
        'final_tests_total': final_tests_total,
        'efi_violation_found': len(efi_violation_steps) > 0,
        'efi_violation_steps': efi_violation_steps,
    }


def run_single_simulation(
    run_id: int,
    performance_level: str,
    problem_id: str,
    max_steps: int,
    llm_model: str,
    duration_multiplier: float,
    seed: Optional[int] = None,
    variant: str = 'baseline',
    cache_bkt: bool = True,
    efi_kc: Optional[str] = None,
    disable_assistance: bool = False,
    disable_offtopic: bool = False,
    force_assistance_steps: Optional[List[int]] = None,
    disable_bkt: bool = False,
    disable_markov: bool = False,
    skip_bkt_on_assisted: bool = False,
    enable_feedback_filter: bool = False,
    bkt_profile: Optional[str] = None,
    bkt_seed: Optional[int] = None,
    profile_override: Optional[str] = None,  # Override Behavioral Profile
    checkpoint_path: Optional[str] = None,
    workspace_id: Optional[str] = None,
    disable_memory_strategist: bool = False,  # Ablation: disable thought_buffer, episodic_memories
    disable_memory_executor: bool = False,  # Ablation: disable monologue_buffer, agent_notes
    enable_merged_pipeline: bool = False,  # Ablation: merge Strategist+Executor into single LLM call
    tutor_type: Optional[str] = None,  # Tutor mechanism: none, rule_based, ml_based, zpd, default
) -> SimulationResult:

    """Run a single simulation and return results."""
    efi_info = f", EFI={efi_kc}" if efi_kc else ""
    assist_info = ", no-assist" if disable_assistance else ""
    offtopic_info = ", no-offtopic" if disable_offtopic else ""
    force_info = f", force-assist@{force_assistance_steps}" if force_assistance_steps else ""
    bkt_info = ", no-bkt" if disable_bkt else ""
    markov_info = ", no-markov" if disable_markov else ""
    skip_bkt_info = ", skip-bkt-on-assisted" if skip_bkt_on_assisted else ""
    bkt_profile_info = f", bkt-profile={bkt_profile}" if bkt_profile else ""
    filter_info = ", feedback-filter" if enable_feedback_filter else ""
    profile_override_info = f", profile={profile_override}" if profile_override else ""
    mem_strat_info = ", no-mem-strategist" if disable_memory_strategist else ""
    mem_exec_info = ", no-mem-executor" if disable_memory_executor else ""
    log.info(
        f"Starting run {run_id} ({performance_level} performer{efi_info}{assist_info}{offtopic_info}{force_info}{bkt_info}{markov_info}{skip_bkt_info}{bkt_profile_info}{filter_info}{profile_override_info}{mem_strat_info}{mem_exec_info})"
    )


    start_time = time.time()
    timestamp = datetime.now().isoformat()

    # Import nodes for setting filter mode
    from beagle.data_generation.studentv2 import nodes
    
    # NOTE: Prompt patching is no longer needed - nodes now use PromptAssembler directly
    # The granularity instructions are handled by the block system
    
    # Set feedback filter mode (legacy, now disabled by default)
    if hasattr(nodes, 'ENABLE_FEEDBACK_FILTER'):
        nodes.ENABLE_FEEDBACK_FILTER = enable_feedback_filter

    # Set memory ablation flags
    nodes.DISABLE_MEMORY_STRATEGIST = disable_memory_strategist
    nodes.DISABLE_MEMORY_EXECUTOR = disable_memory_executor
    
    # Set merged pipeline flag (ablation: tests if Strategist/Executor split prevents psychic debugging)
    nodes.ENABLE_MERGED_PIPELINE = enable_merged_pipeline

    try:
        # Load problem
        problem_def = IDEOracle.load_problem(problem_id)

        # Initialize oracle with isolated tmp directory for parallel safety
        if workspace_id:
            tmp_dir = project_root / f"beagle/data_generation/tmp_{workspace_id}"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            oracle = IDEOracle(save_history=True, tmp_dir=tmp_dir)
        else:
            oracle = IDEOracle(save_history=True)

        # Initialize student
        model_path = project_root / "beagle/data_generation/studentv2/semi_markov_model.joblib"

        # Create tutor strategy if specified
        tutor_strategy = None
        if tutor_type:
            from beagle.utils.llm_client import LLMClient
            tutor_llm = LLMClient(model=llm_model)
            tutor_strategy = TutorFactory.from_string(tutor_type, llm_client=tutor_llm)
            log.info(f"  Using tutor: {tutor_type}")

        # Initialize student with optional EFI and forced assistance
        efi_kcs = [efi_kc] if efi_kc else []
        student = Student(
            performance_level=performance_level,
            profile_override=profile_override,  # NEW: for ablation experiments
            model_path=str(model_path),
            llm_model=llm_model,
            ide_oracle=oracle,
            duration_multiplier=duration_multiplier,
            cache_bkt=cache_bkt,
            efi_kcs=efi_kcs,
            force_assistance_steps=force_assistance_steps or [],
            tutor_strategy=tutor_strategy,
            skip_bkt_on_assisted=skip_bkt_on_assisted,
            bkt_profile=bkt_profile,
            bkt_seed=bkt_seed
        )


        # Disable BKT if requested (for ablation study)
        if disable_bkt:
            student.bkt = None

        # Disable Semi-Markov model if requested (for ablation study)
        # Use uniform random sampling for metacog and cog states
        if disable_markov and student.markov_model:
            import random
            METACOG_STATES = ['Planning', 'Monitoring', 'Reflecting']
            COG_ACTIONS = ['CONSTRUCTING', 'DEBUGGING', 'ASSESSING']

            # Store original methods
            original_sample_next_state = student.markov_model.sample_next_state
            original_sample_action = student.markov_model.sample_action

            # Replace with uniform random sampling
            def random_sample_next_state(history, performance_level='low'):
                return random.choice(METACOG_STATES)

            def random_sample_action(metacog, performance_level='low', previous_action=None, is_first_step=False):
                return random.choice(COG_ACTIONS)

            student.markov_model.sample_next_state = random_sample_next_state
            student.markov_model.sample_action = random_sample_action

        # Disable assistance and/or off-topic if requested (for EFI tests / ablation)
        import beagle.data_generation.studentv2.nodes as nodes_module
        original_p_assistance = nodes_module.p_assistance
        original_p_offtopic = nodes_module.p_offtopic

        if disable_assistance:
            nodes_module.p_assistance = lambda progress, level: 0.0

        if disable_offtopic:
            nodes_module.p_offtopic = lambda progress, level: 0.0

        # Run simulation
        history = student.solve_problem(
            problem_description=problem_def.description,
            problem_id=problem_def.problem_id,
            required_kcs=problem_def.required_kcs,
            max_steps=max_steps,
            starting_code=problem_def.starting_code,
            checkpoint_path=checkpoint_path
        )

        # Restore assistance/offtopic functions if they were modified
        if disable_assistance:
            nodes_module.p_assistance = original_p_assistance
        if disable_offtopic:
            nodes_module.p_offtopic = original_p_offtopic

        # Restore semi-markov methods if they were modified
        if disable_markov and student.markov_model:
            student.markov_model.sample_next_state = original_sample_next_state
            student.markov_model.sample_action = original_sample_action

        # Extract metrics
        metrics = extract_metrics(history, efi_kc=efi_kc)

        duration = time.time() - start_time

        result = SimulationResult(
            run_id=run_id,
            performance_level=performance_level,
            problem_id=problem_id,
            seed=seed,
            solved=metrics['solved'],
            total_steps=metrics['total_steps'],
            assistance_count=metrics['assistance_count'],
            offtopic_count=metrics['offtopic_count'],
            metacog_counts=metrics['metacog_counts'],
            action_counts=metrics['action_counts'],
            final_tests_passed=metrics['final_tests_passed'],
            final_tests_total=metrics['final_tests_total'],
            duration_seconds=duration,
            timestamp=timestamp,
            history=history,
            efi_kc=efi_kc,
            efi_violation_found=metrics.get('efi_violation_found', False),
            efi_violation_steps=metrics.get('efi_violation_steps', [])
        )

        efi_status = ""
        if efi_kc:
            efi_status = f", EFI_VIOLATION={'YES' if result.efi_violation_found else 'NO'}"
        log.info(
            f"  Run {run_id} complete: {'SOLVED' if result.solved else 'NOT SOLVED'} in {result.total_steps} steps ({duration:.1f}s){efi_status}"
        )

        return result

    except Exception as e:
        import traceback
        error_msg = str(e) if str(e) else traceback.format_exc()
        log.error(f"  Run {run_id} FAILED: {error_msg}")
        duration = time.time() - start_time

        # Return a failed result
        return SimulationResult(
            run_id=run_id,
            performance_level=performance_level,
            problem_id=problem_id,
            seed=seed,
            solved=False,
            total_steps=0,
            assistance_count=0,
            offtopic_count=0,
            metacog_counts={},
            action_counts={},
            final_tests_passed=0,
            final_tests_total=0,
            duration_seconds=duration,
            timestamp=timestamp,
            history=[{
                'error': error_msg
            }]
        )


def save_single_run(result: SimulationResult, output_dir: Path):
    """Save a single run immediately after completion."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save individual run (full history)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(exist_ok=True)

    run_file = runs_dir / f"run_{result.run_id:03d}_{result.performance_level}.json"
    with open(run_file, 'w') as f:
        json.dump(result.to_dict(), f, indent=2, default=str)

    log.info(f"  Saved run to {run_file}")


def save_results(results: List[SimulationResult], output_dir: Path):
    """Save all results to the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save individual runs (full history) - in case any weren't saved incrementally
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(exist_ok=True)

    for result in results:
        run_file = runs_dir / f"run_{result.run_id:03d}_{result.performance_level}.json"
        with open(run_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

    # RELOAD ALL RESULTS FROM DISK TO ENSURE AGGREGATION IS COMPLETE
    # (This handles cases where we run a partial batch like runs 11-30)
    all_results = []
    run_files = sorted(runs_dir.glob("run_*.json"))
    
    # Track loaded IDs to avoid duplicates if specific results were passed in *and* on disk
    loaded_ids = set()

    # Get valid fields for SimulationResult to avoid "unexpected keyword argument" errors
    # (Checkpoint files contain extra fields like 'status', 'current_code', etc.)
    from dataclasses import fields
    valid_fields = {f.name for f in fields(SimulationResult)}

    for run_file in run_files:
        try:
            with open(run_file, 'r') as f:
                data = json.load(f)
            
            # Filter data to only valid fields
            filtered_data = {k: v for k, v in data.items() if k in valid_fields}

            # Check if this is a valid result file (must have run_id)
            # Checkpoints from StudentState don't have run_id, so we skip them
            if 'run_id' not in filtered_data:
                log.warning(f"Skipping {run_file} - missing run_id (likely an in-progress checkpoint)")
                continue

            # Handle optionals that might be missing in older files
            if 'efi_violation_steps' not in filtered_data:
                 filtered_data['efi_violation_steps'] = []
            
            result = SimulationResult(**filtered_data)
            all_results.append(result)
            loaded_ids.add((result.run_id, result.performance_level))
        except Exception as e:
            log.warning(f"Failed to load {run_file} for aggregation: {e}")

    # Use the comprehensive list for summary and stats
    results = all_results

    # 2. Save summary (without full history)
    summary_results = []
    for r in results:
        summary = r.to_dict()
        if 'history' in summary:
            del summary['history']  # Remove full history for summary
        summary_results.append(summary)

    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(
            {
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

    def compute_stats(group: List[SimulationResult], name: str) -> dict:
        if not group:
            return {
                'name': name,
                'n': 0
            }

        n = len(group)
        solved = sum(1 for r in group if r.solved)
        steps = [r.total_steps for r in group]
        assistance = sum(r.assistance_count for r in group)
        offtopic = sum(r.offtopic_count for r in group)

        # Aggregate metacog distribution
        metacog_total = {}
        action_total = {}
        for r in group:
            for k, v in r.metacog_counts.items():
                metacog_total[k] = metacog_total.get(k, 0) + v
            for k, v in r.action_counts.items():
                action_total[k] = action_total.get(k, 0) + v

        total_steps = sum(steps)

        # EFI statistics
        efi_violations = sum(1 for r in group if r.efi_violation_found)
        efi_kc = next((r.efi_kc for r in group if r.efi_kc), None)

        result_dict = {
            'name': name,
            'n': n,
            'solved_count': solved,
            'solved_rate': solved / n,
            'mean_steps': sum(steps) / n,
            'min_steps': min(steps),
            'max_steps': max(steps),
            'total_assistance': assistance,
            'assistance_rate':
            assistance / total_steps if total_steps > 0 else 0,
            'total_offtopic': offtopic,
            'offtopic_rate': offtopic / total_steps if total_steps > 0 else 0,
            'metacog_distribution': {
                k: v / total_steps
                for k, v in metacog_total.items()
            } if total_steps > 0 else {},
            'action_distribution': {
                k: v / total_steps
                for k, v in action_total.items()
            } if total_steps > 0 else {},
        }

        # Add EFI stats if EFI was enabled
        if efi_kc:
            result_dict['efi_kc'] = efi_kc
            result_dict['efi_violation_count'] = efi_violations
            result_dict['efi_violation_rate'] = efi_violations / n

        return result_dict

    stats = {
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


def print_summary(results: List[SimulationResult]):
    """Print a summary table of results."""
    print("\n" + "=" * 80)
    print("BATCH SIMULATION SUMMARY")
    print("=" * 80)

    # Group by performance level
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
        total_steps = sum(steps)

        print(
            f"  Solved: {solved}/{len(group)} ({solved/len(group)*100:.1f}%)"
        )
        print(
            f"  Steps: mean={sum(steps)/len(group):.1f}, min={min(steps)}, max={max(steps)}"
        )

        # Interrupt rates
        total_assist = sum(r.assistance_count for r in group)
        total_offtopic = sum(r.offtopic_count for r in group)

        assist_pct = (total_assist/total_steps*100) if total_steps > 0 else 0
        offtopic_pct = (total_offtopic/total_steps*100) if total_steps > 0 else 0
        print(
            f"  Assistance: {total_assist} total ({assist_pct:.1f}% of steps)"
        )
        print(
            f"  Off-topic: {total_offtopic} total ({offtopic_pct:.1f}% of steps)"
        )

        # Check if EFI was enabled
        has_efi = any(r.efi_kc for r in group)

        # Per-run table
        if has_efi:
            print(
                f"\n  {'Run':>4} | {'Steps':>5} | {'Solved':>6} | {'Assist':>6} | {'EFI Viol':>8}"
            )
            print("  " + "-" * 55)
            for r in group:
                print(
                    f"  {r.run_id:>4} | {r.total_steps:>5} | {'YES' if r.solved else 'NO':>6} | {r.assistance_count:>6} | {'YES' if r.efi_violation_found else 'NO':>8}"
                )

            # EFI summary
            violations = sum(1 for r in group if r.efi_violation_found)
            print(
                f"\n  EFI Violation Rate: {violations}/{len(group)} ({violations/len(group)*100:.1f}%)"
            )
        else:
            print(
                f"\n  {'Run':>4} | {'Steps':>5} | {'Solved':>6} | {'Assist':>6} | {'OffTopic':>8}"
            )
            print("  " + "-" * 50)
            for r in group:
                print(
                    f"  {r.run_id:>4} | {r.total_steps:>5} | {'YES' if r.solved else 'NO':>6} | {r.assistance_count:>6} | {r.offtopic_count:>8}"
                )

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="BEAGLE Batch Simulation Runner"
    )

    parser.add_argument(
        '--n-per-level',
        '-n',
        type=int,
        default=15,
        help='Number of simulations per performance level (default: 15)'
    )
    parser.add_argument(
        '--output-dir',
        '-o',
        type=str,
        default='results/batch',
        help='Output directory for results (default: results/batch)'
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
        default='google-gla:gemini-2.5-flash',
        help='LLM model to use'
    )
    parser.add_argument(
        '--duration-multiplier',
        type=float,
        default=0.5,
        help='Duration multiplier for metacog segments (default: 0.5)'
    )
    parser.add_argument(
        '--levels',
        type=str,
        default='both',
        choices=['low', 'high', 'both'],
        help='Which performance levels to run (default: both)'
    )
    parser.add_argument(
        '--variant',
        type=str,
        default='baseline',
        choices=list(PROMPT_VARIANTS.keys()),
        help=
        f'Prompt variant to use (default: baseline). Options: {list(PROMPT_VARIANTS.keys())}'
    )
    parser.add_argument(
        '--no-cache-bkt',
        action='store_true',
        help=
        'Disable BKT caching (sample BKT per cognitive step instead of per metacog phase)'
    )
    parser.add_argument(
        '--efi-kc',
        type=str,
        default=None,
        help=
        'Knowledge Component to apply EFI (Explicit Flaw Injection) to. E.g., KC_C2_MATH_LIBRARY'
    )
    parser.add_argument(
        '--disable-assistance',
        action='store_true',
        help='Disable tutor assistance (for strict EFI testing)'
    )
    parser.add_argument(
        '--disable-offtopic',
        action='store_true',
        help='Disable off-topic behavior (for ablation study)'
    )
    parser.add_argument(
        '--force-assistance-steps',
        type=str,
        default=None,
        help=
        'Comma-separated list of step indices (0-based) to force assistance. '
        'E.g., "4,9,14" forces assistance at steps 5, 10, 15 (step_count is checked before increment).'
    )
    parser.add_argument(
        '--no-bkt',
        action='store_true',
        help='Disable BKT knowledge tracking entirely (ablation study)'
    )
    parser.add_argument(
        '--disable-markov',
        action='store_true',
        help='Disable Semi-Markov model: use random metacog/cog state sampling (ablation study)'
    )
    parser.add_argument(
        '--skip-bkt-on-assisted',
        action='store_true',
        help='Skip BKT updates when student uses tutor help (Performance != Competence mode for transfer test case study)'
    )
    parser.add_argument(
        '--bkt-profile',
        type=str,
        default=None,
        choices=['zero', 'low', 'average', 'high'],
        help='Override BKT profile for all students: zero (all start at 0.0), low (BELOW_AVERAGE), average (AVERAGE), or high (ABOVE_AVERAGE).'
    )
    parser.add_argument(
        '--bkt-seed',
        type=int,
        default=None,
        help='Seed for BKT initialization. If set, all runs use identical initial P(L) values. If not set, P(L) is sampled randomly per run.'
    )
    parser.add_argument(
        '--profile-override',
        type=str,
        default=None,
        choices=['low', 'high'],
        help='Override Behavioral Profile independently of Semi-Markov level. '
             'Use for ablation: --levels high --profile-override low tests '
             'HIGH Markov + LOW Profile (Mixed A condition).'
    )

    parser.add_argument(
        '--feedback-filter',
        action='store_true',
        help='Enable feedback quality differentiation: LOW sees pass/fail only, HIGH sees full tracebacks'
    )

    # Memory Ablation Flags (for ablation study)
    parser.add_argument(
        '--disable-memory-strategist',
        action='store_true',
        help='Disable Strategist memory (thought_buffer, episodic_memories) for ablation study'
    )
    parser.add_argument(
        '--disable-memory-executor',
        action='store_true',
        help='Disable Executor memory (monologue_buffer, agent_notes) for ablation study'
    )
    parser.add_argument(
        '--enable-merged-pipeline',
        action='store_true',
        help='ABLATION: Merge Strategist+Executor into single LLM call (tests if split prevents psychic debugging)'
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
    parser.add_argument(
        '--disable-emotional-comments',
        action='store_true',
        help='Disable emotional comments like "# hope this works", use sparse/realistic comments instead'
    )
    parser.add_argument(
        '--workspace-id',
        type=str,
        default=None,
        help='Unique workspace ID for parallel runs. Creates isolated tmp_<id>/ directory for solution.py to avoid conflicts.'
    )
    parser.add_argument(
        '--tutor-type',
        type=str,
        default='zpd',
        choices=['none', 'rule_based', 'ml_based', 'zpd', 'simple_llm'],
        help='Tutor mechanism: none (no hints), rule_based (Hint Factory), ml_based (BKT-informed), zpd (ZPD scaffolding, default), simple_llm (simple LLM prompts)'
    )

    args = parser.parse_args()

    # Parse force_assistance_steps
    force_assistance_steps = None
    if args.force_assistance_steps:
        force_assistance_steps = [
            int(s.strip()) for s in args.force_assistance_steps.split(',')
        ]

    # Determine which levels to run
    if args.levels == 'both':
        levels = ['low', 'high']
    else:
        levels = [args.levels]

    total_runs = args.n_per_level * len(levels)

    variant_config = PROMPT_VARIANTS[args.variant]

    cache_bkt = not args.no_cache_bkt

    # *** EARLY MODEL VALIDATION - Fail fast if model format is invalid ***
    from beagle.utils.llm_client import LLMClient
    try:
        # Test initialization to validate model format
        test_client = LLMClient(model=args.model)
        del test_client
    except Exception as e:
        log.error(f"Invalid model configuration: {e}")
        sys.exit(1)

    # Configure comment style in prompts (both old and new prompting systems)
    import beagle.data_generation.studentv2.nodes as nodes_module
    nodes_module.USE_SPARSE_COMMENTS = args.disable_emotional_comments
    
    # Also set in prompting module
    from beagle.data_generation.studentv2.prompting import set_sparse_comments
    set_sparse_comments(args.disable_emotional_comments)

    print(f"\n{'='*60}")
    print(f"BEAGLE BATCH SIMULATION")
    print(f"{'='*60}")
    print(f"Problem: {args.problem}")
    print(f"Variant: {args.variant}")
    print(f"  Executor System: {variant_config[0]}")
    if variant_config[1]:
        print(f"  Strategist: {variant_config[1]}")
    if len(variant_config) > 2 and variant_config[2]:
        print(f"  Executor User: {variant_config[2]}")
    print(f"BKT: {'DISABLED' if args.no_bkt else 'enabled'} (caching: {cache_bkt})")
    print(f"Semi-Markov: {'DISABLED (random sampling)' if args.disable_markov else 'enabled'}")
    print(f"Comment style: {'sparse/realistic' if args.disable_emotional_comments else 'emotional (# hope this works)'}")
    print(f"Assistance: {'DISABLED' if args.disable_assistance else 'enabled'}")
    print(f"Off-topic: {'DISABLED' if args.disable_offtopic else 'enabled'}")
    print(f"Tutor: {args.tutor_type}")
    if args.efi_kc:
        print(f"EFI Target KC: {args.efi_kc}")
    if force_assistance_steps:
        print(f"Force Assistance at Steps: {force_assistance_steps}")
    print(f"Runs per level: {args.n_per_level}")
    print(f"Levels: {levels}")
    print(f"Total runs: {total_runs}")
    print(f"Max steps: {args.max_steps}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}\n")

    results = []
    output_dir = Path(args.output_dir)
    runs_dir = output_dir / 'runs'

    for level in levels:
        # Cleanup pass: any run_NNN_level.json that's a mid-flight checkpoint
        # (no `run_id` field — only the completed-run write adds that key) gets
        # deleted so its run_id is re-attempted from step 0 on restart. Without
        # this, the resume logic below treats the incomplete file as "done"
        # because max_existing is computed from filenames alone.
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
                        f"(steps={data.get('step_count', '?')}/{data.get('max_steps', '?')}) "
                        f"so run will restart from step 0"
                    )
                    f.unlink(missing_ok=True)

        # Find the max existing run_id for this level to continue from
        existing_runs = list(runs_dir.glob(f"run_*_{level}.json")
                             ) if runs_dir.exists() else []
        if existing_runs:
            max_existing = max(
                int(f.stem.split('_')[1]) for f in existing_runs
            )
            run_id = max_existing + 1
        else:
            run_id = 1

        # Target a TOTAL count for this level, not "N more iterations". Without
        # this, a resume after partial completion (e.g. 22 of 30 done) would do
        # n_per_level fresh iterations starting at run 23 and overshoot to 37.
        target_total = args.n_per_level
        remaining = max(target_total - (run_id - 1), 0)

        print(
            f"\n--- Running {remaining}/{target_total} simulations for {level.upper()} performers (starting from run {run_id}) ---\n"
        )

        for i in range(remaining):
            # Generate checkpoint path for incremental saving
            checkpoint_file = runs_dir / f"run_{run_id:03d}_{level}.json"
            
            result = run_single_simulation(
                run_id=run_id,
                performance_level=level,
                problem_id=args.problem,
                max_steps=args.max_steps,
                llm_model=args.model,
                duration_multiplier=args.duration_multiplier,
                seed=None,  # No pre-generated sequence for now
                variant=args.variant,
                cache_bkt=cache_bkt,
                efi_kc=args.efi_kc,
                disable_assistance=args.disable_assistance,
                disable_offtopic=args.disable_offtopic,
                force_assistance_steps=force_assistance_steps,
                disable_bkt=args.no_bkt,
                disable_markov=args.disable_markov,
                skip_bkt_on_assisted=args.skip_bkt_on_assisted,
                enable_feedback_filter=args.feedback_filter,
                bkt_profile=args.bkt_profile,
                bkt_seed=args.bkt_seed,
                profile_override=args.profile_override,
                checkpoint_path=str(checkpoint_file),
                workspace_id=args.workspace_id,
                disable_memory_strategist=args.disable_memory_strategist,
                disable_memory_executor=args.disable_memory_executor,
                enable_merged_pipeline=args.enable_merged_pipeline,
                tutor_type=args.tutor_type,
            )

            results.append(result)

            # Save immediately after each run (incremental saving)
            save_single_run(result, output_dir)

            run_id += 1

    # Save summary and statistics (final aggregation)
    save_results(results, output_dir)

    # Print summary
    print_summary(results)

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
            '--full-context',
            '--batch-mode',
        ]
        log.info(f"Running: {' '.join(judge_cmd)}")
        subprocess.run(judge_cmd, check=True)
        print(f"\nLLM Judge evaluation complete. Results in: {output_dir / 'llm_judgments.json'}")


if __name__ == "__main__":
    main()
