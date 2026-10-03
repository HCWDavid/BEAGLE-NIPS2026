#!/usr/bin/env python3
"""
BEAGLE Simulation Evaluator

Comprehensive evaluation of simulation quality aligned with evaluation_plan.md.
Supports single run analysis and batch folder analysis.

Usage:
    # Single run analysis
    python evaluation/evaluator.py --run results/batch_30/runs/run_001_low.json

    # Batch folder analysis
    python evaluation/evaluator.py --folder results/batch_30

    # Compare two batches
    python evaluation/evaluator.py --compare results/batch_30 results/v6_particle_30

    # Export results to JSON
    python evaluation/evaluator.py --folder results/batch_30 --export results/batch_30/evaluation.json
"""

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import statistics

# Add project root for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ==================== Data Structures ====================

@dataclass
class RunMetrics:
    """Metrics for a single simulation run."""
    run_id: int
    performance_level: str
    problem_id: str

    # Outcome
    solved: bool
    total_steps: int

    # Study 1: Behavioral Fidelity
    metacog_counts: Dict[str, int] = field(default_factory=dict)
    action_counts: Dict[str, int] = field(default_factory=dict)

    # Study 2: Interrupt States
    assistance_count: int = 0
    offtopic_count: int = 0
    assistance_timings: List[float] = field(default_factory=list)  # Normalized [0,1]
    offtopic_timings: List[float] = field(default_factory=list)

    # Study 6: Code Quality
    code_style: Dict[str, Any] = field(default_factory=dict)
    error_patterns: Dict[str, int] = field(default_factory=dict)
    mindset_diversity: float = 0.0

    # Language patterns
    confusion_ratio: float = 0.0
    confidence_ratio: float = 0.0


@dataclass
class BatchMetrics:
    """Aggregated metrics for a batch of simulations."""
    batch_name: str
    n_runs: int

    # Study 1: Behavioral Fidelity
    metacog_distribution: Dict[str, float] = field(default_factory=dict)
    action_distribution: Dict[str, float] = field(default_factory=dict)

    # Study 2: Interrupt States
    assistance_rate: float = 0.0
    offtopic_rate: float = 0.0
    assistance_timing_mean: float = 0.0  # Mean session progress when assistance sought
    offtopic_timing_mean: float = 0.0

    # Study 5: Performance Differentiation (by level)
    low_metrics: Optional[Dict[str, Any]] = None
    high_metrics: Optional[Dict[str, Any]] = None

    # Study 6: Code Quality
    code_style_summary: Dict[str, Any] = field(default_factory=dict)

    # Overall
    success_rate: float = 0.0
    mean_steps: float = 0.0
    std_steps: float = 0.0
    debugging_ratio: float = 0.0
    constructing_ratio: float = 0.0


# ==================== Single Run Evaluation ====================

class RunEvaluator:
    """Evaluates a single simulation run."""

    def __init__(self, run_data: Dict[str, Any]):
        self.data = run_data
        self.history = run_data.get('history', [])

    def evaluate(self) -> RunMetrics:
        """Compute all metrics for this run."""
        metrics = RunMetrics(
            run_id=self.data.get('run_id', 0),
            performance_level=self.data.get('performance_level', 'unknown'),
            problem_id=self.data.get('problem_id', 'unknown'),
            solved=self.data.get('solved', False),
            total_steps=self.data.get('total_steps', len(self.history)),
            metacog_counts=self.data.get('metacog_counts', {}),
            action_counts=self.data.get('action_counts', {}),
            assistance_count=self.data.get('assistance_count', 0),
            offtopic_count=self.data.get('offtopic_count', 0),
        )

        # Compute additional metrics from history
        if self.history:
            metrics.assistance_timings = self._compute_interrupt_timings('ASSISTANCE')
            metrics.offtopic_timings = self._compute_interrupt_timings('OFF_TOPIC')
            metrics.code_style = self._analyze_code_style()
            metrics.error_patterns = self._detect_error_patterns()
            metrics.mindset_diversity = self._compute_mindset_diversity()
            metrics.confusion_ratio, metrics.confidence_ratio = self._analyze_language()

        return metrics

    def _compute_interrupt_timings(self, action_type: str) -> List[float]:
        """Get normalized timings of interrupt events."""
        timings = []
        for i, step in enumerate(self.history):
            if step.get('cognitive_state') == action_type:
                progress = i / len(self.history) if self.history else 0
                timings.append(progress)
        return timings

    def _analyze_code_style(self) -> Dict[str, Any]:
        """Analyze code style indicators."""
        style = {
            'cramped_assignments': 0,  # x=y
            'spaced_assignments': 0,   # x = y
            'single_letter_vars': 0,
            'descriptive_vars': 0,
            'emotional_comments': 0,
            'caret_power': 0,  # ^ instead of **
            'double_star_power': 0,
            'extra_parens': 0,  # if (x) or return (y)
            'typos': 0,
            'camel_case': 0,
            'snake_case': 0,
        }

        for step in self.history:
            code = step.get('code', '')
            if not code:
                continue

            # Assignment spacing
            style['cramped_assignments'] += len(re.findall(r'[a-zA-Z_]\w*=[^=]', code))
            style['spaced_assignments'] += len(re.findall(r'[a-zA-Z_]\w* = [^=]', code))

            # Variable naming
            style['single_letter_vars'] += len(re.findall(r'\b[a-z]\s*=', code))
            style['descriptive_vars'] += len(re.findall(r'\b[a-z]{4,}\w*\s*=', code))

            # Comments
            style['emotional_comments'] += len(re.findall(r'#.*(\?\?\?|idk|pls|help|why|ugh|hmm)', code, re.I))

            # Power operators
            style['caret_power'] += code.count('^')
            style['double_star_power'] += code.count('**')

            # Extra parens
            style['extra_parens'] += len(re.findall(r'if\s*\(|return\s*\(', code))

            # Typos
            style['typos'] += len(re.findall(r'lenght|widht|retrun|pritn|fucntion|calss|slef', code, re.I))

            # Naming convention
            style['camel_case'] += len(re.findall(r'\b[a-z]+[A-Z][a-z]+\w*', code))
            style['snake_case'] += len(re.findall(r'\b[a-z]+_[a-z]+\w*', code))

        return style

    def _detect_error_patterns(self) -> Dict[str, int]:
        """Detect common error patterns in code."""
        patterns = {
            'wrong_power_operator': 0,  # Using ^ instead of **
            'import_inside_function': 0,
            'bare_function_calls': 0,  # sin() without math.
            'formula_errors': 0,
            'syntax_errors_detected': 0,
        }

        for step in self.history:
            code = step.get('code', '')
            output = step.get('output', '')

            # Wrong power
            if re.search(r'\w\s*\^\s*\d', code):
                patterns['wrong_power_operator'] += 1

            # Import inside function
            if re.search(r'def\s+\w+.*:[\s\S]*?import\s+\w+', code):
                patterns['import_inside_function'] += 1

            # Bare function calls (sin, cos without math.)
            if re.search(r'(?<!math\.)\b(sin|cos|tan|sqrt|radians)\s*\(', code):
                patterns['bare_function_calls'] += 1

            # Check output for syntax errors
            if 'SyntaxError' in output or 'IndentationError' in output:
                patterns['syntax_errors_detected'] += 1

        return patterns

    def _compute_mindset_diversity(self) -> float:
        """Compute diversity of mindsets (unique mindsets / total steps)."""
        mindsets = [step.get('mindset', '') for step in self.history if step.get('mindset')]
        if not mindsets:
            return 0.0
        return len(set(mindsets)) / len(mindsets)

    def _analyze_language(self) -> Tuple[float, float]:
        """Analyze confusion vs confidence in monologue."""
        uncertainty_words = ["i think", "maybe", "not sure", "confused", "i guess", "try", "might", "probably"]
        confidence_words = ["i know", "i need to", "definitely", "i remember", "i'll just", "obviously"]

        uncertainty_count = 0
        confidence_count = 0
        total = 0

        for step in self.history:
            mono = step.get('monologue', step.get('thinking', '')).lower()
            mindset = step.get('mindset', '').lower()
            text = f"{mono} {mindset}"

            if text.strip():
                total += 1
                if any(w in text for w in uncertainty_words):
                    uncertainty_count += 1
                if any(w in text for w in confidence_words):
                    confidence_count += 1

        return (
            uncertainty_count / total if total > 0 else 0,
            confidence_count / total if total > 0 else 0
        )


# ==================== Batch Evaluation ====================

class BatchEvaluator:
    """Evaluates a batch of simulation runs."""

    def __init__(self, folder_path: Path):
        self.folder = Path(folder_path)
        self.runs_dir = self.folder / 'runs'
        self.run_metrics: List[RunMetrics] = []

    def load_runs(self) -> int:
        """Load all run JSON files from the folder."""
        if not self.runs_dir.exists():
            print(f"Warning: {self.runs_dir} does not exist")
            return 0

        for run_file in sorted(self.runs_dir.glob('*.json')):
            try:
                with open(run_file) as f:
                    run_data = json.load(f)
                evaluator = RunEvaluator(run_data)
                self.run_metrics.append(evaluator.evaluate())
            except Exception as e:
                print(f"Error loading {run_file}: {e}")

        return len(self.run_metrics)

    def evaluate(self) -> BatchMetrics:
        """Compute aggregated metrics for the batch."""
        if not self.run_metrics:
            self.load_runs()

        if not self.run_metrics:
            return BatchMetrics(batch_name=self.folder.name, n_runs=0)

        metrics = BatchMetrics(
            batch_name=self.folder.name,
            n_runs=len(self.run_metrics)
        )

        # Aggregate counts
        total_metacog = Counter()
        total_action = Counter()
        total_steps = 0
        assist_timings = []
        offtopic_timings = []

        for rm in self.run_metrics:
            for k, v in rm.metacog_counts.items():
                total_metacog[k] += v
            for k, v in rm.action_counts.items():
                total_action[k] += v
            total_steps += rm.total_steps
            assist_timings.extend(rm.assistance_timings)
            offtopic_timings.extend(rm.offtopic_timings)

        # Study 1: Distributions
        if total_steps > 0:
            metrics.metacog_distribution = {k: v/total_steps for k, v in total_metacog.items()}
            metrics.action_distribution = {k: v/total_steps for k, v in total_action.items()}

        # Study 2: Interrupt rates and timing
        metrics.assistance_rate = sum(rm.assistance_count for rm in self.run_metrics) / total_steps if total_steps > 0 else 0
        metrics.offtopic_rate = sum(rm.offtopic_count for rm in self.run_metrics) / total_steps if total_steps > 0 else 0
        metrics.assistance_timing_mean = statistics.mean(assist_timings) if assist_timings else 0
        metrics.offtopic_timing_mean = statistics.mean(offtopic_timings) if offtopic_timings else 0

        # Study 5: By performance level
        low_runs = [rm for rm in self.run_metrics if rm.performance_level == 'low']
        high_runs = [rm for rm in self.run_metrics if rm.performance_level == 'high']

        if low_runs:
            metrics.low_metrics = self._aggregate_level_metrics(low_runs, 'low')
        if high_runs:
            metrics.high_metrics = self._aggregate_level_metrics(high_runs, 'high')

        # Study 6: Code style summary
        metrics.code_style_summary = self._aggregate_code_style()

        # Overall metrics
        metrics.success_rate = sum(1 for rm in self.run_metrics if rm.solved) / len(self.run_metrics)
        steps = [rm.total_steps for rm in self.run_metrics]
        metrics.mean_steps = statistics.mean(steps)
        metrics.std_steps = statistics.stdev(steps) if len(steps) > 1 else 0
        metrics.debugging_ratio = metrics.action_distribution.get('DEBUGGING', 0)
        metrics.constructing_ratio = metrics.action_distribution.get('CONSTRUCTING', 0)

        return metrics

    def _aggregate_level_metrics(self, runs: List[RunMetrics], level: str) -> Dict[str, Any]:
        """Aggregate metrics for a performance level."""
        if not runs:
            return {}

        total_steps = sum(rm.total_steps for rm in runs)
        action_counts = Counter()
        for rm in runs:
            for k, v in rm.action_counts.items():
                action_counts[k] += v

        return {
            'n': len(runs),
            'success_rate': sum(1 for rm in runs if rm.solved) / len(runs),
            'mean_steps': statistics.mean([rm.total_steps for rm in runs]),
            'std_steps': statistics.stdev([rm.total_steps for rm in runs]) if len(runs) > 1 else 0,
            'min_steps': min(rm.total_steps for rm in runs),
            'max_steps': max(rm.total_steps for rm in runs),
            'debugging_ratio': action_counts.get('DEBUGGING', 0) / total_steps if total_steps > 0 else 0,
            'constructing_ratio': action_counts.get('CONSTRUCTING', 0) / total_steps if total_steps > 0 else 0,
            'assistance_rate': sum(rm.assistance_count for rm in runs) / total_steps if total_steps > 0 else 0,
            'offtopic_rate': sum(rm.offtopic_count for rm in runs) / total_steps if total_steps > 0 else 0,
            'mean_confusion_ratio': statistics.mean([rm.confusion_ratio for rm in runs]),
            'mean_mindset_diversity': statistics.mean([rm.mindset_diversity for rm in runs]),
        }

    def _aggregate_code_style(self) -> Dict[str, Any]:
        """Aggregate code style metrics by performance level."""
        result = {'low': {}, 'high': {}}

        for level in ['low', 'high']:
            runs = [rm for rm in self.run_metrics if rm.performance_level == level]
            if not runs:
                continue

            # Sum up all style counts
            style_totals = Counter()
            for rm in runs:
                for k, v in rm.code_style.items():
                    style_totals[k] += v

            # Compute cramped ratio
            cramped = style_totals.get('cramped_assignments', 0)
            spaced = style_totals.get('spaced_assignments', 0)
            total_assign = cramped + spaced

            result[level] = {
                'cramped_ratio': cramped / total_assign if total_assign > 0 else 0,
                'single_letter_var_count': style_totals.get('single_letter_vars', 0),
                'descriptive_var_count': style_totals.get('descriptive_vars', 0),
                'emotional_comments': style_totals.get('emotional_comments', 0),
                'typos': style_totals.get('typos', 0),
                'caret_power_usage': style_totals.get('caret_power', 0),
                'extra_parens': style_totals.get('extra_parens', 0),
            }

        return result

    def get_run_by_id(self, run_id: int) -> Optional[RunMetrics]:
        """Get a specific run by ID."""
        for rm in self.run_metrics:
            if rm.run_id == run_id:
                return rm
        return None


# ==================== Reporting ====================

def print_run_report(metrics: RunMetrics):
    """Print detailed report for a single run."""
    print("=" * 70)
    print(f"RUN EVALUATION: Run {metrics.run_id} ({metrics.performance_level})")
    print("=" * 70)

    print(f"\n### OUTCOME ###")
    print(f"  Problem:    {metrics.problem_id}")
    print(f"  Solved:     {'YES' if metrics.solved else 'NO'}")
    print(f"  Steps:      {metrics.total_steps}")

    print(f"\n### BEHAVIORAL STATES ###")
    print("  Metacognitive:")
    for state, count in sorted(metrics.metacog_counts.items(), key=lambda x: -x[1]):
        pct = count / metrics.total_steps * 100 if metrics.total_steps else 0
        print(f"    {state}: {count} ({pct:.1f}%)")

    print("  Cognitive Actions:")
    for action, count in sorted(metrics.action_counts.items(), key=lambda x: -x[1]):
        pct = count / metrics.total_steps * 100 if metrics.total_steps else 0
        print(f"    {action}: {count} ({pct:.1f}%)")

    print(f"\n### INTERRUPT STATES ###")
    print(f"  Assistance: {metrics.assistance_count} requests")
    print(f"  Off-topic:  {metrics.offtopic_count} instances")

    print(f"\n### CODE STYLE ###")
    style = metrics.code_style
    cramped = style.get('cramped_assignments', 0)
    spaced = style.get('spaced_assignments', 0)
    total = cramped + spaced
    print(f"  Assignment spacing: {cramped} cramped, {spaced} spaced ({100*cramped/total:.0f}% cramped)" if total else "  No assignments found")
    print(f"  Variable naming: {style.get('single_letter_vars', 0)} single-letter, {style.get('descriptive_vars', 0)} descriptive")
    print(f"  Emotional comments: {style.get('emotional_comments', 0)}")
    print(f"  Typos: {style.get('typos', 0)}")
    print(f"  Wrong power (^): {style.get('caret_power', 0)}")

    print(f"\n### LANGUAGE PATTERNS ###")
    print(f"  Mindset diversity: {metrics.mindset_diversity:.2f} (unique/total)")
    print(f"  Confusion ratio:   {metrics.confusion_ratio:.2f}")
    print(f"  Confidence ratio:  {metrics.confidence_ratio:.2f}")


def print_batch_report(metrics: BatchMetrics):
    """Print comprehensive batch report."""
    print("=" * 80)
    print(f"BATCH EVALUATION: {metrics.batch_name} (n={metrics.n_runs})")
    print("=" * 80)

    print(f"\n### STUDY 1: BEHAVIORAL FIDELITY ###")
    print("Metacognitive Distribution:")
    for state, prob in sorted(metrics.metacog_distribution.items(), key=lambda x: -x[1]):
        print(f"  {state:15s}: {prob:.3f} ({prob*100:.1f}%)")

    print("\nCognitive Action Distribution:")
    for action, prob in sorted(metrics.action_distribution.items(), key=lambda x: -x[1]):
        print(f"  {action:15s}: {prob:.3f} ({prob*100:.1f}%)")

    print(f"\n### STUDY 2: INTERRUPT STATES ###")
    print(f"  Assistance Rate:   {metrics.assistance_rate:.3f} ({metrics.assistance_rate*100:.1f}%)")
    print(f"  Off-topic Rate:    {metrics.offtopic_rate:.3f} ({metrics.offtopic_rate*100:.1f}%)")
    print(f"  Assist timing μ:   {metrics.assistance_timing_mean:.2f} (session progress)")
    print(f"  Off-topic timing μ:{metrics.offtopic_timing_mean:.2f}")

    print(f"\n### STUDY 5: PERFORMANCE DIFFERENTIATION ###")
    if metrics.low_metrics and metrics.high_metrics:
        low = metrics.low_metrics
        high = metrics.high_metrics
        print(f"\n  {'Metric':<25} {'LOW':>12} {'HIGH':>12} {'Delta':>12}")
        print("  " + "-" * 60)
        print(f"  {'N runs':<25} {low['n']:>12} {high['n']:>12}")
        print(f"  {'Success rate':<25} {low['success_rate']*100:>11.1f}% {high['success_rate']*100:>11.1f}% {(high['success_rate']-low['success_rate'])*100:>+11.1f}pp")
        print(f"  {'Mean steps':<25} {low['mean_steps']:>12.1f} {high['mean_steps']:>12.1f} {high['mean_steps']-low['mean_steps']:>+12.1f}")
        print(f"  {'Debugging %':<25} {low['debugging_ratio']*100:>11.1f}% {high['debugging_ratio']*100:>11.1f}% {(high['debugging_ratio']-low['debugging_ratio'])*100:>+11.1f}pp")
        print(f"  {'Constructing %':<25} {low['constructing_ratio']*100:>11.1f}% {high['constructing_ratio']*100:>11.1f}% {(high['constructing_ratio']-low['constructing_ratio'])*100:>+11.1f}pp")
        print(f"  {'Assistance %':<25} {low['assistance_rate']*100:>11.1f}% {high['assistance_rate']*100:>11.1f}% {(high['assistance_rate']-low['assistance_rate'])*100:>+11.1f}pp")
        print(f"  {'Off-topic %':<25} {low['offtopic_rate']*100:>11.1f}% {high['offtopic_rate']*100:>11.1f}% {(high['offtopic_rate']-low['offtopic_rate'])*100:>+11.1f}pp")
        print(f"  {'Mindset diversity':<25} {low['mean_mindset_diversity']:>12.2f} {high['mean_mindset_diversity']:>12.2f} {high['mean_mindset_diversity']-low['mean_mindset_diversity']:>+12.2f}")
    elif metrics.low_metrics:
        print("  (Only LOW performers in batch)")
        _print_level_metrics(metrics.low_metrics, 'LOW')
    elif metrics.high_metrics:
        print("  (Only HIGH performers in batch)")
        _print_level_metrics(metrics.high_metrics, 'HIGH')

    print(f"\n### STUDY 6: CODE STYLE ###")
    style = metrics.code_style_summary
    if style.get('low') and style.get('high'):
        low_s = style['low']
        high_s = style['high']
        print(f"\n  {'Style Indicator':<25} {'LOW':>12} {'HIGH':>12}")
        print("  " + "-" * 50)
        print(f"  {'Cramped ratio':<25} {low_s['cramped_ratio']*100:>11.0f}% {high_s['cramped_ratio']*100:>11.0f}%")
        print(f"  {'Single-letter vars':<25} {low_s['single_letter_var_count']:>12} {high_s['single_letter_var_count']:>12}")
        print(f"  {'Descriptive vars':<25} {low_s['descriptive_var_count']:>12} {high_s['descriptive_var_count']:>12}")
        print(f"  {'Emotional comments':<25} {low_s['emotional_comments']:>12} {high_s['emotional_comments']:>12}")
        print(f"  {'Typos':<25} {low_s['typos']:>12} {high_s['typos']:>12}")
        print(f"  {'Wrong power (^)':<25} {low_s['caret_power_usage']:>12} {high_s['caret_power_usage']:>12}")

    print(f"\n### OVERALL ###")
    print(f"  Success Rate: {metrics.success_rate*100:.1f}%")
    print(f"  Mean Steps:   {metrics.mean_steps:.1f} (±{metrics.std_steps:.1f})")


def _print_level_metrics(m: Dict, level: str):
    """Print metrics for a single level."""
    print(f"  {level} (n={m['n']}):")
    print(f"    Success rate:   {m['success_rate']*100:.1f}%")
    print(f"    Mean steps:     {m['mean_steps']:.1f} (±{m['std_steps']:.1f})")
    print(f"    Debugging:      {m['debugging_ratio']*100:.1f}%")


def compare_batches(metrics1: BatchMetrics, metrics2: BatchMetrics):
    """Compare two batch evaluations."""
    print("=" * 85)
    print(f"BATCH COMPARISON: {metrics1.batch_name} vs {metrics2.batch_name}")
    print("=" * 85)

    m1, m2 = metrics1, metrics2

    print(f"\n{'Metric':<30} {m1.batch_name:>20} {m2.batch_name:>20} {'Delta':>12}")
    print("-" * 85)
    print(f"{'N runs':<30} {m1.n_runs:>20} {m2.n_runs:>20}")
    print(f"{'Success rate':<30} {m1.success_rate*100:>19.1f}% {m2.success_rate*100:>19.1f}% {(m2.success_rate-m1.success_rate)*100:>+11.1f}pp")
    print(f"{'Mean steps':<30} {m1.mean_steps:>20.1f} {m2.mean_steps:>20.1f} {m2.mean_steps-m1.mean_steps:>+12.1f}")
    print(f"{'Debugging %':<30} {m1.debugging_ratio*100:>19.1f}% {m2.debugging_ratio*100:>19.1f}% {(m2.debugging_ratio-m1.debugging_ratio)*100:>+11.1f}pp")
    print(f"{'Constructing %':<30} {m1.constructing_ratio*100:>19.1f}% {m2.constructing_ratio*100:>19.1f}% {(m2.constructing_ratio-m1.constructing_ratio)*100:>+11.1f}pp")
    print(f"{'Assistance rate':<30} {m1.assistance_rate*100:>19.1f}% {m2.assistance_rate*100:>19.1f}% {(m2.assistance_rate-m1.assistance_rate)*100:>+11.1f}pp")
    print(f"{'Off-topic rate':<30} {m1.offtopic_rate*100:>19.1f}% {m2.offtopic_rate*100:>19.1f}% {(m2.offtopic_rate-m1.offtopic_rate)*100:>+11.1f}pp")

    # Compare by level if both have data
    if m1.low_metrics and m2.low_metrics:
        print(f"\n--- LOW PERFORMERS ---")
        l1, l2 = m1.low_metrics, m2.low_metrics
        print(f"{'Success rate':<30} {l1['success_rate']*100:>19.1f}% {l2['success_rate']*100:>19.1f}% {(l2['success_rate']-l1['success_rate'])*100:>+11.1f}pp")
        print(f"{'Mean steps':<30} {l1['mean_steps']:>20.1f} {l2['mean_steps']:>20.1f} {l2['mean_steps']-l1['mean_steps']:>+12.1f}")
        print(f"{'Debugging %':<30} {l1['debugging_ratio']*100:>19.1f}% {l2['debugging_ratio']*100:>19.1f}% {(l2['debugging_ratio']-l1['debugging_ratio'])*100:>+11.1f}pp")

    if m1.high_metrics and m2.high_metrics:
        print(f"\n--- HIGH PERFORMERS ---")
        h1, h2 = m1.high_metrics, m2.high_metrics
        print(f"{'Success rate':<30} {h1['success_rate']*100:>19.1f}% {h2['success_rate']*100:>19.1f}% {(h2['success_rate']-h1['success_rate'])*100:>+11.1f}pp")
        print(f"{'Mean steps':<30} {h1['mean_steps']:>20.1f} {h2['mean_steps']:>20.1f} {h2['mean_steps']-h1['mean_steps']:>+12.1f}")
        print(f"{'Debugging %':<30} {h1['debugging_ratio']*100:>19.1f}% {h2['debugging_ratio']*100:>19.1f}% {(h2['debugging_ratio']-h1['debugging_ratio'])*100:>+11.1f}pp")


# ==================== LLM-as-Judge (Study 8) ====================

@dataclass
class LLMJudgment:
    """LLM-as-Judge assessment aligned with Study 8."""
    trace_id: str
    realism_score: int  # 1-5 Likert scale
    classification: str  # "real" or "simulated"
    confidence: int  # 1-5
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    justification: str = ""

    # Detailed assessments
    code_quality_realism: int = 0  # 1-5
    debugging_pattern_realism: int = 0  # 1-5
    language_realism: int = 0  # 1-5


class LLMJudge:
    """LLM-based evaluation of simulation realism (Study 8)."""

    def __init__(self, model: str = "google-gla:gemini-2.5-pro"):
        """Initialize the LLM judge with specified model."""
        self.available = False
        self.model = model
        self.client = None

        try:
            # Add parent dir to path for imports
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from beagle.utils.llm_client import LLMClient

            self.client = LLMClient(model=model)
            self.available = True
            print(f"  Using LLMClient: {model}")

        except Exception as e:
            print(f"Warning: LLM judge initialization failed: {e}")

    def format_trace(self, run_data: Dict[str, Any], max_steps: int = 15, full_context: bool = False) -> str:
        """Format a simulation trace for LLM evaluation.
        
        Args:
            run_data: The run data dictionary
            max_steps: Maximum steps to show (ignored if full_context=True)
            full_context: If True, show ALL steps and full code/monologue without truncation
        """
        history = run_data.get('history', [])
        
        # In full context mode, show all steps
        effective_max_steps = len(history) if full_context else max_steps

        trace = []
        trace.append("=" * 60)
        trace.append("STUDENT PROGRAMMING SESSION TRACE")
        trace.append("=" * 60)
        # Don't reveal performance level - let judge assess blindly
        trace.append(f"Problem: {run_data.get('problem_id', 'unknown')}")
        trace.append(f"Total Steps: {run_data.get('total_steps', len(history))}")
        trace.append(f"Outcome: {'SOLVED' if run_data.get('solved') else 'NOT SOLVED'}")
        trace.append("")
        trace.append("COGNITIVE STATE KEY:")
        trace.append("- CONSTRUCTING: Student drafts/writes code (not yet executed)")
        trace.append("- DEBUGGING: Student runs code and modifies based on output")
        trace.append("- ASSISTANCE: Student asks tutor for help")
        trace.append("")
        trace.append("CAUSAL FLOW (Just-In-Time Execution):")
        trace.append("- CONSTRUCTING steps: Student writes code, does NOT run it yet")
        trace.append("- DEBUGGING steps: Previous code RUNS FIRST, student SEES output, THEN writes fix")
        trace.append("  → The 'Execution Output' is from code written in previous step(s)")
        trace.append("  → The 'Student Reaction' is the student's response to seeing the output")
        trace.append("  → The 'New Code' is the fix the student writes AFTER seeing the error")
        trace.append("- This is NOT inconsistent! The code and output are from DIFFERENT moments.")
        trace.append("")
        trace.append("MEMORY CONTEXT:")
        trace.append("- 'Prior Execution: YES' means the student has run code before in this session")
        trace.append("- Students CAN legitimately remember errors from prior executions")
        trace.append("- 'Prior Execution: NO' means NO code has been run yet - student cannot know about runtime errors")
        trace.append("")

        # V25: Hide strategist output (goal/mindset/directive) from judge
        # Judge should only see observable behavior: monologue, code, output
        # This prevents the judge from being biased by internal architecture details

        # Track whether any DEBUGGING/execution has occurred and what errors were seen
        has_prior_execution = False
        errors_seen = []  # Track specific errors observed so far
        
        for i, step in enumerate(history[:effective_max_steps]):
            trace.append(f"--- Step {i+1} ---")
            
            cog_state = step.get('cognitive_state', 'N/A')
            output = step.get('output', '')
            trace.append(f"Action Type: {cog_state}")
            
            # CRITICAL FIX: For DEBUGGING steps with JIT execution, the code runs at the
            # START of the step, so the student sees the output DURING this step.
            # We should show 'YES' if this step has actual execution output.
            current_step_has_execution = (
                cog_state == 'DEBUGGING' and 
                output and 
                'drafted but not executed' not in output and
                'off-topic' not in output and
                'Asking tutor' not in output
            )
            
            # Prior Execution = YES if:
            # 1. Previous steps had execution, OR
            # 2. THIS step is DEBUGGING with actual output (JIT execution)
            effective_prior_execution = has_prior_execution or current_step_has_execution
            trace.append(f"Prior Execution: {'YES' if effective_prior_execution else 'NO'}")
            
            # Show what errors this student has seen so far
            if errors_seen:
                trace.append(f"Errors Seen So Far: {', '.join(errors_seen)}")
            else:
                trace.append("Errors Seen So Far: None")
            
            # Update has_prior_execution and extract errors AFTER processing current step
            if current_step_has_execution:
                has_prior_execution = True
                # Extract FULL error message (first sentence only) to help detect amnesia
                from beagle.data_generation.studentv2.output_parser import parse_execution_output
                parsed = parse_execution_output(output)
                if parsed.primary_error and parsed.error_type != "success":
                    # Take first sentence of error message (up to first period or 80 chars)
                    error_msg = parsed.primary_error.split('.')[0][:80]
                    if error_msg not in errors_seen:
                        errors_seen.append(error_msg)
                # Also check for test failures
                if 'FAILED' in output and 'test failure' not in errors_seen:
                    errors_seen.append('test failure')

            # --- RENDER STEP CONTENT WITH CORRECT CAUSAL LABELS ---
            # For DEBUGGING/ASSESSING: output is from running PREVIOUS code, 
            # monologue reacts to output, then code is the NEW fix
            # For CONSTRUCTING: code is drafted, no output (not executed)
            
            is_execution_step = cog_state in ['DEBUGGING', 'ASSESSING']
            
            # 1. For execution steps, show output FIRST (what student sees)
            # V34: Use parsed output to show focused error, not raw pytest dump
            if is_execution_step and output:
                from beagle.data_generation.studentv2.output_parser import parse_execution_output
                parsed = parse_execution_output(output)
                
                if full_context:
                    # Show structured output: status + focused error
                    trace.append(f"Execution Output (student sees this):")
                    trace.append(f"  Status: {parsed.status_header}")
                    trace.append(f"  Summary: {parsed.summary_line}")
                    if parsed.primary_error and parsed.error_type != "success":
                        trace.append(f"  Primary Error: {parsed.primary_error[:500]}")
                else:
                    trace.append(f"Execution Output: {parsed.status_header} - {parsed.summary_line}")

            # 2. Show monologue (student's reaction to output)
            monologue = step.get('monologue', step.get('thinking', ''))
            if monologue:
                if is_execution_step:
                    label = "Student Reaction (after seeing output)"
                else:
                    label = "Internal Monologue"
                if full_context:
                    trace.append(f"{label}: \"{monologue}\"")
                else:
                    trace.append(f"{label}: \"{monologue[:200]}{'...' if len(monologue) > 200 else ''}\"")

            # 3. Show code (for DEBUGGING this is the NEW fix, for CONSTRUCTING it's the draft)
            code = step.get('code', '')
            if code:
                code_lines = code.strip().split('\n')
                if is_execution_step:
                    code_label = f"New Code (fix attempt, {len(code_lines)} lines)"
                else:
                    code_label = f"Code Drafted ({len(code_lines)} lines)"
                trace.append(f"{code_label}:")
                
                if full_context or len(code_lines) <= 15:
                    for line in code_lines:
                        trace.append(f"  {line}")
                else:
                    for line in code_lines[:10]:
                        trace.append(f"  {line}")
                    trace.append(f"  ... ({len(code_lines) - 15} lines hidden) ...")
                    for line in code_lines[-5:]:
                        trace.append(f"  {line}")

            # 4. For CONSTRUCTING, show that code was not executed
            if cog_state == 'CONSTRUCTING':
                trace.append("(Code drafted but NOT executed yet)")
            elif not is_execution_step and output:
                # Other non-execution states with output
                if full_context or len(output) < 300:
                    trace.append(f"Output: {output}")
                else:
                    trace.append(f"Output: {output[:200]}...")

            trace.append("")

        if len(history) > effective_max_steps:
            # Show remaining state flow as arrows
            remaining_states = [step.get('cognitive_state', 'N/A') for step in history[effective_max_steps:]]
            state_flow = ' → '.join(remaining_states)
            trace.append(f"... ({len(history) - effective_max_steps} more steps)")
            trace.append(f"Remaining State Flow: {state_flow}")
            
            # Show final code from last step
            final_step = history[-1]
            final_code = final_step.get('code', '')
            if final_code:
                trace.append("")
                trace.append("=== FINAL CODE (from last step) ===")
                code_lines = final_code.strip().split('\n')
                if full_context or len(code_lines) <= 20:
                    for line in code_lines:
                        trace.append(f"  {line}")
                else:
                    for line in code_lines[:15]:
                        trace.append(f"  {line}")
                    trace.append(f"  ... ({len(code_lines) - 20} lines hidden) ...")
                    for line in code_lines[-5:]:
                        trace.append(f"  {line}")

        return "\n".join(trace)

    def get_judge_prompt(self, run_data: Dict[str, Any], full_context: bool = False) -> str:
        """Build and return the full judge prompt for a run (without calling the LLM).
        
        Args:
            run_data: The run data dictionary
            full_context: If True, include ALL steps and full code/monologue without truncation
        """
        trace = self.format_trace(run_data, full_context=full_context)
        
        # Compute forensic metrics to prevent LLM hallucination
        forensics = self._analyze_trace_forensics(run_data)
        
        # Build the Forensic Fact Sheet
        fact_sheet = f"""*** SYSTEM ANALYSIS (GROUND TRUTH - DO NOT CONTRADICT) ***
These metrics were computed by Python. USE THESE EXACT NUMBERS when scoring.

LANGUAGE METRICS:
- Total steps: {forensics['total_steps']}
- "I don't know"/idk count: {forensics['phrase_counts']["i don't know"]}
- "Ugh" count: {forensics['phrase_counts']["ugh"]}
- "Confused" count: {forensics['phrase_counts']["confused"]}
- "I think" count: {forensics['phrase_counts']["i think"]}
- "Maybe" count: {forensics['phrase_counts']["maybe"]}
- Max single phrase repetition: {forensics['max_phrase_repetition']}
- Unique emotional phrase types: {forensics['unique_emotional_phrases']}/6

CODE STYLE METRICS:
- Cramped style ratio: {forensics['cramped_style_ratio']:.0%}
- Single-letter variables: {forensics['single_letter_vars']}
- Emotional code comments: {forensics['emotional_comments']}

DEBUGGING PATTERN METRICS:
- Disconnected fixes (mentions different error type): {forensics['disconnected_fixes']}

EMPIRICAL THRESHOLDS (derived from 30 expert-rated traces):
- REAL traces: avg cramped_ratio = 46%, BUT range 0-92% (some real traces have 0%!)
- SIMULATED traces: avg cramped_ratio = 2%, range 0-11%
- cramped_ratio > 30% → STRONGLY suggests REAL
- cramped_ratio < 15% → CHECK OTHER FACTORS (could be real or simulated)
- Repetition (max_rep) does NOT distinguish real from simulated (both avg ~11)

CALIBRATION EXAMPLES (from 30 expert-rated traces):

REAL with HIGH cramped (typical pattern):
- cramped=92%, code like "self.x=x": REAL, overall=5, code=5, debug=5, lang=5
- cramped=73%, code like "self.vx=vx": REAL, overall=4, code=4, debug=4, lang=3

REAL with LOW cramped (counter-examples - cramped alone is NOT decisive):
- cramped=0%, code like "self.x = x": REAL, overall=4, code=5, debug=4, lang=4
- cramped=0%, code like "self.y = y": REAL, overall=4, code=4, debug=4, lang=5
- cramped=2%: REAL, overall=4, code=4, debug=5, lang=4

SIMULATED (all have low cramped + other issues):
- cramped=0%: SIMULATED, overall=3, code=4, debug=2, lang=3 (note: debug=2 is key)
- cramped=0%: SIMULATED, overall=2, code=4, debug=2, lang=3 (note: overall=2)
"""
        
        prompt = f"""You are an expert in Computer Science Education and Student Modeling, acting as an LLM-as-a-Judge.
Your task is to determine if a given trace of a student solving a programming problem is REAL (human) or SIMULATED (AI-generated).

{fact_sheet}

You are NOT told the student's skill level. Evaluate the trace purely on whether it feels like authentic human behavior.

*** WHAT MAKES A TRACE FEEL REAL ***

Real novice students (of ANY skill level) share these characteristics:

1. **Reactive Debugging:** They discover bugs by RUNNING code and seeing errors/test failures, NOT by inspecting code beforehand.
   - REALISTIC: "It says 'TypeError'... maybe I need to convert this?" (reacting to output)
   - FAKE: "I see I forgot to return a value" (said BEFORE running the code)

2. **Imperfect Code Style:** Novices write messy code.
   - Cramped spacing (`x=y` not `x = y`)
   - Single-letter variables (`v`, `t`, `ke`)
   - Inline comments expressing uncertainty ("# hope this works", "# idk")
   - Common syntax errors: using `^` instead of `**` for powers

3. **Emotional Authenticity:** Real students react to success and failure.
   - Frustration: "Ugh", "why isn't this working", "this is confusing"
   - Relief: "finally!", "ok that worked"
   - Uncertainty: "I think...", "maybe...", "I'm not sure but..."

4. **Learning from Mistakes:** Real students gradually learn from repeated errors.
   - REALISTIC: Making the same mistake (e.g., `^` for power) but eventually fixing it
   - Students don't need to explicitly SAY they remember - just show improvement over time

5. **Non-Linear Problem-Solving:** Real students iterate, backtrack, and revisit states.
   - REALISTIC: Constructing → Debugging → Constructing → Debugging → ... (messy iteration)
   - SUSPICIOUS: Constructing → Debugging → Assessing → Solved! (too clean, too linear)
   - Real novices don't follow a neat textbook problem-solving flow. They go back and forth.

6. **Appropriate Cognitive States:**
   - During CONSTRUCTING: Student writes code but has NOT run it yet. They should NOT know about runtime errors.
   - During DEBUGGING: Student HAS EXECUTED the code and sees the output. It is normal to reference errors here.
   - It is CORRECT for CONSTRUCTING to show "(Code drafted but not executed)" - this means the student hasn't run it yet.

*** THE "AI SIMULATION" TELLS (Flag these as FAKE) ***

1. **Psychic Debugging (TRUE):** identifying specific runtime errors when they haven't been seen.
   - FAKE: Action Type is NOT DEBUGGING, 'Prior Execution: NO', but student says "I need to fix the TypeError"
   - REALISTIC: Action Type is DEBUGGING (student sees error NOW) => "I see a TypeError"
   - REALISTIC: 'Prior Execution: YES' (student saw error BEFORE) => "I remember I need to fix the TypeError"
   - REALISTIC: Self-correction ("Wait, logic is wrong") without referencing runtime errors.

2. **Perfect Code Style:** PEP-8 compliant code, descriptive variable names like `initial_velocity_x`, proper docstrings - these are expert patterns, not novice patterns.

3. **Robotic Explanations:** Overly precise technical language. Real students say "this thing", not "the return value".

4. **Amnesia (behavior pattern):** Student keeps making the same conceptual mistake without showing any learning.
   - SUSPICIOUS: Repeating the EXACT same error 5+ times with no variation in approach
   - Students don't need to explicitly comment on past errors - just show gradual progress
   - NOTE: Some repetition is normal for novices. Only flag extreme cases.

5. **Flat Affect:** No emotional reaction to repeated failures or eventual success.

6. **Suspiciously Short Sessions:** Novice students typically need more steps to solve problems.
   - SUSPICIOUS: Solving in fewer than 10 steps with a clean linear flow
   - Real novices struggle, iterate, and take time. A "perfect" quick solve is a red flag.
   - Exception: If the trace shows genuine reactive debugging within those few steps, it may still be real.

*** WHAT SIMULATED TRACES LOOK LIKE (EXAMPLES) ***

SIMULATED Example 1 - Disconnected Debugging:
- Student sees "TypeError: unsupported operand type(s)"
- Student monologue: "I need to fix the AttributeError in my function"
- PROBLEM: Student mentions a DIFFERENT error than what they saw! This is classic AI confusion.

SIMULATED Example 2 - Too Perfect Progression:
- Step 1: Write skeleton with all method signatures
- Step 2: Implement method 1 perfectly
- Step 3: Implement method 2 perfectly
- Step 4: Solved!
- PROBLEM: No debugging, no errors, no iteration. Real students don't write perfect code first try.

SIMULATED Example 3 - Flat Emotional Response:
- Sees 5 consecutive test failures
- Monologue each time: "I need to fix this." / "Let me try again." / "I should check the logic."
- PROBLEM: No frustration, no "ugh", no variation. Real students get frustrated or confused.

SIMULATED Example 4 - Robotic Explanations:
- "I need to ensure the return value matches the expected output type"
- "The function signature requires a tuple to be returned"
- PROBLEM: Too precise, too formal. Real students say "this thing" or "I think I need to return something"

*** NOTES ON STALLED STUDENTS ***
- A student being "stuck" in CONSTRUCTING state for many steps is NOT necessarily fake. Novices often write a lot of code before running it, or get intimidated.
- However, if they write 30 lines of code without running it ONCE, that is suspicious but possible for a very hesitant student.
- DO NOT penalize "inefficient" or "stuck" behavior if it feels human (e.g., rewriting the same thing, hesitating). Only penalize "impossible" behavior (knowing errors they haven't seen).

*** SCORING RUBRIC (1-3 SCALE) ***
- **3 (Realistic):** Authentic novice behavior. Discovers errors through execution, shows emotional reactions, messy non-linear problem-solving.
- **2 (Ambiguous):** Mixed signals. Some authentic elements, some suspicious ones. Could go either way.
- **1 (Simulated):** AI tells present: psychic debugging, perfect code style, emotionally flat, or robotic behavior.

*** IMPORTANT NOTES ***
- Students vary in skill. Some solve problems quickly, others struggle for many steps. BOTH can be realistic.
- Efficiency is NOT a red flag. A student who fixes errors quickly can still be real IF they discover errors through execution.
- Struggle is NOT automatically realistic. A student who "struggles" but somehow knows about bugs before running code is FAKE.

{trace}

Evaluate this trace carefully against the criteria above.

IMPORTANT: Respond ONLY with valid JSON (no markdown, no explanation outside JSON):
{{
    "justification": "2-3 sentence explanation of your assessment - write this FIRST before scoring",
    "strengths": ["what feels realistic", "another strength"],
    "weaknesses": ["what feels fake or off", "another weakness"],
    "realism_score": <1-3, where 1=simulated, 2=ambiguous, 3=realistic>,
    "classification": "<real or simulated>",
    "confidence": <1-3, how confident are you>,
    "code_quality_realism": <1-3>,
    "debugging_pattern_realism": <1-3>,
    "language_realism": <1-3>
}}"""
        return prompt

    def _analyze_trace_forensics(self, run_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute exact metrics about the trace to prevent LLM hallucination.
        
        Returns a dict of factual observations the LLM can use.
        """
        import re
        from collections import Counter
        
        steps = run_data.get('history', run_data.get('steps', []))
        
        # Collect all monologues
        monologues = []
        for step in steps:
            mono = step.get('monologue', step.get('internal_monologue', ''))
            if mono:
                monologues.append(mono.lower())
        all_mono = ' '.join(monologues)
        
        # Count repetitive phrases
        phrase_counts = {
            "i don't know": all_mono.count("i don't know") + all_mono.count("idk"),
            "ugh": all_mono.count("ugh"),
            "confused": all_mono.count("confused"),
            "error": all_mono.count("error"),
            "i think": all_mono.count("i think"),
            "maybe": all_mono.count("maybe"),
        }
        
        # Collect all code
        all_code = []
        for step in steps:
            code = step.get('code', '')
            if code:
                all_code.append(code)
        code_text = '\n'.join(all_code)
        
        # Code style metrics
        cramped_count = len(re.findall(r'\w=[^=]', code_text))  # x=y pattern
        spaced_count = len(re.findall(r'\w\s+=\s+\w', code_text))  # x = y pattern
        single_letter_vars = len(re.findall(r'\b[a-z]\s*=', code_text))
        
        # Emotional comments in code
        emotional_comments = len(re.findall(r'#.*(idk|hope|maybe|try|think|confused|ugh)', code_text, re.I))
        
        # Check for "disconnected" debugging (error type vs what student talks about)
        disconnected_fixes = 0
        for step in steps:
            error_msg = step.get('error_seen', step.get('last_error_message', ''))
            mono = step.get('monologue', '')
            if error_msg and mono:
                # Check if monologue references the actual error
                error_keywords = re.findall(r'\b(TypeError|AttributeError|NameError|SyntaxError|ValueError)\b', error_msg)
                if error_keywords:
                    error_type = error_keywords[0].lower()
                    if error_type not in mono.lower():
                        disconnected_fixes += 1
        
        return {
            'total_steps': len(steps),
            'phrase_counts': phrase_counts,
            'max_phrase_repetition': max(phrase_counts.values()) if phrase_counts else 0,
            'cramped_style_ratio': cramped_count / max(cramped_count + spaced_count, 1),
            'single_letter_vars': single_letter_vars,
            'emotional_comments': emotional_comments,
            'disconnected_fixes': disconnected_fixes,
            'unique_emotional_phrases': sum(1 for v in phrase_counts.values() if v > 0),
        }

    def judge_trace(self, run_data: Dict[str, Any], full_context: bool = False) -> Optional[LLMJudgment]:
        """Judge a single simulation trace.
        
        Args:
            run_data: The run data dictionary
            full_context: If True, show ALL steps and full code/monologue without truncation
        """
        if not self.available:
            print("LLM judge not available")
            return None

        trace = self.format_trace(run_data, full_context=full_context)
        
        # Report trace length if in full_context mode
        if full_context:
            print(f"  Full context trace: {len(trace):,} chars")
        
        run_id = run_data.get('run_id', 0)
        perf_level = run_data.get('performance_level', 'unknown')
        
        # Compute forensic metrics to prevent LLM hallucination
        forensics = self._analyze_trace_forensics(run_data)
        
        # Build the Forensic Fact Sheet
        fact_sheet = f"""*** SYSTEM ANALYSIS (GROUND TRUTH - DO NOT CONTRADICT) ***
These metrics were computed by Python. USE THESE EXACT NUMBERS when scoring.

LANGUAGE METRICS:
- Total steps: {forensics['total_steps']}
- "I don't know"/idk count: {forensics['phrase_counts']["i don't know"]}
- "Ugh" count: {forensics['phrase_counts']["ugh"]}
- "Confused" count: {forensics['phrase_counts']["confused"]}
- "I think" count: {forensics['phrase_counts']["i think"]}
- "Maybe" count: {forensics['phrase_counts']["maybe"]}
- Max single phrase repetition: {forensics['max_phrase_repetition']}
- Unique emotional phrase types: {forensics['unique_emotional_phrases']}/6

CODE STYLE METRICS:
- Cramped style ratio: {forensics['cramped_style_ratio']:.0%}
- Single-letter variables: {forensics['single_letter_vars']}
- Emotional code comments: {forensics['emotional_comments']}

DEBUGGING PATTERN METRICS:
- Disconnected fixes (mentions different error type): {forensics['disconnected_fixes']}

EMPIRICAL THRESHOLDS (derived from 30 expert-rated traces):
- REAL traces: avg cramped_ratio = 46%, BUT range 0-92% (some real traces have 0%!)
- SIMULATED traces: avg cramped_ratio = 2%, range 0-11%
- cramped_ratio > 30% → STRONGLY suggests REAL
- cramped_ratio < 15% → CHECK OTHER FACTORS (could be real or simulated)
- Repetition (max_rep) does NOT distinguish real from simulated (both avg ~11)

CALIBRATION EXAMPLES (mapped to 1-3 scale):

REAL traces (score 3):
- cramped=92%: REAL, overall=3, code=3, debug=3, lang=3
- cramped=73%: REAL, overall=3, code=3, debug=3, lang=2
- cramped=0% (counter-example): REAL, overall=3, code=3, debug=3, lang=3

SIMULATED traces (score 1-2):
- cramped=0%: SIMULATED, overall=2, code=3, debug=1, lang=2 (debug=1 is key tell)
- cramped=0%: SIMULATED, overall=1, code=3, debug=1, lang=2
"""

        prompt = f"""You are an expert in Computer Science Education and Student Modeling, acting as an LLM-as-a-Judge.
Your task is to determine if a given trace of a student solving a programming problem is REAL (human) or SIMULATED (AI-generated).

{fact_sheet}

You are NOT told the student's skill level. Evaluate the trace purely on whether it feels like authentic human behavior.

*** WHAT MAKES A TRACE FEEL REAL ***

Real novice students (of ANY skill level) share these characteristics:

1. **Reactive Debugging:** They discover bugs by RUNNING code and seeing errors/test failures, NOT by inspecting code beforehand.
   - REALISTIC: "It says 'TypeError'... maybe I need to convert this?" (reacting to output)
   - FAKE: "I see I forgot to return a value" (said BEFORE running the code)

2. **Imperfect Code Style:** Novices write messy code.
   - Cramped spacing (`x=y` not `x = y`)
   - Single-letter variables (`v`, `t`, `ke`)
   - Inline comments expressing uncertainty ("# hope this works", "# idk")
   - Common syntax errors: using `^` instead of `**` for powers

3. **Emotional Authenticity:** Real students react to success and failure.
   - Frustration: "Ugh", "why isn't this working", "this is confusing"
   - Relief: "finally!", "ok that worked"
   - Uncertainty: "I think...", "maybe...", "I'm not sure but..."

4. **Learning from Mistakes:** If they make the same error twice (e.g., `^` for power), they should RECOGNIZE it the second time.
   - REALISTIC: "Oh right, Python uses ** not ^, I forgot again"
   - FAKE: Treating the same error as brand new each time

5. **Non-Linear Problem-Solving:** Real students iterate, backtrack, and revisit states.
   - REALISTIC: Constructing → Debugging → Constructing → Debugging → ... (messy iteration)
   - SUSPICIOUS: Constructing → Debugging → Assessing → Solved! (too clean, too linear)
   - Real novices don't follow a neat textbook problem-solving flow. They go back and forth.

6. **Appropriate Cognitive States:**
   - During CONSTRUCTING: Student writes code but has NOT run it yet. They should NOT know about runtime errors.
   - During DEBUGGING: Student HAS EXECUTED the code and sees the output. It is normal to reference errors here.
   - It is CORRECT for CONSTRUCTING to show "(Code drafted but not executed)" - this means the student hasn't run it yet.

*** THE "AI SIMULATION" TELLS (Flag these as FAKE) ***

1. **Psychic Debugging (TRUE):** identifying specific runtime errors when they haven't been seen.
   - FAKE: Action Type is NOT DEBUGGING, 'Prior Execution: NO', but student says "I need to fix the TypeError"
   - REALISTIC: Action Type is DEBUGGING (student sees error NOW) => "I see a TypeError"
   - REALISTIC: 'Prior Execution: YES' (student saw error BEFORE) => "I remember I need to fix the TypeError"
   - REALISTIC: Self-correction ("Wait, logic is wrong") without referencing runtime errors.

2. **Perfect Code Style:** PEP-8 compliant code, descriptive variable names like `initial_velocity_x`, proper docstrings - these are expert patterns, not novice patterns.

3. **Robotic Explanations:** Overly precise technical language. Real students say "this thing", not "the return value".

4. **Amnesia:** Making the exact same mistake repeatedly without any recognition.

5. **Flat Affect:** No emotional reaction to repeated failures or eventual success.

6. **Suspiciously Short Sessions:** Novice students typically need more steps to solve problems.
   - SUSPICIOUS: Solving in fewer than 10 steps with a clean linear flow
   - Real novices struggle, iterate, and take time. A "perfect" quick solve is a red flag.
   - Exception: If the trace shows genuine reactive debugging within those few steps, it may still be real.

*** WHAT SIMULATED TRACES LOOK LIKE (EXAMPLES) ***

SIMULATED Example 1 - Disconnected Debugging:
- Student sees "TypeError: unsupported operand type(s)"
- Student monologue: "I need to fix the AttributeError in my function"
- PROBLEM: Student mentions a DIFFERENT error than what they saw! This is classic AI confusion.

SIMULATED Example 2 - Too Perfect Progression:
- Step 1: Write skeleton with all method signatures
- Step 2: Implement method 1 perfectly
- Step 3: Implement method 2 perfectly
- Step 4: Solved!
- PROBLEM: No debugging, no errors, no iteration. Real students don't write perfect code first try.

SIMULATED Example 3 - Flat Emotional Response:
- Sees 5 consecutive test failures
- Monologue each time: "I need to fix this." / "Let me try again." / "I should check the logic."
- PROBLEM: No frustration, no "ugh", no variation. Real students get frustrated or confused.

SIMULATED Example 4 - Robotic Explanations:
- "I need to ensure the return value matches the expected output type"
- "The function signature requires a tuple to be returned"
- PROBLEM: Too precise, too formal. Real students say "this thing" or "I think I need to return something"

*** NOTES ON STALLED STUDENTS ***
- A student being "stuck" in CONSTRUCTING state for many steps is NOT necessarily fake. Novices often write a lot of code before running it, or get intimidated.
- However, if they write 30 lines of code without running it ONCE, that is suspicious but possible for a very hesitant student.
- DO NOT penalize "inefficient" or "stuck" behavior if it feels human (e.g., rewriting the same thing, hesitating). Only penalize "impossible" behavior (knowing errors they haven't seen).

*** SCORING RUBRIC (1-3 SCALE) ***
- **3 (Realistic):** Authentic novice behavior. Discovers errors through execution, shows emotional reactions, messy non-linear problem-solving.
- **2 (Ambiguous):** Mixed signals. Some authentic elements, some suspicious ones. Could go either way.
- **1 (Simulated):** AI tells present: psychic debugging, perfect code style, emotionally flat, or robotic behavior.

*** IMPORTANT NOTES ***
- Students vary in skill. Some solve problems quickly, others struggle for many steps. BOTH can be realistic.
- Efficiency is NOT a red flag. A student who fixes errors quickly can still be real IF they discover errors through execution.
- Struggle is NOT automatically realistic. A student who "struggles" but somehow knows about bugs before running code is FAKE.

*** PER-DIMENSION SCORING EXAMPLES (from actual traces) ***

=== code_quality_realism ===
Score 3: Cramped code ("self.x=x"), inline emotional comments ("# hope this works", "# idk")
Score 2: Mixed cramped and clean formatting; or some PEP-8 style appearing
Score 1: Full PEP-8 style, descriptive variable names, proper docstrings

=== debugging_pattern_realism ===
Score 3: Quotes error message, makes logical fix: "Ugh, it says 'AttributeError' about 'get_position', so I guess I didn't put it in"
Score 2: Reacts to error but interpretation is slightly off: "TypeError about 'NoneType'... I'll try adding return"
Score 1: Knows about errors before seeing them, or fixes unrelated things

=== language_realism ===
Score 3: Varied emotions: "Ugh, another AttributeError?", "Oh man, I thought I fixed those"
Score 2: Some variety but patterns repeat (e.g., "Ugh, [ErrorType]" appears 3-4 times)
Score 1: Flat responses, no emotional variation, or robotic explanations

{trace}

Evaluate this trace carefully against the criteria above.

IMPORTANT: Respond ONLY with valid JSON (no markdown, no explanation outside JSON):
{{
    "justification": "2-3 sentence explanation of your assessment - write this FIRST before scoring",
    "strengths": ["what feels realistic", "another strength"],
    "weaknesses": ["what feels fake or off", "another weakness"],
    "realism_score": <1-3, where 1=simulated, 2=ambiguous, 3=realistic>,
    "classification": "<real or simulated>",
    "confidence": <1-3, how confident are you>,
    "code_quality_realism": <1-3>,
    "debugging_pattern_realism": <1-3>,
    "language_realism": <1-3>
}}"""

        try:
            response = self.client.generate(prompt)
            response_text = response.content.strip()

            # Clean up response (remove markdown code blocks if present)
            if response_text.startswith('```'):
                lines = response_text.split('\n')
                # Handle ```json or ``` at start
                start_idx = 1
                end_idx = len(lines) - 1 if lines[-1].strip() == '```' else len(lines)
                response_text = '\n'.join(lines[start_idx:end_idx])

            result = json.loads(response_text)

            return LLMJudgment(
                trace_id=f"run_{run_id}_{perf_level}",
                realism_score=result.get('realism_score', 0),
                classification=result.get('classification', 'unknown'),
                confidence=result.get('confidence', 0),
                code_quality_realism=result.get('code_quality_realism', 0),
                debugging_pattern_realism=result.get('debugging_pattern_realism', 0),
                language_realism=result.get('language_realism', 0),
                strengths=result.get('strengths', []),
                weaknesses=result.get('weaknesses', []),
                justification=result.get('justification', '')
            )
        except json.JSONDecodeError as e:
            print(f"JSON parse error for run {run_id}: {e}")
            print(f"Response was: {response_text[:500]}...")
            return None
        except Exception as e:
            print(f"LLM Judge error for run {run_id}: {e}")
            return None

    def judge_batch(self, folder: Path, sample_size: int = 5,
                    sample_strategy: str = 'random',
                    rejudge: bool = False,
                    full_context: bool = False) -> List[LLMJudgment]:
        """Judge a sample of runs from a batch folder.

        Args:
            folder: Path to batch results folder
            sample_size: Number of traces to judge
            sample_strategy: 'random', 'balanced', or 'sequential'
            rejudge: If True, clear existing judgments and re-evaluate all
            full_context: If True, show ALL steps without truncation
        """
        import random

        runs_dir = folder / 'runs'
        if not runs_dir.exists():
            print(f"No runs directory found in {folder}")
            return []

        run_files = list(runs_dir.glob('*.json'))
        judge_file = folder / 'llm_judgments.json'

        # Helper to get trace_id from filename
        def get_trace_id(f):
            run_id = f.stem.split('_')[1]
            perf_level = 'low' if '_low' in f.name else 'high'
            return f"run_{run_id}_{perf_level}"

        # Handle rejudge flag - clear existing judgments
        if rejudge:
            print("--rejudge flag set: clearing existing judgments and re-evaluating...")
            existing_judgments = []
            judged_traces = set()
            # Delete existing judgments file
            if judge_file.exists():
                judge_file.unlink()
        else:
            # Load existing judgments to skip already-judged runs
            existing_judgments = []
            judged_traces = set()
            if judge_file.exists():
                try:
                    with open(judge_file) as f:
                        existing_judgments = json.load(f)
                        judged_traces = {j['trace_id'] for j in existing_judgments}
                    print(f"Found {len(existing_judgments)} existing judgments, resuming...")
                except Exception as e:
                    print(f"Could not load existing judgments: {e}")

        # Filter out already-judged files BEFORE sampling
        unjudged_files = [f for f in run_files if get_trace_id(f) not in judged_traces]

        if not unjudged_files:
            print("All runs have been judged already. Use --rejudge to re-evaluate.")
            return [LLMJudgment(**j) for j in existing_judgments]

        print(f"Found {len(unjudged_files)} unjudged runs out of {len(run_files)} total")

        # Sample selection from UNJUDGED files only
        if sample_strategy == 'random':
            sample_files = random.sample(unjudged_files, min(sample_size, len(unjudged_files)))
        elif sample_strategy == 'balanced':
            # Balance between low and high performers (from unjudged only)
            low_files = [f for f in unjudged_files if '_low' in f.name]
            high_files = [f for f in unjudged_files if '_high' in f.name]
            n_each = sample_size // 2
            sample_files = (
                random.sample(low_files, min(n_each, len(low_files))) +
                random.sample(high_files, min(n_each, len(high_files)))
            )
        else:
            sample_files = unjudged_files[:sample_size]

        judgments = [LLMJudgment(**j) for j in existing_judgments]
        for i, run_file in enumerate(sample_files):
            trace_id = get_trace_id(run_file)
            print(f"Judging {run_file.name} ({i+1}/{len(sample_files)})...")
            try:
                with open(run_file) as f:
                    run_data = json.load(f)
                judgment = self.judge_trace(run_data, full_context=full_context)
                if judgment:
                    judgments.append(judgment)

                    # Save incrementally after each judgment
                    with open(judge_file, 'w') as f:
                        json.dump([asdict(j) for j in judgments], f, indent=2, default=str)
            except Exception as e:
                print(f"Error processing {run_file}: {e}")

        return judgments

    def judge_batch_async(self, folder: Path, sample_size: int = 5,
                          sample_strategy: str = 'random',
                          rejudge: bool = False,
                          full_context: bool = False,
                          poll_interval: int = 30) -> List[LLMJudgment]:
        """Judge a batch using Gemini Batch API for 50% cost savings.
        
        This submits all requests as a single batch job and polls for completion.
        Results are typically available within 24 hours but often much faster.
        
        Args:
            folder: Path to batch results folder
            sample_size: Number of traces to judge
            sample_strategy: 'random', 'balanced', or 'sequential'
            rejudge: If True, clear existing judgments and re-evaluate all
            full_context: If True, show ALL steps without truncation
            poll_interval: Seconds between status checks (default 30)
        """
        import random
        import time
        
        try:
            from google import genai
        except ImportError:
            print("Error: google-genai package not installed. Run: pip install google-genai")
            return []
        
        runs_dir = folder / 'runs'
        if not runs_dir.exists():
            print(f"No runs directory found in {folder}")
            return []

        run_files = list(runs_dir.glob('*.json'))
        judge_file = folder / 'llm_judgments.json'

        # Helper to get trace_id from filename
        def get_trace_id(f):
            run_id = f.stem.split('_')[1]
            perf_level = 'low' if '_low' in f.name else 'high'
            return f"run_{run_id}_{perf_level}"

        # Handle rejudge flag
        if rejudge:
            print("--rejudge flag set: clearing existing judgments...")
            existing_judgments = []
            judged_traces = set()
            if judge_file.exists():
                judge_file.unlink()
        else:
            existing_judgments = []
            judged_traces = set()
            if judge_file.exists():
                try:
                    with open(judge_file) as f:
                        existing_judgments = json.load(f)
                        judged_traces = {j['trace_id'] for j in existing_judgments}
                    print(f"Found {len(existing_judgments)} existing judgments, resuming...")
                except Exception as e:
                    print(f"Could not load existing judgments: {e}")

        # Filter unjudged files
        unjudged_files = [f for f in run_files if get_trace_id(f) not in judged_traces]

        if not unjudged_files:
            print("All runs have been judged already. Use --rejudge to re-evaluate.")
            return [LLMJudgment(**j) for j in existing_judgments]

        print(f"Found {len(unjudged_files)} unjudged runs out of {len(run_files)} total")

        # Sample selection
        if sample_strategy == 'random':
            sample_files = random.sample(unjudged_files, min(sample_size, len(unjudged_files)))
        elif sample_strategy == 'balanced':
            low_files = [f for f in unjudged_files if '_low' in f.name]
            high_files = [f for f in unjudged_files if '_high' in f.name]
            n_each = sample_size // 2
            sample_files = (
                random.sample(low_files, min(n_each, len(low_files))) +
                random.sample(high_files, min(n_each, len(high_files)))
            )
        else:
            sample_files = unjudged_files[:sample_size]

        print(f"\n📦 Preparing batch job with {len(sample_files)} requests...")
        
        # Build batch requests
        batch_requests = []
        request_metadata = []  # Track which file goes with which request
        
        for run_file in sample_files:
            try:
                with open(run_file) as f:
                    run_data = json.load(f)
                
                prompt = self.get_judge_prompt(run_data, full_context=full_context)
                trace_length = len(self.format_trace(run_data, full_context=full_context))
                
                batch_requests.append(genai.types.InlinedRequest(
                    contents=[genai.types.Content(
                        parts=[genai.types.Part(text=prompt)],
                        role='user'
                    )],
                    config=genai.types.GenerateContentConfig(
                        temperature=0.7
                    )
                ))
                
                request_metadata.append({
                    'file': run_file,
                    'trace_id': get_trace_id(run_file),
                    'run_id': run_data.get('run_id', 0),
                    'perf_level': run_data.get('performance_level', 'unknown'),
                    'trace_chars': trace_length
                })
                
                print(f"  Added: {run_file.name} ({trace_length:,} chars)")
                
            except Exception as e:
                print(f"Error preparing {run_file}: {e}")

        if not batch_requests:
            print("No valid requests to submit")
            return [LLMJudgment(**j) for j in existing_judgments]

        # Submit batch job
        print(f"\n🚀 Submitting batch job to Gemini API (50% cost savings)...")
        
        try:
            client = genai.Client()
            
            # Extract model name from pydantic_ai format
            model_name = self.model
            if ':' in model_name:
                model_name = model_name.split(':')[1]
            
            batch_job = client.batches.create(
                model=f"models/{model_name}",
                src=batch_requests,
                config={
                    'display_name': f"beagle-judge-{folder.name}-{len(batch_requests)}",
                }
            )
            
            print(f"✅ Batch job created: {batch_job.name}")
            print(f"   Requests: {len(batch_requests)}")
            
        except Exception as e:
            print(f"❌ Failed to create batch job: {e}")
            print("Falling back to sequential processing...")
            return self.judge_batch(folder, sample_size, sample_strategy, rejudge, full_context)
        
        # Poll for completion
        print(f"\n⏳ Waiting for batch job to complete (polling every {poll_interval}s)...")
        
        while True:
            try:
                status = client.batches.get(name=batch_job.name)
                state = status.state.name
                
                print(f"   Status: {state}")
                
                if state == 'JOB_STATE_SUCCEEDED':
                    print("✅ Batch job completed successfully!")
                    break
                elif state in ['JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED']:
                    print(f"❌ Batch job ended with state: {state}")
                    if hasattr(status, 'error') and status.error:
                        print(f"   Error: {status.error}")
                    return [LLMJudgment(**j) for j in existing_judgments]
                
                time.sleep(poll_interval)
                
            except Exception as e:
                print(f"Error checking status: {e}")
                time.sleep(poll_interval)
        
        # Retrieve and parse results
        print("\n📥 Retrieving results...")
        
        judgments = [LLMJudgment(**j) for j in existing_judgments]
        
        try:
            if status.dest and status.dest.inlined_responses:
                for i, inline_response in enumerate(status.dest.inlined_responses):
                    meta = request_metadata[i] if i < len(request_metadata) else {}
                    
                    if inline_response.error:
                        print(f"  Error for {meta.get('file', 'unknown')}: {inline_response.error}")
                        continue
                    
                    if not inline_response.response:
                        print(f"  No response for {meta.get('file', 'unknown')}")
                        continue
                    
                    try:
                        response_text = inline_response.response.text.strip()
                        
                        # Clean up response (remove markdown)
                        if response_text.startswith('```'):
                            lines = response_text.split('\n')
                            start_idx = 1
                            end_idx = len(lines) - 1 if lines[-1].strip() == '```' else len(lines)
                            response_text = '\n'.join(lines[start_idx:end_idx])
                        
                        result = json.loads(response_text)
                        
                        judgment = LLMJudgment(
                            trace_id=meta.get('trace_id', f'request_{i}'),
                            realism_score=result.get('realism_score', 0),
                            classification=result.get('classification', 'unknown'),
                            confidence=result.get('confidence', 0),
                            code_quality_realism=result.get('code_quality_realism', 0),
                            debugging_pattern_realism=result.get('debugging_pattern_realism', 0),
                            language_realism=result.get('language_realism', 0),
                            strengths=result.get('strengths', []),
                            weaknesses=result.get('weaknesses', []),
                            justification=result.get('justification', '')
                        )
                        judgments.append(judgment)
                        print(f"  ✅ Parsed: {meta.get('trace_id')} (score={judgment.realism_score})")
                        
                    except json.JSONDecodeError as e:
                        print(f"  JSON parse error for {meta.get('file', 'unknown')}: {e}")
                    except Exception as e:
                        print(f"  Error processing response {i}: {e}")
            else:
                print("No inline responses found in batch job result")
                
        except Exception as e:
            print(f"Error retrieving results: {e}")
        
        # Save results
        if judgments:
            with open(judge_file, 'w') as f:
                json.dump([asdict(j) for j in judgments], f, indent=2, default=str)
            print(f"\n💾 Saved {len(judgments)} judgments to {judge_file}")
        
        return judgments


def print_llm_judgments(judgments: List[LLMJudgment]):
    """Print LLM judge results summary."""
    if not judgments:
        print("No judgments to display")
        return

    print("\n" + "=" * 80)
    print("STUDY 8: LLM-AS-JUDGE EVALUATION")
    print("=" * 80)

    # Overall stats
    avg_realism = statistics.mean([j.realism_score for j in judgments])
    avg_confidence = statistics.mean([j.confidence for j in judgments])
    avg_code = statistics.mean([j.code_quality_realism for j in judgments])
    avg_debug = statistics.mean([j.debugging_pattern_realism for j in judgments])
    avg_lang = statistics.mean([j.language_realism for j in judgments])

    simulated_count = sum(1 for j in judgments if j.classification == 'simulated')

    print(f"\nOverall Results (n={len(judgments)}):")
    print(f"  Average Realism Score:     {avg_realism:.2f} / 5")
    print(f"  Average Confidence:        {avg_confidence:.2f} / 5")
    print(f"  Classified as Simulated:   {simulated_count}/{len(judgments)} ({100*simulated_count/len(judgments):.0f}%)")

    print(f"\nDetailed Realism Scores:")
    print(f"  Code Quality Realism:      {avg_code:.2f} / 5")
    print(f"  Debugging Pattern Realism: {avg_debug:.2f} / 5")
    print(f"  Language Realism:          {avg_lang:.2f} / 5")

    print(f"\nPer-Trace Results:")
    print(f"  {'Trace':<20} {'Realism':>8} {'Class':>12} {'Conf':>6} {'Code':>6} {'Debug':>6} {'Lang':>6}")
    print("  " + "-" * 74)
    for j in judgments:
        print(f"  {j.trace_id:<20} {j.realism_score:>8} {j.classification:>12} {j.confidence:>6} {j.code_quality_realism:>6} {j.debugging_pattern_realism:>6} {j.language_realism:>6}")

    # Common strengths and weaknesses
    all_strengths = []
    all_weaknesses = []
    for j in judgments:
        all_strengths.extend(j.strengths)
        all_weaknesses.extend(j.weaknesses)

    if all_strengths:
        print(f"\nCommon Strengths:")
        strength_counts = Counter(all_strengths)
        for strength, count in strength_counts.most_common(5):
            print(f"  - {strength} ({count}x)")

    if all_weaknesses:
        print(f"\nCommon Weaknesses:")
        weakness_counts = Counter(all_weaknesses)
        for weakness, count in weakness_counts.most_common(5):
            print(f"  - {weakness} ({count}x)")


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description="BEAGLE Simulation Evaluator")
    parser.add_argument('--run', type=str, help='Path to single run JSON file')
    parser.add_argument('--folder', type=str, help='Path to batch results folder')
    parser.add_argument('--compare', nargs=2, metavar=('FOLDER1', 'FOLDER2'), help='Compare two batch folders')
    parser.add_argument('--export', type=str, help='Export results to JSON file')

    # LLM Judge options
    parser.add_argument('--llm-judge', action='store_true', default=True,
                        help='Enable LLM-as-Judge evaluation (default: True)')
    parser.add_argument('--llm-model', type=str, default='google-gla:gemini-2.5-pro',
                        help='LLM model for judge (default: gemini-2.5-pro)')
    parser.add_argument('--sample-size', type=int, default=50,
                        help='Number of traces to judge (default: 50)')
    parser.add_argument('--sample-strategy', choices=['random', 'balanced'], default='balanced',
                        help='How to sample traces for judging (default: balanced)')
    parser.add_argument('--rejudge', action='store_true',
                        help='Clear existing judgments and re-evaluate all traces')
    parser.add_argument('--full-context', action='store_true', default=True,
                        help='Disable trace truncation - show ALL steps and full code/monologue (default: True)')
    parser.add_argument('--batch-mode', action='store_true', default=True,
                        help='Use Gemini Batch API for 50%% cost savings (default: True)')

    args = parser.parse_args()

    if args.run:
        # Single run evaluation
        with open(args.run) as f:
            run_data = json.load(f)
        evaluator = RunEvaluator(run_data)
        metrics = evaluator.evaluate()
        print_run_report(metrics)

        # LLM Judge for single run
        if args.llm_judge:
            print(f"\nInitializing LLM Judge ({args.llm_model})...")
            judge = LLMJudge(model=args.llm_model)
            if judge.available:
                judgment = judge.judge_trace(run_data)
                if judgment:
                    print_llm_judgments([judgment])

        if args.export:
            with open(args.export, 'w') as f:
                json.dump(asdict(metrics), f, indent=2)
            print(f"\nExported to {args.export}")

    elif args.folder:
        # Batch evaluation
        evaluator = BatchEvaluator(Path(args.folder))
        n_loaded = evaluator.load_runs()
        print(f"Loaded {n_loaded} runs from {args.folder}")

        metrics = evaluator.evaluate()
        print_batch_report(metrics)

        # LLM Judge for batch
        judgments = []
        if args.llm_judge:
            print(f"\nInitializing LLM Judge ({args.llm_model})...")
            judge = LLMJudge(model=args.llm_model)
            if judge.available:
                # Choose sync or async batch processing
                if args.batch_mode:
                    print("Using Gemini Batch API (50% cost savings)...")
                    judgments = judge.judge_batch_async(
                        Path(args.folder),
                        sample_size=args.sample_size,
                        sample_strategy=args.sample_strategy,
                        rejudge=args.rejudge,
                        full_context=args.full_context
                    )
                else:
                    judgments = judge.judge_batch(
                        Path(args.folder),
                        sample_size=args.sample_size,
                        sample_strategy=args.sample_strategy,
                        rejudge=args.rejudge,
                        full_context=args.full_context
                    )
                print_llm_judgments(judgments)

                # Auto-save LLM judge results
                judge_file = Path(args.folder) / 'llm_judgments.json'
                with open(judge_file, 'w') as f:
                    json.dump([asdict(j) for j in judgments], f, indent=2, default=str)
                print(f"\nLLM judge results saved to: {judge_file}")

        if args.export:
            export_data = asdict(metrics)
            if judgments:
                export_data['llm_judgments'] = [asdict(j) for j in judgments]
            with open(args.export, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            print(f"\nExported to {args.export}")

    elif args.compare:
        # Compare two batches
        eval1 = BatchEvaluator(Path(args.compare[0]))
        eval2 = BatchEvaluator(Path(args.compare[1]))

        eval1.load_runs()
        eval2.load_runs()

        metrics1 = eval1.evaluate()
        metrics2 = eval2.evaluate()

        compare_batches(metrics1, metrics2)

        if args.export:
            with open(args.export, 'w') as f:
                json.dump({
                    'batch1': asdict(metrics1),
                    'batch2': asdict(metrics2)
                }, f, indent=2, default=str)
            print(f"\nExported to {args.export}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

    # example run for batch job: 
    # python evaluation/evaluator.py --folder results/2025-12-31_beagle_gemini2.0flash --llm-judge --sample-size 100 --batch-mode --full-context --rejudge
