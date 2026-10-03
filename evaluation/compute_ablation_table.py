#!/usr/bin/env python3
"""
Compute Ablation Table Metrics

Generates LaTeX table with:
- Steps, D_KL, P_recur, Debug, Lang, Realism

Usage:
    python evaluation/compute_ablation_table.py
"""

import json
from pathlib import Path
import numpy as np
import joblib
from collections import defaultdict

# =============================================================================
# CONFIGURATION
# =============================================================================
RESULTS_BASE = Path(__file__).parent.parent / "results"
MODEL_PATH = Path(__file__).parent.parent / "beagle/data_generation/studentv2/semi_markov_model.joblib"

# Ablation directories to include in table
ABLATIONS = {
    'Full BEAGLE': RESULTS_BASE / '2026-01-10-r2_beagle_gemini2.0flash',
    'No BKT': RESULTS_BASE / '2026-01-11_ablation_no_bkt_gemini2.0flash',
    'No semi-Markov': RESULTS_BASE / '2026-01-17_ablation_no_markov_gemini2.0flash',
    'No Interrupts': RESULTS_BASE / '2026-01-11_ablation_no_interrupts_gemini2.0flash',
    'No $\\Gamma_{\\text{exec}}$': RESULTS_BASE / '2026-01-11_ablation_no_mem_executor_gemini2.0flash',
    'No $\\Gamma_{\\text{strat}}$': RESULTS_BASE / '2026-01-11_ablation_no_mem_strategist_gemini2.0flash',
    'Combined Agent': RESULTS_BASE / '2026-01-15_merged_pipeline_ablation',
}


# =============================================================================
# DATA LOADING
# =============================================================================
def load_reference_distribution() -> dict:
    """Load reference cognitive distribution from combined test dataset (pilot + aied26)."""
    from collections import Counter
    
    # Paths to test datasets
    data_dir = Path(__file__).parent.parent / "data"
    pilot_path = data_dir / "pilot_data" / "pilot_data_30s_aggregated.json"
    aied26_path = data_dir / "aied26" / "aied26_30s_aggregated.json"
    
    counts = Counter()
    
    # Load pilot data
    if pilot_path.exists():
        with open(pilot_path) as f:
            pilot = json.load(f)
        for session in pilot['sessions']:
            for ep in session['episodes']:
                counts[ep['cognitive_state']] += 1
    
    # Load AIED26 data
    if aied26_path.exists():
        with open(aied26_path) as f:
            aied26 = json.load(f)
        for session in aied26['sessions']:
            for ep in session['episodes']:
                counts[ep['cognitive_state']] += 1
    
    total = sum(counts.values())
    if total == 0:
        # Fallback if data not found
        return {'CONSTRUCTING': 0.5439, 'DEBUGGING': 0.4561, 'ASSESSING': 0.0}
    
    # Ensure all 3 states are present (ASSESSING may be 0 in test data)
    result = {s: counts.get(s, 0) / total for s in ['CONSTRUCTING', 'DEBUGGING', 'ASSESSING']}
    return result


def load_runs(path: Path) -> list:
    """Load all run files from results directory."""
    runs = []
    runs_dir = path / 'runs'
    if runs_dir.exists():
        for rf in sorted(runs_dir.glob('*.json')):
            with open(rf) as f:
                runs.append(json.load(f))
    return runs


def load_llm_judgments(path: Path) -> list:
    """Load LLM judge results if available."""
    judgments_file = path / 'llm_judgments.json'
    if judgments_file.exists():
        with open(judgments_file) as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    return []


# =============================================================================
# D_debug REFERENCE (from combined test data)
# =============================================================================
REF_STICKINESS = 0.54  # Debug→Debug rate
REF_DEBUG_RATIO = 0.46  # % debugging steps


# =============================================================================
# METRIC COMPUTATION
# =============================================================================
def compute_stickiness_kl_single(run: dict) -> float:
    """Compute D_debug (compound KL) for a single run."""
    history = run.get('history', [])
    debug_to_debug = 0
    debug_total = 0
    total_steps = 0
    debug_steps = 0
    
    for i in range(len(history)):
        state = history[i].get('cognitive_state', history[i].get('action', ''))
        if state in ['CONSTRUCTING', 'DEBUGGING', 'ASSESSING']:
            total_steps += 1
            if state == 'DEBUGGING':
                debug_steps += 1
        if i < len(history) - 1:
            next_s = history[i + 1].get('cognitive_state', history[i + 1].get('action', ''))
            if state == 'DEBUGGING':
                debug_total += 1
                if next_s == 'DEBUGGING':
                    debug_to_debug += 1
    
    # Stickiness KL
    if debug_total >= 3:
        stickiness = debug_to_debug / debug_total
        p_dd, p_do = stickiness, 1 - stickiness
        q_dd, q_do = REF_STICKINESS, 1 - REF_STICKINESS
        stickiness_kl = 0.0
        if p_dd > 0 and q_dd > 0:
            stickiness_kl += p_dd * np.log(p_dd / q_dd)
        if p_do > 0 and q_do > 0:
            stickiness_kl += p_do * np.log(p_do / q_do)
        stickiness_kl = abs(stickiness_kl)
    else:
        stickiness_kl = 1.0
    
    # Debug ratio KL
    if total_steps > 0:
        debug_ratio = debug_steps / total_steps
        p_d = max(debug_ratio, 0.01)
        p_o = 1 - p_d
        q_d, q_o = REF_DEBUG_RATIO, 1 - REF_DEBUG_RATIO
        ratio_kl = 0.0
        if p_d > 0 and q_d > 0:
            ratio_kl += p_d * np.log(p_d / q_d)
        if p_o > 0 and q_o > 0:
            ratio_kl += p_o * np.log(p_o / q_o)
        ratio_kl = abs(ratio_kl)
    else:
        ratio_kl = 1.0
    
    return 0.5 * stickiness_kl + 0.5 * ratio_kl
def _aggregate_kl(runs: list, ref_dist: dict) -> float:
    """Pool ALL steps across runs into one empirical state distribution and
    compute a single KL vs reference (zero-floor convention).
    """
    from collections import Counter
    counts = Counter()
    total = 0
    for run in runs:
        for step in run.get('history', []):
            s = step.get('cognitive_state', step.get('action', ''))
            if s in ('CONSTRUCTING', 'DEBUGGING', 'ASSESSING'):
                counts[s] += 1
                total += 1
    if total == 0:
        return 0.0
    dist = {s: counts.get(s, 0) / total for s in ('CONSTRUCTING', 'DEBUGGING', 'ASSESSING')}
    kl = 0.0
    for state, q in ref_dist.items():
        p = max(dist.get(state, 0), 1e-10)
        q_val = max(q, 1e-10)
        kl += p * np.log(p / q_val)
    return abs(float(kl))


def _aggregate_debug_kl(runs: list) -> float:
    """Pool ALL transitions across runs and compute one compound D_debug."""
    debug_to_debug = 0
    debug_total = 0
    total_steps = 0
    debug_steps = 0
    for run in runs:
        history = run.get('history', [])
        for i in range(len(history)):
            state = history[i].get('cognitive_state', history[i].get('action', ''))
            if state in ('CONSTRUCTING', 'DEBUGGING', 'ASSESSING'):
                total_steps += 1
                if state == 'DEBUGGING':
                    debug_steps += 1
            if i < len(history) - 1:
                next_s = history[i + 1].get('cognitive_state', history[i + 1].get('action', ''))
                if state == 'DEBUGGING':
                    debug_total += 1
                    if next_s == 'DEBUGGING':
                        debug_to_debug += 1
    if debug_total >= 3:
        s = debug_to_debug / debug_total
        p_dd, p_do = s, 1 - s
        q_dd, q_do = REF_STICKINESS, 1 - REF_STICKINESS
        stickiness_kl = 0.0
        if p_dd > 0 and q_dd > 0:
            stickiness_kl += p_dd * np.log(p_dd / q_dd)
        if p_do > 0 and q_do > 0:
            stickiness_kl += p_do * np.log(p_do / q_do)
        stickiness_kl = abs(stickiness_kl)
    else:
        stickiness_kl = 1.0
    if total_steps > 0:
        d = max(debug_steps / total_steps, 0.01)
        p_d, p_o = d, 1 - d
        q_d, q_o = REF_DEBUG_RATIO, 1 - REF_DEBUG_RATIO
        ratio_kl = 0.0
        if p_d > 0 and q_d > 0:
            ratio_kl += p_d * np.log(p_d / q_d)
        if p_o > 0 and q_o > 0:
            ratio_kl += p_o * np.log(p_o / q_o)
        ratio_kl = abs(ratio_kl)
    else:
        ratio_kl = 1.0
    return 0.5 * stickiness_kl + 0.5 * ratio_kl


def _bootstrap(runs: list, fn, n_boot: int = 2000, seed: int = 42) -> tuple:
    """Generic bootstrap: returns (point_estimate, std)."""
    if not runs:
        return 0.0, 0.0
    point = fn(runs)
    rng = np.random.default_rng(seed)
    n = len(runs)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sample = [runs[i] for i in idx]
        boot.append(fn(sample))
    return float(point), float(np.std(boot))


def compute_kl_divergence(runs: list, ref_dist: dict) -> tuple:
    """Aggregate D_KL with bootstrap-CI std (resampling trajectories).

    The previous per-trajectory mean ± std conflated student-to-student
    heterogeneity with population-level Markov fidelity. This is the
    population-level statistic plus a bootstrap std.
    """
    return _bootstrap(runs, lambda rs: _aggregate_kl(rs, ref_dist))


def compute_debug_kl(runs: list) -> tuple:
    """Aggregate D_debug with bootstrap-CI std."""
    return _bootstrap(runs, _aggregate_debug_kl)


def compute_error_recurrence(runs: list) -> tuple:
    """
    Compute error recurrence rate (stable misconceptions).
    
    Formula: For each (run, error_type) pair, check if that error appears 2+ times.
    Rate = recurrent_pairs / total_pairs_with_errors
    
    Returns: (mean_rate, std_rate) as percentages
    """
    error_types = ['TypeError', 'NameError', 'AttributeError', 'ValueError', 
                   'AssertionError', 'ImportError']
    
    per_run_rates = []
    
    for run in runs:
        history = run.get('history', [])
        
        # Count each error type in this run
        run_errors = defaultdict(int)
        for step in history:
            output = str(step.get('output', ''))
            for err in error_types:
                if err in output:
                    run_errors[err] += 1
                    break
        
        # Compute recurrence rate for this run
        if run_errors:
            total_error_types = len(run_errors)
            recurrent_types = sum(1 for count in run_errors.values() if count >= 2)
            per_run_rates.append(recurrent_types / total_error_types * 100)
    
    if per_run_rates:
        return np.mean(per_run_rates), np.std(per_run_rates)
    return 0.0, 0.0


def extract_llm_judge_scores(judgments: list) -> dict:
    """Extract mean and std scores from LLM judge results."""
    debug_scores = []
    lang_scores = []
    realism_scores = []
    
    for j in judgments:
        if 'debugging_pattern_realism' in j:
            debug_scores.append(j['debugging_pattern_realism'])
        if 'language_realism' in j:
            lang_scores.append(j['language_realism'])
        if 'realism_score' in j:
            realism_scores.append(j['realism_score'])
    
    return {
        'debug_mean': np.mean(debug_scores) if debug_scores else None,
        'debug_std': np.std(debug_scores) if debug_scores else None,
        'lang_mean': np.mean(lang_scores) if lang_scores else None,
        'lang_std': np.std(lang_scores) if lang_scores else None,
        'realism_mean': np.mean(realism_scores) if realism_scores else None,
        'realism_std': np.std(realism_scores) if realism_scores else None,
    }


# =============================================================================
# TABLE GENERATION
# =============================================================================
def generate_latex_table():
    """Generate LaTeX ablation table."""
    # Load reference distribution
    ref_dist = load_reference_distribution()
    
    print("Reference distribution from trained model:")
    for k, v in sorted(ref_dist.items()):
        print(f"  {k}: {v:.1%}")
    print()
    
    # Generate table rows
    rows = []
    for name, path in ABLATIONS.items():
        if not path.exists():
            print(f"Warning: {name} path not found: {path}")
            continue
        
        runs = load_runs(path)
        if not runs:
            print(f"Warning: {name} has no runs")
            continue
        
        # Compute metrics
        steps_list = [r.get('total_steps', 30) for r in runs]
        steps_mean = np.mean(steps_list)
        steps_std = np.std(steps_list)
        kl_mean, kl_std = compute_kl_divergence(runs, ref_dist)
        debug_kl_mean, debug_kl_std = compute_debug_kl(runs)
        recur_mean, recur_std = compute_error_recurrence(runs)
        
        # LLM judge scores
        judgments = load_llm_judgments(path)
        scores = extract_llm_judge_scores(judgments)
        
        rows.append({
            'name': name,
            'steps_mean': steps_mean,
            'steps_std': steps_std,
            'kl_mean': kl_mean,
            'kl_std': kl_std,
            'debug_kl_mean': debug_kl_mean,
            'debug_kl_std': debug_kl_std,
            'recur_mean': recur_mean,
            'recur_std': recur_std,
            'llm_debug_mean': scores['debug_mean'],
            'llm_debug_std': scores['debug_std'],
            'lang_mean': scores['lang_mean'],
            'lang_std': scores['lang_std'],
            'realism_mean': scores['realism_mean'],
            'realism_std': scores['realism_std'],
        })
    
    # No bold/underline highlighting on the ablation table — present plain values
    # so readers can see the raw deltas without visual emphasis.
    def fmt(val, std, decimals=2, suffix=''):
        if val is None:
            return '-'
        if std is None:
            return f"{val:.{decimals}f}{suffix}"
        return f"{val:.{decimals}f}$\\pm${std:.{decimals}f}{suffix}"

    # ------------------------------------------------------------------
    # Emit full LaTeX table (matches user-provided ablation template).
    # ------------------------------------------------------------------
    out: list = []
    out.append(r"\begin{table}[t]")
    out.append(r"\centering")
    out.append(
        r"\caption{Ablation Study ($N=50$, $T=30$, Gemini 2.0 Flash). "
        r"$D_{\mathrm{KL}}$ = behavioral divergence vs.\ combined test data. "
        r"Combined Agent merges Strategist/Executor into a single LLM.}"
    )
    out.append(r"\label{tab:ablation}")
    out.append(r"\footnotesize")
    out.append(r"\setlength{\tabcolsep}{5pt}")
    out.append(r"\begin{tabular}{@{}l c c c c c c@{}}")
    out.append(r"    \toprule")
    out.append(
        r"    \textbf{Variant} & Steps & $D_{\mathrm{KL}}$$\downarrow$ & "
        r"$P_{\text{recur}}$$\uparrow$ & \llmcol{Debug$\uparrow$} & "
        r"\llmcol{Lang$\uparrow$} & \llmcol{Realism$\uparrow$} \\"
    )
    out.append(r"    \midrule")

    for row in rows:
        steps_str = f"{row['steps_mean']:.0f}$\\pm${row['steps_std']:.0f}"
        kl_str     = fmt(row['kl_mean'], row['kl_std'], 2)
        recur_str  = fmt(row['recur_mean'], row['recur_std'], 1, '\\%')
        ldebug_str = fmt(row['llm_debug_mean'], row['llm_debug_std'], 2)
        lang_str   = fmt(row['lang_mean'], row['lang_std'], 2)
        real_str   = fmt(row['realism_mean'], row['realism_std'], 2)
        out.append(
            f"    {row['name']:<25s} & {steps_str} & {kl_str} & {recur_str} & "
            f"{ldebug_str} & {lang_str} & {real_str} \\\\"
        )

    out.append(r"    \bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"\end{table}")

    print("\n".join(out))
    return rows


if __name__ == "__main__":
    generate_latex_table()
