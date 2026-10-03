"""
Train the complete Semi-Markov student model.

This script combines both sub-models:
1. Metacognitive transitions: P(next_metacog | prev2_metacog, prev1_metacog)
2. Action emissions: P(next_action | metacog, prev_action)

The resulting model is saved as a joblib file for use in simulation.
"""

import os
import joblib
from pathlib import Path
from typing import Dict, Any

from .load_scores import load_scores
from .train_metacog_transitions import train_metacog_transitions
from .train_action_emissions import train_action_emissions


def train_semi_markov_model(
    data_dir: Path,
    output_path: Path,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Train the complete Semi-Markov model and save to disk.

    Args:
        data_dir: Path to LAK24 data directory
        output_path: Path to save the trained model
        verbose: Whether to print progress

    Returns:
        The trained model dictionary
    """
    if verbose:
        print("=" * 70)
        print("TRAINING SEMI-MARKOV STUDENT MODEL")
        print("=" * 70)

    # Load performance labels
    if verbose:
        print("\n1. Loading performance scores...")
    group_perf = load_scores(data_dir / "score.csv")

    high_groups = [g for g, p in group_perf.items() if p == 'high']
    low_groups = [g for g, p in group_perf.items() if p == 'low']
    if verbose:
        print(f"   High performers: {len(high_groups)} groups")
        print(f"   Low performers: {len(low_groups)} groups")

    # Train metacognitive transitions
    if verbose:
        print("\n2. Training metacognitive transitions...")
        print("   Model: P(next_metacog | prev2_metacog, prev1_metacog)")
    metacog_model = train_metacog_transitions(data_dir, group_perf)

    # Train action emissions
    if verbose:
        print("\n3. Training action emissions...")
        print("   Model: P(next_action | metacog, prev_action)")
        print("   NOTE: Only counting transitions where action CHANGES")
    action_model = train_action_emissions(data_dir, group_perf)

    # Combine into single model structure
    # Structure matches what SemiMarkovModel.load() expects
    model_data = {}

    for perf in ['low', 'high']:
        model_data[perf] = {
            'meta_transitions': metacog_model[perf]['transitions'],
            'durations': metacog_model[perf]['durations'],
            'cog_emissions': action_model[perf]
        }

    # Save model
    if verbose:
        print(f"\n4. Saving model to {output_path}...")
    os.makedirs(output_path.parent, exist_ok=True)
    joblib.dump(model_data, output_path)

    if verbose:
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print_model_summary(model_data)

    return model_data


def print_model_summary(model: Dict[str, Any]) -> None:
    """Print a summary of the trained model."""

    for perf in ['low', 'high']:
        print(f"\n### {perf.upper()} PERFORMERS ###")

        # Meta transitions
        n_contexts = len(model[perf]['meta_transitions'])
        print(f"\nMetacognitive Transitions: {n_contexts} contexts")

        # Sample a few transitions
        print("  Sample transitions:")
        for i, (history, probs) in enumerate(model[perf]['meta_transitions'].items()):
            if i >= 3:
                break
            top_next = max(probs.items(), key=lambda x: x[1])
            print(f"    {history} -> {top_next[0]} ({top_next[1]:.0%})")

        # Durations
        print(f"\nDuration Models:")
        for meta, dist in model[perf]['durations'].items():
            if dist['type'] == 'gamma':
                shape, loc, scale = dist['params']
                mean = shape * scale + loc
                print(f"    {meta}: mean={mean:.1f} steps")
            else:
                print(f"    {meta}: fixed={dist['params']} steps")

        # Action emissions
        print(f"\nAction Emissions (by metacog state):")
        for meta in sorted(model[perf]['cog_emissions'].keys()):
            emissions = model[perf]['cog_emissions'][meta]
            n_contexts = len(emissions)
            print(f"    {meta}: {n_contexts} action contexts")

            # Show start probabilities
            if 'start' in emissions:
                start_probs = emissions['start']
                top_start = max(start_probs.items(), key=lambda x: x[1])
                print(f"      start -> {top_start[0]} ({top_start[1]:.0%})")


def main():
    """Main entry point for training."""
    DATA_DIR = Path("data/lak24")
    OUTPUT_PATH = Path("beagle/data_generation/studentv2/semi_markov_model.joblib")

    train_semi_markov_model(DATA_DIR, OUTPUT_PATH, verbose=True)


if __name__ == "__main__":
    main()
