"""
Train 1st-order Markov model for cognitive action emissions within metacognitive states.

Model: P(next_action | current_metacog, prev_action)

This module trains action transition probabilities conditioned on the current
metacognitive state.

TIME-BASED AGGREGATION:
- Actions within the same time window (default: 30 seconds) are grouped as one "episode"
- This allows DEBUGGING -> DEBUGGING transitions when there's a meaningful time gap
- For g2 which only has minute-level timestamps, we interpolate seconds within each minute

See documentation/action_emission_training_analysis.md for detailed analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
from datetime import datetime

# Mapping for cognitive actions
COGNITIVE_ACTIONS = {'CONSTRUCTING', 'DEBUGGING', 'ASSESSING'}

# Mapping from LAK24 process codes to our simplified metacognitive states
METACOG_MAPPING = {
    'Planning': 'Planning',
    'Planning and Enacting': 'Planning',
    'Planning and Reflecting': 'Reflecting',
    'Reflecting': 'Reflecting',
    'Enacting and Monitoring': 'Monitoring',
    'Enacting':
    'Enacting',  # V24: Added - key differentiator (LOW 30% vs HIGH 18%)
}

# Time threshold in seconds - actions within this window are considered one episode
# Based on analysis: 90th percentile of same-action gaps is ~31s
DEFAULT_TIME_THRESHOLD_SECONDS = 30


def parse_timestamp(time_str: str) -> datetime:
    """
    Parse timestamp string to datetime object.

    Args:
        time_str: Time string like "9/29/2021 12:53:11" or "9/29/21 12:53"

    Returns:
        datetime object or None if parsing fails
    """
    time_str = str(time_str).strip()

    formats = [
        "%m/%d/%Y %H:%M:%S",  # 9/29/2021 12:53:11
        "%m/%d/%y %H:%M:%S",  # 9/29/21 12:53:11
        "%m/%d/%Y %H:%M",  # 9/29/2021 12:53
        "%m/%d/%y %H:%M",  # 9/29/21 12:53
    ]

    for fmt in formats:
        try:
            return datetime.strptime(time_str, fmt)
        except ValueError:
            continue

    return None


def detect_has_seconds(segs_df: pd.DataFrame) -> bool:
    """Check if timestamps have seconds precision."""
    sample_time = str(segs_df['time'].iloc[0])
    return sample_time.count(':') >= 2


def interpolate_seconds_for_minute_data(
    actions_with_times: List[Tuple[str, datetime]]
) -> List[Tuple[str, datetime]]:
    """
    For data without seconds (like g2), interpolate seconds within each minute.

    If we have 5 actions all at 12:53, we spread them as:
    12:53:00, 12:53:12, 12:53:24, 12:53:36, 12:53:48

    Args:
        actions_with_times: List of (action, datetime) tuples

    Returns:
        List with interpolated seconds
    """
    if not actions_with_times:
        return []

    # Group by minute
    minute_groups = defaultdict(list)
    for action, dt in actions_with_times:
        minute_key = dt.replace(second=0, microsecond=0)
        minute_groups[minute_key].append(action)

    # Interpolate seconds within each minute
    result = []
    for minute_dt in sorted(minute_groups.keys()):
        actions = minute_groups[minute_dt]
        n_actions = len(actions)

        for i, action in enumerate(actions):
            # Spread evenly across 60 seconds
            interpolated_second = int((i / max(n_actions, 1)) * 60)
            new_dt = minute_dt.replace(second=interpolated_second)
            result.append((action, new_dt))

    return result


def aggregate_by_time(
    actions_with_times: List[Tuple[str, datetime]],
    threshold_seconds: int = DEFAULT_TIME_THRESHOLD_SECONDS
) -> List[str]:
    """
    Aggregate consecutive same-action entries within time threshold into single episodes.

    Example: If we have (with 30s threshold):
        DEBUGGING at 12:53:00
        DEBUGGING at 12:53:15  (within 30s of previous DEBUGGING -> same episode)
        DEBUGGING at 12:54:00  (> 30s gap -> new episode!)
        CONSTRUCTING at 12:54:30

    We get: [DEBUGGING, DEBUGGING, CONSTRUCTING]

    This preserves same-action transitions when there's a meaningful time gap.

    Args:
        actions_with_times: List of (action, datetime) tuples, sorted by time
        threshold_seconds: Time window in seconds

    Returns:
        Aggregated action sequence
    """
    if not actions_with_times:
        return []

    result = []
    current_action = actions_with_times[0][0]
    last_same_action_time = actions_with_times[0][1]

    for action, time in actions_with_times[1:]:
        if action == current_action:
            # Same action - check if it's a new episode (time gap > threshold)
            time_diff = (time - last_same_action_time).total_seconds()
            if time_diff > threshold_seconds:
                # New episode of same action - record the previous episode
                result.append(current_action)
            # Update last time for this action type
            last_same_action_time = time
        else:
            # Different action - record the previous action and switch
            result.append(current_action)
            current_action = action
            last_same_action_time = time

    # Don't forget the last action
    result.append(current_action)

    return result


def load_group_action_sequences_with_time(
    sl_path: Path,
    segs_path: Path,
    threshold_seconds: int = DEFAULT_TIME_THRESHOLD_SECONDS
) -> List[Tuple[str, List[str]]]:
    """
    Load action sequences grouped by metacognitive state, using time-based aggregation.

    Args:
        sl_path: Path to the SL CSV (contains metacog labels per snum)
        segs_path: Path to the truck-segs CSV (contains actions with timestamps)
        threshold_seconds: Time window for aggregating same-action entries

    Returns:
        List of (metacog_state, action_sequence) tuples
    """
    try:
        sl_df = pd.read_csv(sl_path)
        segs_df = pd.read_csv(segs_path)
    except Exception as e:
        print(f"Error reading files: {e}")
        return []

    # Detect if this file has seconds
    has_seconds = detect_has_seconds(segs_df)

    # Map snum to metacognitive state
    snum_to_meta = {}
    for _, row in sl_df.iterrows():
        process_code = row.get('process-code', '')
        meta_state = METACOG_MAPPING.get(process_code)
        if meta_state:
            snum_to_meta[row['snum']] = meta_state

    # Group actions by snum with timestamps
    segments = defaultdict(list)  # snum -> [(action, datetime), ...]

    for _, row in segs_df.iterrows():
        snum = row.get('snum')
        speaker = str(row.get('speaker', ''))
        time_str = row.get('time')

        # Check if this is a cognitive action
        action = None
        for act in COGNITIVE_ACTIONS:
            if act in speaker:
                action = act
                break

        if action and snum in snum_to_meta:
            dt = parse_timestamp(time_str)
            if dt:
                segments[snum].append((action, dt))

    # Build list of (metacog, actions) pairs
    result = []
    for snum in sorted(segments.keys()):
        if snum not in snum_to_meta:
            continue

        meta = snum_to_meta[snum]
        actions_with_times = segments[snum]

        if not actions_with_times:
            continue

        # Sort by time
        actions_with_times.sort(key=lambda x: x[1])

        # Interpolate seconds if needed (for g2)
        if not has_seconds:
            actions_with_times = interpolate_seconds_for_minute_data(
                actions_with_times
            )

        # Aggregate by time
        aggregated_actions = aggregate_by_time(
            actions_with_times, threshold_seconds
        )

        if aggregated_actions:
            result.append((meta, aggregated_actions))

    return result


def train_action_emissions(
    data_dir: Path,
    group_performance: Dict[str, str],
    threshold_seconds: int = DEFAULT_TIME_THRESHOLD_SECONDS
) -> Dict[str, Dict]:
    """
    Train action emission model: P(next_action | metacog, prev_action)

    Args:
        data_dir: Path to LAK24 data directory
        group_performance: Dict mapping group_id to 'high' or 'low'
        threshold_seconds: Time window for aggregating same-action entries

    Returns:
        Dict with structure:
        {
            'low': {
                metacog: {
                    'start': {action: prob, ...},
                    prev_action: {next_action: prob, ...},
                    ...
                },
                ...
            },
            'high': {...}
        }
    """
    # emissions[perf][metacog][prev_action][next_action] = count
    emissions = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    
    # V34.5: Track session_start separately (very first action of session)
    session_start_counts = defaultdict(Counter)  # perf -> {action: count}

    # Process each group
    for group_id, perf in group_performance.items():
        sl_path = data_dir / f"{group_id}-sl.csv"
        segs_path = data_dir / f"{group_id}-truck-segs.csv"

        if not sl_path.exists() or not segs_path.exists():
            continue

        segments = load_group_action_sequences_with_time(
            sl_path, segs_path, threshold_seconds
        )

        is_first_segment = True
        for meta, actions in segments:
            if not actions:
                continue

            # V34.5: Track session start (first action of first segment)
            if is_first_segment:
                session_start_counts[perf][actions[0]] += 1
                is_first_segment = False

            # Count first action (start transition within segment)
            emissions[perf][meta]['start'][actions[0]] += 1

            # Count subsequent transitions
            for i in range(len(actions) - 1):
                prev_action = actions[i]
                next_action = actions[i + 1]
                emissions[perf][meta][prev_action][next_action] += 1

    # Convert counts to probabilities
    result = {}

    for perf in ['low', 'high']:
        result[perf] = {}
        
        # V34.5: Add session_start probabilities (very first action of session)
        if session_start_counts[perf]:
            total = sum(session_start_counts[perf].values())
            result[perf]['session_start'] = {
                action: count / total
                for action, count in session_start_counts[perf].items()
            }
        else:
            # Fallback: no DEBUGGING on session start
            result[perf]['session_start'] = {'CONSTRUCTING': 0.5, 'ASSESSING': 0.5}

        for meta, transitions in emissions[perf].items():
            result[perf][meta] = {}

            for prev_action, counts in transitions.items():
                total = sum(counts.values())
                result[perf][meta][prev_action] = {
                    action: count / total
                    for action, count in counts.items()
                }

    return result


def print_model_summary(model: Dict[str, Dict]) -> None:
    """Print a summary of the trained model."""
    print("\n" + "=" * 70)
    print("ACTION EMISSION MODEL SUMMARY")
    print("P(next_action | metacog, prev_action)")
    print(
        f"Time-based aggregation: {DEFAULT_TIME_THRESHOLD_SECONDS}s threshold"
    )
    print("=" * 70)

    for perf in ['low', 'high']:
        print(f"\n### {perf.upper()} PERFORMERS ###\n")

        for meta in sorted(model[perf].keys()):
            print(f"Metacognitive State: {meta}")
            print("-" * 50)

            for prev_action in [
                'start', 'CONSTRUCTING', 'DEBUGGING', 'ASSESSING'
            ]:
                if prev_action in model[perf][meta]:
                    probs = model[perf][meta][prev_action]
                    print(f"  Previous: {prev_action}")
                    for action, prob in sorted(
                        probs.items(), key=lambda x: -x[1]
                    ):
                        print(f"    -> {action}: {prob:.1%}")

            print()


if __name__ == "__main__":
    from beagle.data_generation.studentv2.training.load_scores import load_scores

    DATA_DIR = Path("data/lak24")

    print("Training action emission model...")
    group_perf = load_scores(DATA_DIR / "score.csv")
    model = train_action_emissions(DATA_DIR, group_perf)
    print_model_summary(model)
