"""
Utility to load student performance scores from LAK24 data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict


def load_scores(score_path: Path, threshold: float = 0.8) -> Dict[str, str]:
    """
    Load scores and determine performance level for each group.

    Args:
        score_path: Path to score.csv
        threshold: Score threshold for high/low classification (default 0.8)

    Returns:
        Dict mapping group_id (e.g., 'g2') to 'high' or 'low'
    """
    scores_df = pd.read_csv(score_path)
    group_perf = {}

    for _, row in scores_df.iterrows():
        try:
            score = float(row['phy'])
            if np.isnan(score):
                continue
            group_id = f"g{int(row['group'])}"
            group_perf[group_id] = 'high' if score > threshold else 'low'
        except (ValueError, KeyError):
            continue

    return group_perf


if __name__ == "__main__":
    DATA_DIR = Path("data/lak24")
    group_perf = load_scores(DATA_DIR / "score.csv")

    high_groups = [g for g, p in group_perf.items() if p == 'high']
    low_groups = [g for g, p in group_perf.items() if p == 'low']

    print(f"High Performers ({len(high_groups)}): {sorted(high_groups)}")
    print(f"Low Performers ({len(low_groups)}): {sorted(low_groups)}")
