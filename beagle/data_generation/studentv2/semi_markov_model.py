import numpy as np
import scipy.stats as stats
import joblib
import logging
import os
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_MODEL_PATH = Path(__file__).parent / "semi_markov_model.joblib"
DEFAULT_DATA_DIR = Path(
    __file__
).parent.parent.parent.parent / "data" / "lak24"


def train_model_if_needed(
    model_path: Path = None, data_dir: Path = None
) -> dict:
    """
    Train the Semi-Markov model if it doesn't exist.
    
    Args:
        model_path: Path to save/load the model (default: semi_markov_model.joblib)
        data_dir: Path to LAK24 data directory (default: data/lak24)
    
    Returns:
        The trained model data dictionary
    """
    model_path = model_path or DEFAULT_MODEL_PATH
    data_dir = data_dir or DEFAULT_DATA_DIR

    if model_path.exists():
        logger.info(f"Model already exists at {model_path}")
        return joblib.load(model_path)

    logger.info(
        f"Model not found at {model_path}, training from {data_dir}..."
    )

    # Import training modules (lazy import to avoid circular dependencies)
    from beagle.data_generation.studentv2.training.train_semi_markov_model import (
        train_semi_markov_model
    )

    # Train and save
    model_data = train_semi_markov_model(data_dir, model_path, verbose=True)

    return model_data


class SemiMarkovModel:
    """
    Semi-Markov Model for Student Behavior.
    
    Models:
    1. State Transitions: P(M_next | M_curr)
    2. State Duration: P(Duration | M_curr) ~ LogNormal
    3. Emissions: P(Action | M_curr)
    """

    def __init__(self, model_data=None):
        self.models = model_data

    @classmethod
    def load(cls, filepath=None, data_dir=None):
        """
        Load the Semi-Markov model from disk, training if necessary.
        
        Args:
            filepath: Path to the model file. If None, uses default path.
            data_dir: Path to training data. If None, uses default path.
                      Only used if model needs to be trained.
        
        Returns:
            SemiMarkovModel instance
        """
        filepath = Path(filepath) if filepath else DEFAULT_MODEL_PATH

        if not filepath.exists():
            logger.warning(f"Model not found at {filepath}, training...")
            model_data = train_model_if_needed(filepath, data_dir)
            return cls(model_data)

        try:
            data = joblib.load(filepath)
            logger.info(f"Loaded Semi-Markov model from {filepath}")
            return cls(data)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def sample_next_state(self, history, performance_level='low'):
        """
        Sample the next metacognitive state given history.
        
        Args:
            history: List of previous metacognitive states (only last 1 used for 1st-order)
            performance_level: 'low' or 'high'
        """
        if not self.models:
            raise ValueError("Model not loaded")

        perf_model = self.models.get(performance_level, self.models['low'])
        transitions = perf_model.get('meta_transitions', {})

        # V24: 1st-order Markov - use just the last state as key
        # Get the previous state (or None if no history)
        prev_state = history[-1] if history else None

        if prev_state in transitions:
            probs_dict = transitions[prev_state]
            states = list(probs_dict.keys())
            probs = list(probs_dict.values())
            return np.random.choice(states, p=probs)
        else:
            # Fallback: use None (start state)
            if None in transitions:
                probs_dict = transitions[None]
                states = list(probs_dict.keys())
                probs = list(probs_dict.values())
                return np.random.choice(states, p=probs)

            # Ultimate fallback: Random from all known states
            all_states = set()
            for next_probs in transitions.values():
                all_states.update(next_probs.keys())

            if not all_states:
                return 'Planning'  # Default

            return np.random.choice(list(all_states))

    def sample_duration(
        self, metacog, performance_level='low', duration_multiplier=1.0
    ):
        """
        Sample duration for the metacognitive state.

        Args:
            metacog: The metacognitive state
            performance_level: 'low' or 'high'
            duration_multiplier: Multiplier applied to sampled duration (e.g., 0.5 = half duration)
                                 Useful for shorter simulations while preserving relative durations.

        Returns:
            Duration as integer (minimum 1)
        """
        perf_model = self.models.get(performance_level, self.models['low'])
        durations = perf_model.get('durations', {})

        if metacog not in durations:
            return 1

        dist_info = durations[metacog]

        if dist_info['type'] == 'gamma':
            shape, loc, scale = dist_info['params']
            duration = stats.gamma.rvs(shape, loc=loc, scale=scale)
        elif dist_info['type'] == 'fixed':
            duration = dist_info['params']
        else:
            duration = 1

        # Apply multiplier and ensure minimum of 1
        scaled_duration = duration * duration_multiplier
        return max(1, int(round(scaled_duration)))

    def sample_action(
        self, metacog, performance_level='low', previous_action=None, is_first_step=False
    ):
        """
        Sample a cognitive action given the metacognitive state and previous action.
        
        Args:
            metacog: The current metacognitive state
            performance_level: 'low' or 'high'
            previous_action: The previous cognitive action (if any)
            is_first_step: If True, use session_start probabilities (no DEBUGGING)
        """
        perf_model = self.models.get(performance_level, self.models['low'])
        emissions = perf_model.get('cog_emissions', {})

        # V34.5: Use session_start for first step of simulation
        if is_first_step:
            if 'session_start' in emissions:
                probs_dict = emissions['session_start']
                actions = list(probs_dict.keys())
                probs = list(probs_dict.values())
                probs = np.array(probs) / np.sum(probs)
                return np.random.choice(actions, p=probs)
            else:
                raise ValueError("No session_start emissions found")

        if metacog not in emissions:
            # Fallback if unknown metacog
            return 'CONSTRUCTING'

        meta_emissions = emissions[metacog]

        # Use previous action if available
        if previous_action and previous_action in meta_emissions:
            probs_dict = meta_emissions[previous_action]
        elif 'start' in meta_emissions:
            probs_dict = meta_emissions['start']
        else:
            # Fallback: pick first available distribution
            probs_dict = list(meta_emissions.values())[0]

        actions = list(probs_dict.keys())
        probs = list(probs_dict.values())

        # Normalize (just in case)
        probs = np.array(probs) / np.sum(probs)

        return np.random.choice(actions, p=probs)

    def update_model(self, *args, **kwargs):
        """Deprecated/Placeholder for compatibility."""
        pass
