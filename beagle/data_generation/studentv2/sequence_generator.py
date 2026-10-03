"""
Markov Sequence Generator

Pre-generates metacognitive and cognitive state sequences using the Semi-Markov model.
This generates ONLY the Markov chain (metacog states, cognitive actions, segment durations).
Interrupt decisions (assistance, off-topic) are NOT pre-generated because they depend
on real-time test progress (tests_passed / tests_total).

Benefits:
1. Reproducible behavioral patterns (same seed = same Markov sequence)
2. Analyze Markov dynamics before LLM execution
3. Batch generation for evaluation studies

Usage:
    from beagle.data_generation.studentv2.sequence_generator import SequenceGenerator

    generator = SequenceGenerator()
    sequence = generator.generate(performance_level="low", max_steps=20, seed=42)

    for step in sequence.steps:
        print(f"{step.step}: {step.metacog_state} -> {step.cognitive_action}")
"""

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Literal

from beagle.data_generation.studentv2.semi_markov_model import SemiMarkovModel


@dataclass
class MarkovStep:
    """A single step in the Markov sequence (no interrupts)."""
    step: int
    metacog_state: str  # Planning, Reflecting, Monitoring
    cognitive_action: str  # CONSTRUCTING, DEBUGGING, ASSESSING
    segment_id: int  # Which metacog segment this belongs to
    is_segment_start: bool  # First step of a new segment

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MarkovSequence:
    """A pre-generated Markov sequence (metacog + cognitive states only)."""
    performance_level: str
    max_steps: int
    seed: int
    steps: List[MarkovStep]

    def __post_init__(self):
        self.total_steps = len(self.steps)
        # Count distributions
        metacog_counts = {}
        action_counts = {}
        for step in self.steps:
            metacog_counts[step.metacog_state] = metacog_counts.get(step.metacog_state, 0) + 1
            action_counts[step.cognitive_action] = action_counts.get(step.cognitive_action, 0) + 1

        self.metacog_distribution = {k: v / self.total_steps for k, v in metacog_counts.items()}
        self.action_distribution = {k: v / self.total_steps for k, v in action_counts.items()}

    def to_dict(self) -> dict:
        return {
            "performance_level": self.performance_level,
            "max_steps": self.max_steps,
            "seed": self.seed,
            "total_steps": self.total_steps,
            "metacog_distribution": self.metacog_distribution,
            "action_distribution": self.action_distribution,
            "steps": [s.to_dict() for s in self.steps]
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str):
        with open(path, 'w') as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> 'MarkovSequence':
        with open(path, 'r') as f:
            data = json.load(f)
        steps = [MarkovStep(**s) for s in data['steps']]
        return cls(
            performance_level=data['performance_level'],
            max_steps=data['max_steps'],
            seed=data['seed'],
            steps=steps
        )


class SequenceGenerator:
    """
    Generates Markov sequences using the Semi-Markov model.

    Only generates the Markov chain:
    - Metacognitive states (Planning, Reflecting, Monitoring)
    - Cognitive actions (CONSTRUCTING, DEBUGGING, ASSESSING)
    - Segment boundaries and durations

    Does NOT generate interrupt events (assistance, off-topic) because
    those depend on real-time session_progress = tests_passed / tests_total.
    """

    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = Path(__file__).parent / "semi_markov_model.joblib"
        self.model = SemiMarkovModel.load(str(model_path))

    def generate(
        self,
        performance_level: Literal["low", "high"] = "low",
        max_steps: int = 20,
        seed: Optional[int] = None,
        duration_multiplier: float = 1.0
    ) -> MarkovSequence:
        """
        Generate a Markov sequence (metacog + cognitive states only).

        Args:
            performance_level: "low" or "high" performer
            max_steps: Maximum number of steps to generate
            seed: Random seed for reproducibility
            duration_multiplier: Multiplier for segment durations

        Returns:
            MarkovSequence with all steps
        """
        if seed is not None:
            random.seed(seed)

        steps = []
        segment_id = 0
        segment_step = 0
        segment_duration = 0

        metacog_history = []
        current_metacog = None
        current_action = None

        for step_num in range(1, max_steps + 1):
            is_segment_start = False

            if segment_step >= segment_duration:
                # Start new segment
                is_segment_start = True
                segment_id += 1
                segment_step = 0

                # Sample next metacognitive state (2nd-order Markov)
                current_metacog = self.model.sample_next_state(
                    history=metacog_history,
                    performance_level=performance_level
                )
                metacog_history.append(current_metacog)

                # Sample segment duration
                raw_duration = self.model.sample_duration(
                    current_metacog,
                    performance_level=performance_level,
                    duration_multiplier=duration_multiplier
                )
                segment_duration = max(1, raw_duration)

            segment_step += 1

            # Sample cognitive action
            current_action = self.model.sample_action(
                current_metacog,
                performance_level=performance_level,
                previous_action=current_action
            )

            steps.append(MarkovStep(
                step=step_num,
                metacog_state=current_metacog,
                cognitive_action=current_action,
                segment_id=segment_id,
                is_segment_start=is_segment_start
            ))

        return MarkovSequence(
            performance_level=performance_level,
            max_steps=max_steps,
            seed=seed if seed is not None else -1,
            steps=steps
        )

    def generate_batch(
        self,
        n_sequences: int,
        performance_level: Literal["low", "high"] = "low",
        max_steps: int = 20,
        base_seed: int = 0,
        **kwargs
    ) -> List[MarkovSequence]:
        """Generate multiple sequences for batch analysis."""
        sequences = []
        for i in range(n_sequences):
            seq = self.generate(
                performance_level=performance_level,
                max_steps=max_steps,
                seed=base_seed + i,
                **kwargs
            )
            sequences.append(seq)
        return sequences


def main():
    """Demo: Generate and display a sequence."""
    generator = SequenceGenerator()
    seq = generator.generate(performance_level="low", max_steps=15, seed=42)

    print(f"Generated Markov sequence (seed={seq.seed})")
    print(f"Performance: {seq.performance_level}")
    print(f"Steps: {seq.total_steps}")
    print(f"\nMetacog distribution: {seq.metacog_distribution}")
    print(f"Action distribution: {seq.action_distribution}")
    print(f"\nSequence:")
    for step in seq.steps:
        marker = " [NEW SEG]" if step.is_segment_start else ""
        print(f"  {step.step:2d}: {step.metacog_state:12s} -> {step.cognitive_action}{marker}")


if __name__ == "__main__":
    main()
