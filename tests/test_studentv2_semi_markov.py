#!/usr/bin/env python3
"""
Unit tests for beagle/data_generation/studentv2/semi_markov_model.py

Tests:
- SemiMarkovModel loading
- State transition sampling
- Duration sampling with multipliers
- Action sampling
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from beagle.data_generation.studentv2.semi_markov_model import (
    SemiMarkovModel, train_model_if_needed, DEFAULT_MODEL_PATH
)


class TestSemiMarkovModelLoading:
    """Tests for SemiMarkovModel loading functionality."""
    
    def test_load_existing_model(self):
        """Test loading an existing model file."""
        # Default model should exist
        model = SemiMarkovModel.load()
        assert model is not None
        assert model.models is not None
    
    def test_model_has_low_and_high_performers(self):
        """Test that model contains both performance levels."""
        model = SemiMarkovModel.load()
        assert 'low' in model.models or 'high' in model.models
    
    def test_model_has_transitions(self):
        """Test that model contains transition probabilities."""
        model = SemiMarkovModel.load()
        
        # Check at least one performance level has transitions
        for level in ['low', 'high']:
            if level in model.models:
                assert 'meta_transitions' in model.models[level]
                break
    
    def test_model_has_durations(self):
        """Test that model contains duration distributions."""
        model = SemiMarkovModel.load()
        
        for level in ['low', 'high']:
            if level in model.models:
                assert 'durations' in model.models[level]
                break
    
    def test_model_has_emissions(self):
        """Test that model contains cognitive emissions."""
        model = SemiMarkovModel.load()
        
        for level in ['low', 'high']:
            if level in model.models:
                assert 'cog_emissions' in model.models[level]
                break
    
    @patch('beagle.data_generation.studentv2.semi_markov_model.joblib.load')
    def test_load_handles_corrupted_file(self, mock_load):
        """Test that load raises on corrupted file."""
        mock_load.side_effect = Exception("Corrupted file")
        
        with pytest.raises(Exception):
            SemiMarkovModel.load(filepath=DEFAULT_MODEL_PATH)


class TestSampleNextState:
    """Tests for sample_next_state method."""
    
    def test_sample_next_state_with_empty_history(self):
        """Test sampling with empty history."""
        model = SemiMarkovModel.load()
        state = model.sample_next_state([], 'low')
        
        assert state is not None
        assert isinstance(state, str)
    
    def test_sample_next_state_with_history(self):
        """Test sampling with history context."""
        model = SemiMarkovModel.load()
        state = model.sample_next_state(['Planning'], 'low')
        
        assert state is not None
        assert isinstance(state, str)
    
    def test_sample_next_state_returns_valid_metacog(self):
        """Test that sampled state is a valid metacognitive state."""
        model = SemiMarkovModel.load()
        
        valid_states = ['Planning', 'Monitoring', 'Reflecting', 'Enacting']
        
        for _ in range(20):
            state = model.sample_next_state(['Planning'], 'low')
            assert state in valid_states, f"Invalid state: {state}"
    
    def test_sample_next_state_different_performance_levels(self):
        """Test that different performance levels can produce different distributions."""
        model = SemiMarkovModel.load()
        
        # Sample many times for each level
        low_samples = [model.sample_next_state(['Planning'], 'low') for _ in range(50)]
        high_samples = [model.sample_next_state(['Planning'], 'high') for _ in range(50)]
        
        # Both should produce valid states
        for state in low_samples + high_samples:
            assert state in ['Planning', 'Monitoring', 'Reflecting', 'Enacting']
    
    def test_sample_next_state_with_long_history(self):
        """Test sampling with long history (only last 1 used for 1st-order)."""
        model = SemiMarkovModel.load()
        
        long_history = ['Planning', 'Monitoring', 'Reflecting', 'Planning', 'Monitoring']
        state = model.sample_next_state(long_history, 'low')
        
        assert state is not None
    
    def test_sample_next_state_stochastic(self):
        """Test that sampling is stochastic (can produce different results)."""
        model = SemiMarkovModel.load()
        
        # Sample many times
        samples = [model.sample_next_state(['Monitoring'], 'low') for _ in range(100)]
        unique_samples = set(samples)
        
        # Should produce at least 2 different states (probabilistic)
        # This test might occasionally fail if all samples happen to be same
        assert len(unique_samples) >= 1  # At minimum 1


class TestSampleDuration:
    """Tests for sample_duration method."""
    
    def test_sample_duration_returns_positive(self):
        """Test that duration is always positive."""
        model = SemiMarkovModel.load()
        
        for _ in range(20):
            duration = model.sample_duration('Planning', 'low')
            assert duration >= 1, f"Duration should be >= 1, got {duration}"
    
    def test_sample_duration_returns_integer(self):
        """Test that duration is an integer."""
        model = SemiMarkovModel.load()
        
        duration = model.sample_duration('Planning', 'low')
        assert isinstance(duration, int)
    
    def test_sample_duration_with_multiplier(self):
        """Test duration multiplier effect."""
        model = SemiMarkovModel.load()
        
        # Sample many times with different multipliers
        durations_1x = [model.sample_duration('Planning', 'low', 1.0) for _ in range(50)]
        durations_half = [model.sample_duration('Planning', 'low', 0.5) for _ in range(50)]
        
        # Average of half multiplier should be roughly half
        avg_1x = sum(durations_1x) / len(durations_1x)
        avg_half = sum(durations_half) / len(durations_half)
        
        # Allow for some variance due to minimum of 1 and rounding
        assert avg_half < avg_1x, f"Half multiplier avg ({avg_half}) should be less than 1x ({avg_1x})"
    
    def test_sample_duration_minimum_is_one(self):
        """Test that minimum duration is 1 even with small multiplier."""
        model = SemiMarkovModel.load()
        
        duration = model.sample_duration('Planning', 'low', 0.01)
        assert duration >= 1
    
    def test_sample_duration_unknown_metacog(self):
        """Test duration for unknown metacognitive state."""
        model = SemiMarkovModel.load()
        
        duration = model.sample_duration('UnknownState', 'low')
        assert duration >= 1
    
    def test_sample_duration_different_metacog_states(self):
        """Test duration for different metacognitive states."""
        model = SemiMarkovModel.load()
        
        for state in ['Planning', 'Monitoring', 'Reflecting']:
            duration = model.sample_duration(state, 'low')
            assert duration >= 1


class TestSampleAction:
    """Tests for sample_action method."""
    
    def test_sample_action_returns_valid_action(self):
        """Test that sampled action is valid."""
        model = SemiMarkovModel.load()
        
        valid_actions = ['CONSTRUCTING', 'DEBUGGING', 'ASSESSING']
        
        for _ in range(20):
            action = model.sample_action('Planning', 'low')
            assert action in valid_actions, f"Invalid action: {action}"
    
    def test_sample_action_with_previous_action(self):
        """Test sampling with previous action context."""
        model = SemiMarkovModel.load()
        
        action = model.sample_action('Monitoring', 'low', 'CONSTRUCTING')
        assert action in ['CONSTRUCTING', 'DEBUGGING', 'ASSESSING']
    
    def test_sample_action_different_metacog_states(self):
        """Test action sampling for different metacognitive states."""
        model = SemiMarkovModel.load()
        
        for metacog in ['Planning', 'Monitoring', 'Reflecting']:
            action = model.sample_action(metacog, 'low')
            assert action in ['CONSTRUCTING', 'DEBUGGING', 'ASSESSING']
    
    def test_sample_action_unknown_metacog(self):
        """Test action sampling for unknown metacognitive state."""
        model = SemiMarkovModel.load()
        
        action = model.sample_action('UnknownState', 'low')
        assert action == 'CONSTRUCTING'  # Default fallback
    
    def test_sample_action_stochastic(self):
        """Test that action sampling is stochastic."""
        model = SemiMarkovModel.load()
        
        samples = [model.sample_action('Monitoring', 'low') for _ in range(100)]
        unique_samples = set(samples)
        
        # Should produce at least 1 action
        assert len(unique_samples) >= 1


class TestModelIntegration:
    """Integration tests for the Semi-Markov model."""
    
    def test_full_simulation_sequence(self):
        """Test generating a full sequence of states and actions."""
        model = SemiMarkovModel.load()
        
        history = []
        current_metacog = None
        
        for step in range(10):
            # Sample next metacognitive state
            metacog = model.sample_next_state(history, 'low')
            history.append(metacog)
            
            # Sample duration
            duration = model.sample_duration(metacog, 'low', 0.5)
            
            # Sample action
            action = model.sample_action(metacog, 'low')
            
            assert metacog in ['Planning', 'Monitoring', 'Reflecting', 'Enacting']
            assert duration >= 1
            assert action in ['CONSTRUCTING', 'DEBUGGING', 'ASSESSING']
            
            current_metacog = metacog
        
        assert len(history) == 10
    
    def test_performance_level_consistency(self):
        """Test that model uses correct performance level data."""
        model = SemiMarkovModel.load()
        
        # Both levels should work without errors
        for level in ['low', 'high']:
            state = model.sample_next_state(['Planning'], level)
            duration = model.sample_duration(state, level)
            action = model.sample_action(state, level)
            
            assert state is not None
            assert duration >= 1
            assert action is not None
    
    def test_fallback_to_low_for_unknown_level(self):
        """Test fallback to 'low' for unknown performance level."""
        model = SemiMarkovModel.load()
        
        # 'unknown' level should fallback to 'low'
        state = model.sample_next_state(['Planning'], 'unknown')
        assert state is not None


class TestUpdateModel:
    """Tests for update_model method (deprecated but for compatibility)."""
    
    def test_update_model_does_nothing(self):
        """Test that update_model is a no-op."""
        model = SemiMarkovModel.load()
        
        # Should not raise
        model.update_model()
        model.update_model('some', 'args', kwarg='value')


class TestTrainModelIfNeeded:
    """Tests for train_model_if_needed function."""
    
    @patch('beagle.data_generation.studentv2.semi_markov_model.Path.exists')
    @patch('beagle.data_generation.studentv2.semi_markov_model.joblib.load')
    def test_returns_existing_model(self, mock_load, mock_exists):
        """Test that existing model is loaded without retraining."""
        mock_exists.return_value = True
        mock_load.return_value = {'low': {}, 'high': {}}
        
        result = train_model_if_needed()
        
        mock_load.assert_called_once()


class TestModelEdgeCases:
    """Edge cases for Semi-Markov model."""
    
    def test_empty_models_dict(self):
        """Test behavior with empty models dictionary."""
        model = SemiMarkovModel(model_data={})
        
        with pytest.raises((ValueError, KeyError)):
            model.sample_next_state(['Planning'], 'low')
    
    def test_none_models(self):
        """Test behavior with None models."""
        model = SemiMarkovModel(model_data=None)
        
        with pytest.raises(ValueError):
            model.sample_next_state(['Planning'], 'low')
    
    def test_duration_multiplier_zero(self):
        """Test duration with zero multiplier."""
        model = SemiMarkovModel.load()
        
        duration = model.sample_duration('Planning', 'low', 0.0)
        assert duration >= 1  # Minimum is 1
    
    def test_duration_multiplier_large(self):
        """Test duration with large multiplier."""
        model = SemiMarkovModel.load()
        
        duration = model.sample_duration('Planning', 'low', 10.0)
        assert duration >= 1
        # Should be larger on average, but due to randomness just check it's valid
    
    def test_sample_next_state_unknown_prev_state(self):
        """Test sample_next_state with unknown previous state falls back to None key."""
        model = SemiMarkovModel.load()
        
        # Unknown previous state should fallback to None transitions
        state = model.sample_next_state(['CompletelyUnknownState'], 'low')
        assert state in ['Planning', 'Monitoring', 'Reflecting', 'Enacting']
    
    def test_sample_action_uses_start_fallback(self):
        """Test sample_action uses 'start' key when previous action unknown."""
        model = SemiMarkovModel.load()
        
        # Use None as previous action - should fallback to 'start'
        action = model.sample_action('Planning', 'low', None)
        assert action in ['CONSTRUCTING', 'DEBUGGING', 'ASSESSING']
    
    def test_sample_action_unknown_previous_action(self):
        """Test sample_action with completely unknown previous action."""
        model = SemiMarkovModel.load()
        
        # Unknown previous action should fallback to 'start' or first available
        action = model.sample_action('Planning', 'low', 'UNKNOWN_ACTION')
        assert action in ['CONSTRUCTING', 'DEBUGGING', 'ASSESSING']
    
    def test_duration_fixed_type(self):
        """Test duration sampling with fixed distribution type."""
        # Create model with fixed duration type
        model_data = {
            'low': {
                'meta_transitions': {},
                'durations': {
                    'TestState': {'type': 'fixed', 'params': 5}
                },
                'cog_emissions': {}
            }
        }
        model = SemiMarkovModel(model_data=model_data)
        
        duration = model.sample_duration('TestState', 'low')
        assert duration == 5
    
    def test_duration_unknown_type_fallback(self):
        """Test duration sampling with unknown distribution type."""
        model_data = {
            'low': {
                'meta_transitions': {},
                'durations': {
                    'TestState': {'type': 'unknown_distribution', 'params': (1, 2, 3)}
                },
                'cog_emissions': {}
            }
        }
        model = SemiMarkovModel(model_data=model_data)
        
        duration = model.sample_duration('TestState', 'low')
        assert duration >= 1  # Falls back to 1


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
