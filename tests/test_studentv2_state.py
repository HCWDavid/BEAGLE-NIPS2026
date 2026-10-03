#!/usr/bin/env python3
"""
Unit tests for beagle/data_generation/studentv2/state.py

Tests:
- StudentState dataclass initialization and methods
- p_assistance probability function
- p_offtopic probability function  
- EpisodicMemory dataclass
- Checkpoint save/load functionality
"""

import json
import math
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from beagle.data_generation.studentv2.state import (
    StudentState, EpisodicMemory, p_assistance, p_offtopic
)


class TestPAssistance:
    """Tests for p_assistance probability function."""
    
    def test_p_assistance_peaks_at_mid_session(self):
        """Assistance probability should peak around mid-session (μ=0.5)."""
        # Sample at different progress points
        early = p_assistance(0.1, 'low')
        mid = p_assistance(0.5, 'low')
        late = p_assistance(0.9, 'low')
        
        # Mid-session should be highest
        assert mid > early, "Mid-session should have higher probability than early"
        assert mid > late, "Mid-session should have higher probability than late"
    
    def test_p_assistance_high_performers_seek_more_help(self):
        """High performers seek help more often (15% vs 11.7% peak)."""
        high_mid = p_assistance(0.5, 'high')
        low_mid = p_assistance(0.5, 'low')
        
        assert high_mid > low_mid, "High performers should seek more help"
        # Check approximate peak rates
        assert 0.12 < high_mid < 0.18, f"High peak should be ~15%, got {high_mid}"
        assert 0.09 < low_mid < 0.15, f"Low peak should be ~11.7%, got {low_mid}"
    
    def test_p_assistance_returns_float(self):
        """Should return a float probability."""
        result = p_assistance(0.5, 'low')
        assert isinstance(result, float)
        assert 0 <= result <= 1, "Probability should be in [0, 1]"
    
    def test_p_assistance_at_boundaries(self):
        """Test at session boundaries (0.0 and 1.0)."""
        start = p_assistance(0.0, 'low')
        end = p_assistance(1.0, 'low')
        
        assert start >= 0
        assert end >= 0
        # Both should be lower than mid-session
        mid = p_assistance(0.5, 'low')
        assert start < mid
        assert end < mid
    
    def test_p_assistance_gaussian_shape(self):
        """Verify the distribution follows a Gaussian shape."""
        # Sample many points
        probs = [p_assistance(i / 10, 'low') for i in range(11)]
        
        # Should increase to middle then decrease
        for i in range(5):
            assert probs[i] < probs[i + 1], f"Should increase before peak: {i}"
        for i in range(5, 10):
            assert probs[i] > probs[i + 1], f"Should decrease after peak: {i}"


class TestPOfftopic:
    """Tests for p_offtopic probability function."""
    
    def test_p_offtopic_peaks_late_session(self):
        """Off-topic probability should peak late in session (μ=0.73)."""
        early = p_offtopic(0.2, 'low')
        mid = p_offtopic(0.5, 'low')
        late = p_offtopic(0.73, 'low')  # At peak
        very_late = p_offtopic(0.95, 'low')
        
        # Peak around 0.73 should be highest
        assert late > early
        assert late > mid
        assert late >= very_late
    
    def test_p_offtopic_low_performers_go_offtopic_more(self):
        """Low performers go off-topic more often (9.2% vs 3.7% peak)."""
        high_peak = p_offtopic(0.73, 'high')
        low_peak = p_offtopic(0.73, 'low')
        
        assert low_peak > high_peak, "Low performers should go off-topic more"
        # Check approximate peak rates
        assert 0.07 < low_peak < 0.12, f"Low peak should be ~9.2%, got {low_peak}"
        assert 0.02 < high_peak < 0.06, f"High peak should be ~3.7%, got {high_peak}"
    
    def test_p_offtopic_returns_float(self):
        """Should return a float probability."""
        result = p_offtopic(0.5, 'low')
        assert isinstance(result, float)
        assert 0 <= result <= 1
    
    def test_p_offtopic_at_boundaries(self):
        """Test at session boundaries."""
        start = p_offtopic(0.0, 'low')
        end = p_offtopic(1.0, 'low')
        
        assert start >= 0
        assert end >= 0


class TestEpisodicMemory:
    """Tests for EpisodicMemory dataclass."""
    
    def test_episodic_memory_initialization(self):
        """Test basic initialization."""
        memory = EpisodicMemory(
            error_pattern="TypeError: unsupported operand",
            realization="Python uses ** not ^ for exponentiation",
            step_learned=5
        )
        
        assert memory.error_pattern == "TypeError: unsupported operand"
        assert memory.realization == "Python uses ** not ^ for exponentiation"
        assert memory.step_learned == 5
        assert memory.fix_applied == ""  # Default
    
    def test_episodic_memory_with_fix(self):
        """Test initialization with fix_applied."""
        memory = EpisodicMemory(
            error_pattern="NameError: 'math' is not defined",
            realization="Need to import math module",
            step_learned=3,
            fix_applied="import math"
        )
        
        assert memory.fix_applied == "import math"


class TestStudentState:
    """Tests for StudentState dataclass."""
    
    def test_student_state_default_initialization(self):
        """Test default values on initialization."""
        state = StudentState()
        
        assert state.problem_description == ""
        assert state.problem_id == ""
        assert state.required_kcs == []
        assert state.current_code == ""
        assert state.last_output == ""
        assert state.execution_success == False
        assert state.step_count == 0
        assert state.max_steps == 50
        assert state.history == []
        assert state.episodic_memories == []
        assert state.thought_buffer == []
        assert state.monologue_buffer == []
        assert state.THOUGHT_BUFFER_SIZE == 3
        assert state.MONOLOGUE_BUFFER_SIZE == 3
    
    def test_student_state_custom_initialization(self):
        """Test custom values on initialization."""
        state = StudentState(
            problem_description="Test problem",
            problem_id="test_1",
            required_kcs=["KC_A", "KC_B"],
            max_steps=30,
            current_code="print('hello')",
            step_count=5
        )
        
        assert state.problem_description == "Test problem"
        assert state.problem_id == "test_1"
        assert state.required_kcs == ["KC_A", "KC_B"]
        assert state.max_steps == 30
        assert state.current_code == "print('hello')"
        assert state.step_count == 5
    
    def test_student_state_history_tracking(self):
        """Test that history list can be modified."""
        state = StudentState()
        
        assert len(state.history) == 0
        
        state.history.append({
            'step': 1,
            'cognitive_state': 'CONSTRUCTING',
            'code': 'x = 1'
        })
        
        assert len(state.history) == 1
        assert state.history[0]['step'] == 1
    
    def test_student_state_episodic_memory_storage(self):
        """Test episodic memory list."""
        state = StudentState()
        
        memory = EpisodicMemory(
            error_pattern="IndentationError",
            realization="Use consistent spacing",
            step_learned=2
        )
        state.episodic_memories.append(memory)
        
        assert len(state.episodic_memories) == 1
        assert state.episodic_memories[0].error_pattern == "IndentationError"
    
    def test_student_state_thought_buffer(self):
        """Test thought buffer for anti-repetition."""
        state = StudentState()
        
        state.thought_buffer.append("First thought")
        state.thought_buffer.append("Second thought")
        state.thought_buffer.append("Third thought")
        
        assert len(state.thought_buffer) == 3
        
        # Simulate buffer rotation
        if len(state.thought_buffer) >= state.THOUGHT_BUFFER_SIZE:
            state.thought_buffer.pop(0)
        state.thought_buffer.append("Fourth thought")
        
        assert len(state.thought_buffer) == 3
        assert state.thought_buffer[0] == "Second thought"
    
    def test_student_state_force_assistance_steps(self):
        """Test force_assistance_steps for evaluation."""
        state = StudentState(
            force_assistance_steps=[4, 9, 14]
        )
        
        assert state.force_assistance_steps == [4, 9, 14]
        assert 4 in state.force_assistance_steps
        assert 0 not in state.force_assistance_steps
    
    def test_student_state_markov_state_tracking(self):
        """Test metacognitive state tracking for Markov model."""
        state = StudentState(
            current_metacognitive_state="Planning",
            current_cognitive_state="CONSTRUCTING"
        )
        
        assert state.current_metacognitive_state == "Planning"
        assert state.current_cognitive_state == "CONSTRUCTING"
        
        # Track history for 2nd order Markov
        state.metacognitive_history.append("Planning")
        state.metacognitive_history.append("Monitoring")
        
        assert len(state.metacognitive_history) == 2
    
    def test_student_state_semi_markov_segment(self):
        """Test Semi-Markov segment tracking."""
        state = StudentState(
            segment_duration=5,
            segment_step=2
        )
        
        assert state.segment_duration == 5
        assert state.segment_step == 2
        
        # Simulate segment progression
        state.segment_step += 1
        assert state.segment_step == 3
    
    def test_student_state_tutor_interaction(self):
        """Test tutor-related state fields."""
        state = StudentState()
        
        assert state.tutor_response == ""
        assert state.just_received_tutor_help == False
        assert state.pending_tutor_question == ""
        assert state.assistance_count == 0
        
        # Simulate assistance request
        state.pending_tutor_question = "What does this error mean?"
        state.assistance_count += 1
        
        assert state.assistance_count == 1
    
    def test_student_state_test_progress(self):
        """Test test progress tracking."""
        state = StudentState(
            tests_passed=3,
            tests_total=5
        )
        
        session_progress = state.tests_passed / state.tests_total if state.tests_total > 0 else 0
        assert session_progress == 0.6
    
    def test_student_state_reflection_carry_forward(self):
        """Test pending_reflection for ASSESSING to DEBUGGING."""
        state = StudentState()
        
        assert state.pending_reflection == ""
        
        state.pending_reflection = "The output shows a TypeError"
        assert state.pending_reflection == "The output shows a TypeError"
    
    def test_student_state_cached_execution_result(self):
        """Test execution result caching."""
        state = StudentState()
        
        assert state._cached_execution_result is None
        
        mock_result = {'passed': 3, 'total': 5, 'output': 'Test output'}
        state._cached_execution_result = mock_result
        
        assert state._cached_execution_result['passed'] == 3
    
    def test_student_state_bkt_caching(self):
        """Test BKT knowledge state caching."""
        state = StudentState()
        
        assert state.cached_knowledge_state is None
        
        state.cached_knowledge_state = "You correctly applied function definition."
        assert "correctly applied" in state.cached_knowledge_state
    
    def test_student_state_pregenerated_sequence(self):
        """Test pre-generated sequence mode."""
        state = StudentState()
        
        assert state.pregenerated_sequence is None
        assert state.sequence_index == 0
        
        # Simulate setting a sequence
        mock_sequence = [('Planning', 'CONSTRUCTING'), ('Monitoring', 'DEBUGGING')]
        state.pregenerated_sequence = mock_sequence
        state.sequence_index = 1
        
        assert state.pregenerated_sequence is not None
        assert state.sequence_index == 1


class TestStudentStateCheckpoint:
    """Tests for StudentState checkpoint functionality."""
    
    def test_save_checkpoint_creates_file(self):
        """Test that save_checkpoint creates a JSON file."""
        state = StudentState(
            problem_id="test_problem",
            step_count=5,
            max_steps=30,
            current_code="x = 42",
            current_cognitive_state="DEBUGGING",
            current_metacognitive_state="Monitoring"
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"
            state.save_checkpoint(str(checkpoint_path))
            
            assert checkpoint_path.exists()
            
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
            
            assert data['status'] == 'in_progress'
            assert data['step_count'] == 5
            assert data['max_steps'] == 30
            assert data['current_code'] == "x = 42"
            assert data['current_cognitive_state'] == "DEBUGGING"
    
    def test_save_checkpoint_creates_parent_directories(self):
        """Test that save_checkpoint creates parent directories."""
        state = StudentState(step_count=1)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "nested" / "dir" / "checkpoint.json"
            state.save_checkpoint(str(checkpoint_path))
            
            assert checkpoint_path.exists()
    
    def test_save_checkpoint_includes_history(self):
        """Test that checkpoint includes full history."""
        state = StudentState()
        state.history = [
            {'step': 1, 'code': 'x = 1'},
            {'step': 2, 'code': 'x = 2'}
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"
            state.save_checkpoint(str(checkpoint_path))
            
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
            
            assert len(data['history']) == 2
            assert data['history'][0]['step'] == 1
    
    def test_save_checkpoint_overwrites_existing(self):
        """Test that checkpoint overwrites existing file."""
        state = StudentState(step_count=1)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"
            state.save_checkpoint(str(checkpoint_path))
            
            # Update and save again
            state.step_count = 5
            state.save_checkpoint(str(checkpoint_path))
            
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
            
            assert data['step_count'] == 5


class TestStateEdgeCases:
    """Edge cases and boundary tests for state module."""
    
    def test_p_assistance_negative_progress(self):
        """Test p_assistance with negative progress (should still work)."""
        result = p_assistance(-0.1, 'low')
        assert result >= 0
    
    def test_p_assistance_progress_over_one(self):
        """Test p_assistance with progress > 1.0."""
        result = p_assistance(1.5, 'low')
        assert result >= 0
    
    def test_p_offtopic_negative_progress(self):
        """Test p_offtopic with negative progress."""
        result = p_offtopic(-0.1, 'low')
        assert result >= 0
    
    def test_p_offtopic_progress_over_one(self):
        """Test p_offtopic with progress > 1.0."""
        result = p_offtopic(1.5, 'low')
        assert result >= 0
    
    def test_p_assistance_unknown_performance_level(self):
        """Test p_assistance with unknown performance level (defaults to low)."""
        # Should not crash, will use 'low' as default due to condition
        result = p_assistance(0.5, 'unknown')
        low_result = p_assistance(0.5, 'low')
        assert result == low_result  # 'unknown' != 'high' so uses low rate
    
    def test_student_state_empty_required_kcs(self):
        """Test StudentState with empty required_kcs."""
        state = StudentState(required_kcs=[])
        assert state.required_kcs == []
    
    def test_student_state_very_high_step_count(self):
        """Test StudentState with very high step count."""
        state = StudentState(step_count=1000, max_steps=50)
        assert state.step_count == 1000
        # Step count > max_steps is valid (termination is handled elsewhere)
    
    def test_student_state_zero_max_steps(self):
        """Test StudentState with zero max_steps."""
        state = StudentState(max_steps=0)
        assert state.max_steps == 0


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
