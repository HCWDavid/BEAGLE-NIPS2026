#!/usr/bin/env python3
"""
Unit tests for beagle/data_generation/studentv2/student.py

Tests:
- Student class initialization
- Model loading
- solve_problem with mocked graph
- solve_problem_with_sequence 
- BKT and EFI integration
- Profile selection and override
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from typing import List, Optional

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# Helper to create a properly mocked Student
def create_mocked_student(**kwargs):
    """Create a Student with all dependencies mocked."""
    from beagle.data_generation.studentv2.student import Student
    
    with patch.object(Student, 'load_model'):
        with patch('beagle.data_generation.studentv2.student.LLMClient') as MockLLM:
            MockLLM.return_value = MagicMock()
            
            defaults = {
                'performance_level': 'low',
                'model_path': 'dummy_path.joblib',
                'llm_model': 'openai:test-model'
            }
            defaults.update(kwargs)
            
            student = Student(**defaults)
    
    return student


# =============================================================================
# Tests for Student Initialization
# =============================================================================

class TestStudentInitialization:
    """Tests for Student class initialization."""
    
    def test_student_default_initialization(self):
        """Test default initialization with minimal parameters."""
        student = create_mocked_student()
        assert student.performance_level == 'low'
    
    def test_student_high_performer_initialization(self):
        """Test initialization as high performer."""
        student = create_mocked_student(performance_level='high')
        assert student.performance_level == 'high'
    
    
    def test_student_with_duration_multiplier(self):
        """Test initialization with custom duration multiplier."""
        student = create_mocked_student(duration_multiplier=0.5)
        assert student.duration_multiplier == 0.5
    
    def test_student_with_efi_kcs(self):
        """Test initialization with EFI knowledge components."""
        student = create_mocked_student(efi_kcs=['KC_C2_MATH_LIBRARY'])
        assert student.efi_kcs == ['KC_C2_MATH_LIBRARY']
    
    def test_student_with_forced_assistance(self):
        """Test initialization with forced assistance steps."""
        student = create_mocked_student(force_assistance_steps=[4, 9, 14])
        assert student.force_assistance_steps == [4, 9, 14]
    
    def test_student_with_cache_bkt_disabled(self):
        """Test initialization with BKT caching disabled."""
        student = create_mocked_student(cache_bkt=False)
        assert student.cache_bkt == False
    
    def test_student_with_bkt_profile(self):
        """Test initialization with custom BKT profile."""
        student = create_mocked_student(bkt_profile='BELOW_AVERAGE')
        assert student.bkt_profile == 'BELOW_AVERAGE'
    
    def test_student_with_bkt_seed(self):
        """Test initialization with BKT seed for reproducibility."""
        student = create_mocked_student(bkt_seed=42)
        assert student.bkt_seed == 42


class TestStudentLoadModel:
    """Tests for Student.load_model method."""
    
    def test_load_model_creates_markov_model(self):
        """Test that load_model creates markov_model attribute."""
        from beagle.data_generation.studentv2.student import Student
        
        with patch('beagle.data_generation.studentv2.student.SemiMarkovModel.load') as mock_load:
            with patch('beagle.data_generation.studentv2.student.LLMClient') as MockLLM:
                MockLLM.return_value = MagicMock()
                mock_model = MagicMock()
                mock_load.return_value = mock_model
                
                student = Student(
                    performance_level='low',
                    model_path='dummy_path.joblib',
                    llm_model='openai:test-model'
                )
        
        assert student.markov_model == mock_model
    
    def test_load_model_handles_errors(self):
        """Test that load_model propagates errors."""
        from beagle.data_generation.studentv2.student import Student
        
        with patch('beagle.data_generation.studentv2.student.SemiMarkovModel.load') as mock_load:
            with patch('beagle.data_generation.studentv2.student.LLMClient') as MockLLM:
                MockLLM.return_value = MagicMock()
                mock_load.side_effect = Exception("Model corrupted")
                
                with pytest.raises(Exception, match="Model corrupted"):
                    student = Student(
                        performance_level='low',
                        model_path='corrupted.joblib',
                        llm_model='openai:test-model'
                    )
    
    def test_student_without_model_path(self):
        """Test Student initialization without model_path."""
        from beagle.data_generation.studentv2.student import Student
        
        with patch('beagle.data_generation.studentv2.student.LLMClient') as MockLLM:
            MockLLM.return_value = MagicMock()
            
            student = Student(
                performance_level='low',
                llm_model='openai:test-model'
            )
        
        # markov_model should be None if no path provided
        assert student.markov_model is None
    
    def test_solve_problem_without_model_raises(self):
        """Test that solve_problem without model raises RuntimeError."""
        from beagle.data_generation.studentv2.student import Student
        
        with patch('beagle.data_generation.studentv2.student.LLMClient') as MockLLM:
            MockLLM.return_value = MagicMock()
            
            student = Student(
                performance_level='low',
                llm_model='openai:test-model'
            )
        
        with pytest.raises(RuntimeError, match="Markov model not loaded"):
            student.solve_problem(
                problem_description="Test",
                problem_id="test",
                required_kcs=[],
                max_steps=5
            )


class TestStudentSolveProblem:
    """Tests for Student.solve_problem method."""
    
    def test_solve_problem_initializes_state(self):
        """Test that solve_problem initializes StudentState."""
        from beagle.data_generation.studentv2.state import StudentState
        
        student = create_mocked_student()
        
        # Properly mock BKT with get_kc_state returning valid state
        mock_kc_state = MagicMock()
        mock_kc_state.p_known = 0.5  # Must be a float for formatting
        
        student.bkt = MagicMock()
        student.bkt.get_kc_state.return_value = mock_kc_state
        student.ide_oracle = MagicMock()
        student.markov_model = MagicMock()
        student.assessment_oracle = MagicMock()
        
        with patch('beagle.data_generation.studentv2.student.student_graph') as mock_graph:
            mock_graph.run_sync = MagicMock()
            
            with patch('beagle.data_generation.studentv2.student.initialize_student_kcs'):
                history = student.solve_problem(
                    problem_description="Test problem",
                    problem_id="test_1",
                    required_kcs=["KC_A"],
                    max_steps=10
                )
        
        # History should be returned
        assert isinstance(history, list)
    
    def test_solve_problem_with_efi(self):
        """Test solve_problem activates EFI on specified KCs."""
        student = create_mocked_student(efi_kcs=['KC_C2_MATH_LIBRARY'])
        
        # Properly mock BKT with get_kc_state returning valid state
        mock_kc_state = MagicMock()
        mock_kc_state.p_known = 0.3  # Must be a float for formatting
        
        student.bkt = MagicMock()
        student.bkt.get_kc_state.return_value = mock_kc_state
        student.ide_oracle = MagicMock()
        student.markov_model = MagicMock()
        student.assessment_oracle = MagicMock()
        
        with patch('beagle.data_generation.studentv2.student.student_graph') as mock_graph:
            mock_graph.run_sync = MagicMock()
            
            with patch('beagle.data_generation.studentv2.student.initialize_student_kcs'):
                student.solve_problem(
                    problem_description="Test",
                    problem_id="test",
                    required_kcs=["KC_C2_MATH_LIBRARY"],
                    max_steps=5
                )
        
        # BKT.set_efi should have been called
        student.bkt.set_efi.assert_called()
    
    def test_solve_problem_with_checkpoint(self):
        """Test solve_problem saves checkpoints."""
        import tempfile
        
        student = create_mocked_student()
        
        student.bkt = MagicMock()
        student.ide_oracle = MagicMock()
        student.markov_model = MagicMock()
        student.assessment_oracle = MagicMock()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = f"{tmpdir}/checkpoint.json"
            
            with patch('beagle.data_generation.studentv2.student.student_graph') as mock_graph:
                mock_graph.run_sync = MagicMock()
                
                with patch('beagle.data_generation.studentv2.student.initialize_student_kcs'):
                    student.solve_problem(
                        problem_description="Test",
                        problem_id="test",
                        required_kcs=[],
                        max_steps=5,
                        checkpoint_path=checkpoint_path
                    )


class TestStudentSolveProblemWithSequence:
    """Tests for Student.solve_problem_with_sequence method."""
    
    def test_solve_problem_with_sequence_consumes_sequence(self):
        """Test that pre-generated sequence is consumed."""
        student = create_mocked_student()
        
        student.bkt = MagicMock()
        student.ide_oracle = MagicMock()
        student.markov_model = MagicMock()
        student.assessment_oracle = MagicMock()
        
        # Mock sequence
        mock_sequence = MagicMock()
        mock_sequence.steps = [
            MagicMock(metacog_state='Planning', cog_action='CONSTRUCTING', is_segment_start=True),
            MagicMock(metacog_state='Monitoring', cog_action='DEBUGGING', is_segment_start=False),
        ]
        
        with patch('beagle.data_generation.studentv2.student.student_graph') as mock_graph:
            mock_graph.run_sync = MagicMock()
            
            with patch('beagle.data_generation.studentv2.student.initialize_student_kcs'):
                student.solve_problem_with_sequence(
                    problem_description="Test",
                    sequence=mock_sequence,
                    problem_id="test",
                    required_kcs=[]
                )


class TestStudentGetHistory:
    """Tests for Student.get_history method."""
    
    def test_get_history_returns_state_history(self):
        """Test that get_history returns the state history."""
        from beagle.data_generation.studentv2.state import StudentState
        
        student = create_mocked_student()
        
        student.state = StudentState()
        student.state.history = [
            {'step': 1, 'code': 'x = 1'},
            {'step': 2, 'code': 'x = 2'}
        ]
        
        history = student.get_history()
        
        assert len(history) == 2
        assert history[0]['step'] == 1


class TestStudentBKTIntegration:
    """Tests for Student BKT integration."""
    
    def test_student_creates_bkt(self):
        """Test that Student creates BKT instance."""
        student = create_mocked_student()
        assert student.bkt is not None
    
    def test_student_skip_bkt_on_assisted(self):
        """Test skip_bkt_on_assisted flag."""
        student = create_mocked_student(skip_bkt_on_assisted=True)
        assert student.skip_bkt_on_assisted == True


class TestStudentEdgeCases:
    """Edge cases for Student class."""
    
    def test_student_with_empty_efi_kcs(self):
        """Test Student with empty EFI KCs list."""
        student = create_mocked_student(efi_kcs=[])
        assert student.efi_kcs == []
    
    def test_student_with_empty_force_steps(self):
        """Test Student with empty force assistance steps."""
        student = create_mocked_student(force_assistance_steps=[])
        assert student.force_assistance_steps == []
    
    def test_student_with_various_llm_models(self):
        """Test Student with various LLM model names."""
        models = [
            'google-gla:gemini-2.5-flash',
            'openai:gpt-4o',
            'anthropic:claude-3-sonnet',
        ]
        
        for model in models:
            student = create_mocked_student(llm_model=model)
            # Student stores llm_client, not llm_model directly
            assert student.llm_client is not None


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
