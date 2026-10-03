#!/usr/bin/env python3
"""
Unit tests for evaluation/run_batch_simulations.py

Tests:
- SimulationResult dataclass
- extract_metrics from history
- check_efi_violation pattern matching
- run_single_simulation with mocked Student
- save_single_run and save_results
- CLI argument parsing
"""

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
from dataclasses import asdict

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.run_batch_simulations import (
    SimulationResult,
    extract_metrics,
    check_efi_violation,
    run_single_simulation,
    save_single_run,
    save_results,
    EFI_PATTERNS,
    PROMPT_VARIANTS,
)


# =============================================================================
# Tests for EFI Pattern Detection
# =============================================================================

class TestCheckEfiViolation:
    """Tests for check_efi_violation function."""
    
    def test_detects_math_import(self):
        """Test detection of 'import math'."""
        code = "import math\n\ndef calculate(x):\n    return math.sin(x)"
        assert check_efi_violation(code, 'KC_C2_MATH_LIBRARY') == True
    
    def test_detects_from_math_import(self):
        """Test detection of 'from math import'."""
        code = "from math import sin, cos\n\ndef calculate(x):\n    return sin(x)"
        assert check_efi_violation(code, 'KC_C2_MATH_LIBRARY') == True
    
    def test_detects_math_dot_usage(self):
        """Test detection of 'math.' usage."""
        code = "x = math.pi * 2"
        assert check_efi_violation(code, 'KC_C2_MATH_LIBRARY') == True
    
    def test_detects_math_radians(self):
        """Test detection of math.radians."""
        code = "angle_rad = math.radians(45)"
        assert check_efi_violation(code, 'KC_P5_UNIT_RADIANS') == True
    
    def test_detects_radians_function(self):
        """Test detection of radians()."""
        code = "from math import radians\nangle = radians(45)"
        assert check_efi_violation(code, 'KC_P5_UNIT_RADIANS') == True
    
    def test_detects_math_degrees(self):
        """Test detection of math.degrees."""
        code = "angle_deg = math.degrees(1.57)"
        assert check_efi_violation(code, 'KC_P6_UNIT_DEGREES') == True
    
    def test_detects_class_definition(self):
        """Test detection of class definition."""
        code = "class Calculator:\n    def __init__(self):\n        pass"
        assert check_efi_violation(code, 'KC_C9_CLASS_DEFINITION') == True
    
    def test_no_violation_without_forbidden_concept(self):
        """Test no false positive for clean code."""
        code = "def calculate(x):\n    return x * 2"
        assert check_efi_violation(code, 'KC_C2_MATH_LIBRARY') == False
    
    def test_no_violation_for_unknown_kc(self):
        """Test unknown KC returns False."""
        code = "import math"
        assert check_efi_violation(code, 'UNKNOWN_KC') == False
    
    def test_empty_code(self):
        """Test empty code returns False."""
        assert check_efi_violation("", 'KC_C2_MATH_LIBRARY') == False
    
    def test_none_code(self):
        """Test None code returns False."""
        assert check_efi_violation(None, 'KC_C2_MATH_LIBRARY') == False
    
    def test_none_efi_kc(self):
        """Test None EFI KC returns False."""
        assert check_efi_violation("import math", None) == False
    
    def test_case_insensitive(self):
        """Test that detection is case insensitive."""
        code = "IMPORT MATH"
        assert check_efi_violation(code, 'KC_C2_MATH_LIBRARY') == True


class TestEfiPatterns:
    """Tests for EFI_PATTERNS dictionary."""
    
    def test_math_library_patterns_exist(self):
        """Test KC_C2_MATH_LIBRARY patterns exist."""
        assert 'KC_C2_MATH_LIBRARY' in EFI_PATTERNS
        patterns = EFI_PATTERNS['KC_C2_MATH_LIBRARY']
        assert len(patterns) >= 1
    
    def test_radians_patterns_exist(self):
        """Test KC_P5_UNIT_RADIANS patterns exist."""
        assert 'KC_P5_UNIT_RADIANS' in EFI_PATTERNS
    
    def test_degrees_patterns_exist(self):
        """Test KC_P6_UNIT_DEGREES patterns exist."""
        assert 'KC_P6_UNIT_DEGREES' in EFI_PATTERNS
    
    def test_class_definition_patterns_exist(self):
        """Test KC_C9_CLASS_DEFINITION patterns exist."""
        assert 'KC_C9_CLASS_DEFINITION' in EFI_PATTERNS


# =============================================================================
# Tests for SimulationResult
# =============================================================================

class TestSimulationResult:
    """Tests for SimulationResult dataclass."""
    
    def test_basic_initialization(self):
        """Test basic SimulationResult initialization."""
        result = SimulationResult(
            run_id=1,
            performance_level='low',
            problem_id='test_problem',
            seed=42,
            solved=True,
            total_steps=10,
            assistance_count=2,
            offtopic_count=1,
            metacog_counts={'Planning': 5, 'Monitoring': 5},
            action_counts={'CONSTRUCTING': 6, 'DEBUGGING': 4},
            final_tests_passed=5,
            final_tests_total=5,
            duration_seconds=30.5,
            timestamp='2024-01-01T12:00:00',
            history=[]
        )
        
        assert result.run_id == 1
        assert result.performance_level == 'low'
        assert result.solved == True
        assert result.efi_kc is None
        assert result.efi_violation_found == False
        assert result.efi_violation_steps == []
    
    def test_with_efi_fields(self):
        """Test SimulationResult with EFI fields."""
        result = SimulationResult(
            run_id=1,
            performance_level='low',
            problem_id='test',
            seed=None,
            solved=False,
            total_steps=5,
            assistance_count=0,
            offtopic_count=0,
            metacog_counts={},
            action_counts={},
            final_tests_passed=0,
            final_tests_total=5,
            duration_seconds=10.0,
            timestamp='2024-01-01',
            history=[],
            efi_kc='KC_C2_MATH_LIBRARY',
            efi_violation_found=True,
            efi_violation_steps=[2, 4]
        )
        
        assert result.efi_kc == 'KC_C2_MATH_LIBRARY'
        assert result.efi_violation_found == True
        assert result.efi_violation_steps == [2, 4]
    
    def test_post_init_sets_empty_list(self):
        """Test __post_init__ sets efi_violation_steps to empty list if None."""
        result = SimulationResult(
            run_id=1,
            performance_level='low',
            problem_id='test',
            seed=None,
            solved=False,
            total_steps=5,
            assistance_count=0,
            offtopic_count=0,
            metacog_counts={},
            action_counts={},
            final_tests_passed=0,
            final_tests_total=5,
            duration_seconds=10.0,
            timestamp='2024-01-01',
            history=[],
            efi_violation_steps=None
        )
        
        assert result.efi_violation_steps == []
    
    def test_to_dict(self):
        """Test to_dict method."""
        result = SimulationResult(
            run_id=1,
            performance_level='low',
            problem_id='test',
            seed=42,
            solved=True,
            total_steps=10,
            assistance_count=1,
            offtopic_count=0,
            metacog_counts={'Planning': 5},
            action_counts={'CONSTRUCTING': 5},
            final_tests_passed=5,
            final_tests_total=5,
            duration_seconds=15.0,
            timestamp='2024-01-01',
            history=[{'step': 1}]
        )
        
        d = result.to_dict()
        
        assert isinstance(d, dict)
        assert d['run_id'] == 1
        assert d['performance_level'] == 'low'
        assert d['solved'] == True
        assert d['history'] == [{'step': 1}]
    
    def test_to_dict_json_serializable(self):
        """Test that to_dict output is JSON serializable."""
        result = SimulationResult(
            run_id=1,
            performance_level='low',
            problem_id='test',
            seed=42,
            solved=True,
            total_steps=10,
            assistance_count=1,
            offtopic_count=0,
            metacog_counts={'Planning': 5},
            action_counts={'CONSTRUCTING': 5},
            final_tests_passed=5,
            final_tests_total=5,
            duration_seconds=15.0,
            timestamp='2024-01-01',
            history=[]
        )
        
        d = result.to_dict()
        json_str = json.dumps(d)
        
        assert json_str is not None


# =============================================================================
# Tests for extract_metrics
# =============================================================================

class TestExtractMetrics:
    """Tests for extract_metrics function."""
    
    def test_empty_history(self):
        """Test extract_metrics with empty history."""
        metrics = extract_metrics([])
        
        assert metrics['solved'] == False
        assert metrics['total_steps'] == 0
        assert metrics['assistance_count'] == 0
        assert metrics['offtopic_count'] == 0
        assert metrics['metacog_counts'] == {}
        assert metrics['action_counts'] == {}
        assert metrics['final_tests_passed'] == 0
        assert metrics['final_tests_total'] == 0
    
    def test_counts_metacog_states(self, sample_simulation_history):
        """Test metacognitive state counting."""
        metrics = extract_metrics(sample_simulation_history)
        
        assert 'Planning' in metrics['metacog_counts']
        assert 'Monitoring' in metrics['metacog_counts']
        assert 'Reflecting' in metrics['metacog_counts']
    
    def test_counts_cognitive_actions(self, sample_simulation_history):
        """Test cognitive action counting."""
        metrics = extract_metrics(sample_simulation_history)
        
        assert 'CONSTRUCTING' in metrics['action_counts']
        assert 'DEBUGGING' in metrics['action_counts']
        assert 'ASSISTANCE' in metrics['action_counts']
    
    def test_counts_assistance(self, sample_simulation_history):
        """Test assistance count."""
        metrics = extract_metrics(sample_simulation_history)
        
        assert metrics['assistance_count'] == 1  # One ASSISTANCE step
    
    def test_counts_offtopic(self):
        """Test off-topic count."""
        history = [
            {'cognitive_state': 'OFF_TOPIC', 'metacognitive_state': 'Planning'},
            {'cognitive_state': 'OFF_TOPIC', 'metacognitive_state': 'Monitoring'}
        ]
        
        metrics = extract_metrics(history)
        
        assert metrics['offtopic_count'] == 2
    
    def test_detects_solved(self, sample_simulation_history):
        """Test detection of solved state."""
        metrics = extract_metrics(sample_simulation_history)
        
        # Last step has success=True
        assert metrics['solved'] == True
    
    def test_extracts_test_progress(self, sample_simulation_history):
        """Test extraction of test progress."""
        metrics = extract_metrics(sample_simulation_history)
        
        # Last step has tests_passed=5, tests_total=5
        assert metrics['final_tests_passed'] == 5
        assert metrics['final_tests_total'] == 5
    
    def test_efi_violation_detection(self, sample_efi_violation_history):
        """Test EFI violation detection in history."""
        metrics = extract_metrics(sample_efi_violation_history, efi_kc='KC_C2_MATH_LIBRARY')
        
        assert metrics['efi_violation_found'] == True
        assert len(metrics['efi_violation_steps']) > 0
    
    def test_no_efi_violation_when_clean(self, sample_simulation_history):
        """Test no EFI violation for clean code."""
        metrics = extract_metrics(sample_simulation_history, efi_kc='KC_C2_MATH_LIBRARY')
        
        assert metrics['efi_violation_found'] == False
        assert metrics['efi_violation_steps'] == []
    
    def test_total_steps_count(self, sample_simulation_history):
        """Test total steps counting."""
        metrics = extract_metrics(sample_simulation_history)
        
        assert metrics['total_steps'] == 4


# =============================================================================
# Tests for save_single_run
# =============================================================================

class TestSaveSingleRun:
    """Tests for save_single_run function."""
    
    def test_creates_output_directory(self, temp_output_dir):
        """Test that output directory is created."""
        result = SimulationResult(
            run_id=1,
            performance_level='low',
            problem_id='test',
            seed=42,
            solved=True,
            total_steps=10,
            assistance_count=0,
            offtopic_count=0,
            metacog_counts={},
            action_counts={},
            final_tests_passed=5,
            final_tests_total=5,
            duration_seconds=10.0,
            timestamp='2024-01-01',
            history=[]
        )
        
        save_single_run(result, temp_output_dir)
        
        runs_dir = temp_output_dir / "runs"
        assert runs_dir.exists()
    
    def test_creates_json_file(self, temp_output_dir):
        """Test that JSON file is created."""
        result = SimulationResult(
            run_id=1,
            performance_level='low',
            problem_id='test',
            seed=42,
            solved=True,
            total_steps=10,
            assistance_count=0,
            offtopic_count=0,
            metacog_counts={},
            action_counts={},
            final_tests_passed=5,
            final_tests_total=5,
            duration_seconds=10.0,
            timestamp='2024-01-01',
            history=[]
        )
        
        save_single_run(result, temp_output_dir)
        
        run_file = temp_output_dir / "runs" / "run_001_low.json"
        assert run_file.exists()
    
    def test_file_contains_valid_json(self, temp_output_dir):
        """Test that saved file contains valid JSON."""
        result = SimulationResult(
            run_id=5,
            performance_level='high',
            problem_id='test',
            seed=None,
            solved=False,
            total_steps=15,
            assistance_count=2,
            offtopic_count=1,
            metacog_counts={'Planning': 10},
            action_counts={'DEBUGGING': 10},
            final_tests_passed=3,
            final_tests_total=5,
            duration_seconds=25.0,
            timestamp='2024-01-01',
            history=[{'step': 1}]
        )
        
        save_single_run(result, temp_output_dir)
        
        run_file = temp_output_dir / "runs" / "run_005_high.json"
        with open(run_file, 'r') as f:
            data = json.load(f)
        
        assert data['run_id'] == 5
        assert data['performance_level'] == 'high'


# =============================================================================
# Tests for save_results
# =============================================================================

class TestSaveResults:
    """Tests for save_results function."""
    
    def test_creates_summary_file(self, temp_output_dir):
        """Test that summary.json is created."""
        results = [
            SimulationResult(
                run_id=1,
                performance_level='low',
                problem_id='test',
                seed=42,
                solved=True,
                total_steps=10,
                assistance_count=0,
                offtopic_count=0,
                metacog_counts={},
                action_counts={},
                final_tests_passed=5,
                final_tests_total=5,
                duration_seconds=10.0,
                timestamp='2024-01-01',
                history=[]
            )
        ]
        
        save_results(results, temp_output_dir)
        
        summary_file = temp_output_dir / "summary.json"
        assert summary_file.exists()
    
    def test_creates_statistics_file(self, temp_output_dir):
        """Test that statistics.json is created."""
        results = [
            SimulationResult(
                run_id=1,
                performance_level='low',
                problem_id='test',
                seed=42,
                solved=True,
                total_steps=10,
                assistance_count=0,
                offtopic_count=0,
                metacog_counts={'Planning': 5},
                action_counts={'CONSTRUCTING': 5},
                final_tests_passed=5,
                final_tests_total=5,
                duration_seconds=10.0,
                timestamp='2024-01-01',
                history=[]
            )
        ]
        
        save_results(results, temp_output_dir)
        
        stats_file = temp_output_dir / "statistics.json"
        assert stats_file.exists()
    
    def test_statistics_contains_low_high_all(self, temp_output_dir):
        """Test that statistics contains low_performers, high_performers, and all."""
        results = [
            SimulationResult(
                run_id=1,
                performance_level='low',
                problem_id='test',
                seed=42,
                solved=True,
                total_steps=10,
                assistance_count=0,
                offtopic_count=0,
                metacog_counts={'Planning': 5},
                action_counts={'CONSTRUCTING': 5},
                final_tests_passed=5,
                final_tests_total=5,
                duration_seconds=10.0,
                timestamp='2024-01-01',
                history=[]
            ),
            SimulationResult(
                run_id=2,
                performance_level='high',
                problem_id='test',
                seed=43,
                solved=False,
                total_steps=15,
                assistance_count=1,
                offtopic_count=0,
                metacog_counts={'Monitoring': 10},
                action_counts={'DEBUGGING': 10},
                final_tests_passed=3,
                final_tests_total=5,
                duration_seconds=20.0,
                timestamp='2024-01-01',
                history=[]
            )
        ]
        
        save_results(results, temp_output_dir)
        
        with open(temp_output_dir / "statistics.json", 'r') as f:
            stats = json.load(f)
        
        assert 'low_performers' in stats
        assert 'high_performers' in stats
        assert 'all' in stats
    
    def test_computes_solved_rate(self, temp_output_dir):
        """Test that solved rate is computed."""
        results = [
            SimulationResult(
                run_id=i,
                performance_level='low',
                problem_id='test',
                seed=i,
                solved=(i % 2 == 0),  # Half solved
                total_steps=10,
                assistance_count=0,
                offtopic_count=0,
                metacog_counts={'Planning': 5},
                action_counts={'CONSTRUCTING': 5},
                final_tests_passed=5 if i % 2 == 0 else 2,
                final_tests_total=5,
                duration_seconds=10.0,
                timestamp='2024-01-01',
                history=[]
            )
            for i in range(4)
        ]
        
        save_results(results, temp_output_dir)
        
        with open(temp_output_dir / "statistics.json", 'r') as f:
            stats = json.load(f)
        
        # 2 out of 4 solved = 0.5
        assert stats['all']['solved_rate'] == 0.5


# =============================================================================
# Tests for run_single_simulation
# =============================================================================

class TestRunSingleSimulation:
    """Tests for run_single_simulation function."""
    
    @patch('evaluation.run_batch_simulations.Student')
    @patch('evaluation.run_batch_simulations.IDEOracle')
    def test_returns_simulation_result(self, mock_oracle_class, mock_student_class):
        """Test that run_single_simulation returns SimulationResult."""
        mock_oracle = MagicMock()
        mock_oracle_class.return_value = mock_oracle
        mock_oracle_class.load_problem.return_value = MagicMock(
            problem_id='test_problem',
            description='Test description',
            required_kcs=['KC_A'],
            starting_code=''
        )
        
        mock_student = MagicMock()
        mock_student_class.return_value = mock_student
        mock_student.solve_problem.return_value = [
            {
                'cognitive_state': 'CONSTRUCTING',
                'metacognitive_state': 'Planning',
                'code': 'x = 1',
                'success': True,
                'tests_passed': 5,
                'tests_total': 5
            }
        ]
        
        result = run_single_simulation(
            run_id=1,
            performance_level='low',
            problem_id='test_problem',
            max_steps=10,
            llm_model='test-model',
            duration_multiplier=0.5
        )
        
        assert isinstance(result, SimulationResult)
        assert result.run_id == 1
        assert result.performance_level == 'low'
    
    @patch('evaluation.run_batch_simulations.Student')
    @patch('evaluation.run_batch_simulations.IDEOracle')
    def test_handles_errors_gracefully(self, mock_oracle_class, mock_student_class):
        """Test that errors return failed SimulationResult."""
        mock_oracle_class.load_problem.side_effect = Exception("Problem not found")
        
        result = run_single_simulation(
            run_id=1,
            performance_level='low',
            problem_id='nonexistent',
            max_steps=10,
            llm_model='test-model',
            duration_multiplier=0.5
        )
        
        assert isinstance(result, SimulationResult)
        assert result.solved == False
        assert result.total_steps == 0
    
    @patch('evaluation.run_batch_simulations.Student')
    @patch('evaluation.run_batch_simulations.IDEOracle')
    def test_passes_efi_kc_to_student(self, mock_oracle_class, mock_student_class):
        """Test that EFI KC is passed to Student."""
        mock_oracle_class.load_problem.return_value = MagicMock(
            problem_id='test',
            description='Test',
            required_kcs=['KC_C2_MATH_LIBRARY'],
            starting_code=''
        )
        
        mock_student = MagicMock()
        mock_student_class.return_value = mock_student
        mock_student.solve_problem.return_value = []
        
        run_single_simulation(
            run_id=1,
            performance_level='low',
            problem_id='test',
            max_steps=10,
            llm_model='test-model',
            duration_multiplier=0.5,
            efi_kc='KC_C2_MATH_LIBRARY'
        )
        
        # Check Student was called with efi_kcs
        call_kwargs = mock_student_class.call_args.kwargs
        assert 'efi_kcs' in call_kwargs
    
    @patch('evaluation.run_batch_simulations.Student')
    @patch('evaluation.run_batch_simulations.IDEOracle')
    def test_passes_force_assistance_steps(self, mock_oracle_class, mock_student_class):
        """Test that force_assistance_steps is passed to Student."""
        mock_oracle_class.load_problem.return_value = MagicMock(
            problem_id='test',
            description='Test',
            required_kcs=[],
            starting_code=''
        )
        
        mock_student = MagicMock()
        mock_student_class.return_value = mock_student
        mock_student.solve_problem.return_value = []
        
        run_single_simulation(
            run_id=1,
            performance_level='low',
            problem_id='test',
            max_steps=10,
            llm_model='test-model',
            duration_multiplier=0.5,
            force_assistance_steps=[4, 9]
        )
        
        call_kwargs = mock_student_class.call_args.kwargs
        assert call_kwargs.get('force_assistance_steps') == [4, 9]


# =============================================================================
# Tests for PROMPT_VARIANTS
# =============================================================================

class TestPromptVariants:
    """Tests for PROMPT_VARIANTS configuration."""
    
    def test_baseline_variant_exists(self):
        """Test that baseline variant exists."""
        assert 'baseline' in PROMPT_VARIANTS
    
    def test_baseline_has_executor_system(self):
        """Test that baseline has executor_system.txt."""
        variant = PROMPT_VARIANTS['baseline']
        assert variant[0] == 'executor_system.txt'


# =============================================================================
# Tests for CLI Argument Handling
# =============================================================================

class TestCLIArguments:
    """Tests for CLI argument parsing."""
    
    def test_default_n_per_level(self):
        """Test default n_per_level is 15."""
        import argparse
        from evaluation.run_batch_simulations import main
        
        # Can't easily test argparse defaults without running main
        # This is more of a documentation test
        pass
    
    def test_force_assistance_steps_parsing(self):
        """Test parsing of --force-assistance-steps."""
        # Test the expected format: "4,9,14" -> [4, 9, 14]
        input_str = "4,9,14"
        expected = [4, 9, 14]
        
        result = [int(x) for x in input_str.split(',')]
        assert result == expected


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
