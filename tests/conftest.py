#!/usr/bin/env python3
"""
Shared pytest fixtures for BEAGLE unit tests.

Provides common mocks and fixtures for testing studentv2 modules
and batch_run_simulations without making actual LLM API calls.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import os

# Tests are fully mocked; dummy keys let SDK clients construct without real credentials.
for _k in ("OPENAI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
    os.environ.setdefault(_k, "test-key")

import pytest
import pytest_asyncio

# Configure pytest-asyncio mode
pytest_plugins = ('pytest_asyncio',)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# Mock LLM Response Fixtures
# =============================================================================

@pytest.fixture
def mock_strategist_output():
    """Factory for creating mock StrategistOutput."""
    def _create(goal="Test goal", mindset="Test mindset", directive="Test directive"):
        from beagle.data_generation.studentv2.models import StrategistOutput
        return StrategistOutput(goal=goal, mindset=mindset, directive=directive)
    return _create


@pytest.fixture
def mock_executor_output():
    """Factory for creating mock ExecutorOutput."""
    def _create(code="print('hello')", monologue="Thinking about the problem"):
        from beagle.data_generation.studentv2.models import ExecutorOutput
        return ExecutorOutput(code=code, monologue=monologue)
    return _create


@pytest.fixture
def mock_constructing_output():
    """Factory for creating mock ConstructingOutput."""
    def _create(code="def calculate():\n    return 42"):
        from beagle.data_generation.studentv2.models import ConstructingOutput
        return ConstructingOutput(code=code)
    return _create


@pytest.fixture
def mock_debugging_output():
    """Factory for creating mock DebuggingOutput."""
    def _create(
        code="def calculate():\n    return 42",
        monologue="Fixed the bug",
        error_seen="Error: expected int",
        memory_note="TypeError - check types"
    ):
        from beagle.data_generation.studentv2.models import DebuggingOutput
        return DebuggingOutput(
            code=code,
            monologue=monologue,
            error_seen=error_seen,
            memory_note=memory_note
        )
    return _create


@pytest.fixture
def mock_assessing_output():
    """Factory for creating mock AssessingOutput."""
    def _create(reflection="The code seems to work correctly"):
        from beagle.data_generation.studentv2.models import AssessingOutput
        return AssessingOutput(reflection=reflection)
    return _create


@pytest.fixture
def mock_assistance_output():
    """Factory for creating mock AssistanceOutput."""
    def _create(
        goal="Understand the error",
        mindset="Confused",
        directive="Ask for help",
        question="What does this error mean?"
    ):
        from beagle.data_generation.studentv2.models import AssistanceOutput
        return AssistanceOutput(
            goal=goal, mindset=mindset, directive=directive, question=question
        )
    return _create


@pytest.fixture
def mock_tutor_output():
    """Factory for creating mock TutorOutput."""
    def _create(response="Try checking your variable types"):
        from beagle.data_generation.studentv2.models import TutorOutput
        return TutorOutput(response=response)
    return _create


# =============================================================================
# State Fixtures
# =============================================================================

@pytest.fixture
def mock_student_state():
    """Create a pre-configured StudentState for testing."""
    from beagle.data_generation.studentv2.state import StudentState
    
    def _create(
        step_count=0,
        max_steps=30,
        current_code="",
        last_output="",
        execution_success=False,
        current_cognitive_state="CONSTRUCTING",
        current_metacognitive_state="Planning",
        tests_passed=0,
        tests_total=5,
        force_assistance_steps=None,
        just_received_tutor_help=False,
        pregenerated_sequence=None,
        segment_duration=0,
        segment_step=0,
        history=None,
        **kwargs
    ):
        return StudentState(
            problem_description="Implement a function to calculate projectile range",
            problem_id="test_problem",
            required_kcs=["KC_C2_MATH_LIBRARY", "KC_P5_UNIT_RADIANS"],
            current_code=current_code,
            last_output=last_output,
            execution_success=execution_success,
            current_cognitive_state=current_cognitive_state,
            current_metacognitive_state=current_metacognitive_state,
            metacognitive_history=[current_metacognitive_state] if current_metacognitive_state else [],
            step_count=step_count,
            max_steps=max_steps,
            tests_passed=tests_passed,
            tests_total=tests_total,
            segment_duration=segment_duration,
            segment_step=segment_step,
            force_assistance_steps=force_assistance_steps or [],
            just_received_tutor_help=just_received_tutor_help,
            pregenerated_sequence=pregenerated_sequence,
            history=history or [],
            **kwargs
        )
    return _create


@pytest.fixture
def mock_student_deps():
    """Create mock StudentDeps for testing."""
    def _create(performance_level='low', persona_type='low_performer'):
        deps = MagicMock()
        deps.performance_level = performance_level
        deps.persona_type = persona_type  # ρ_persona: controls language style
        
        # Mock markov model
        deps.markov_model = MagicMock()
        deps.markov_model.sample_next_state.return_value = "Planning"
        deps.markov_model.sample_action.return_value = "CONSTRUCTING"
        deps.markov_model.sample_duration.return_value = 2
        
        # Mock LLM client
        deps.llm_client = MagicMock()
        
        # Mock IDE Oracle
        deps.ide_oracle = MagicMock()
        deps.ide_oracle.run_tests.return_value = MagicMock(
            passed=2,
            total=5,
            output="2/5 tests passed",
            success=False
        )
        
        # Mock BKT
        deps.bkt = MagicMock()
        deps.bkt.get_sampled_knowledge_state.return_value = "You correctly applied function definition."
        deps.bkt.sample_and_feedback.return_value = {}
        deps.bkt.update.return_value = None
        
        # Mock assessment oracle
        deps.assessment_oracle = MagicMock()
        
        # Duration multiplier
        deps.duration_multiplier = 0.5
        
        # BKT caching disabled by default
        deps.cache_bkt = False
        
        return deps
    return _create


@pytest.fixture
def mock_graph_context(mock_student_state, mock_student_deps):
    """Create a mock GraphRunContext for testing nodes."""
    def _create(state_kwargs=None, deps_kwargs=None):
        state = mock_student_state(**(state_kwargs or {}))
        deps = mock_student_deps(**(deps_kwargs or {}))
        
        ctx = MagicMock()
        ctx.state = state
        ctx.deps = deps
        return ctx
    return _create


# =============================================================================
# Simulation History Fixtures
# =============================================================================

@pytest.fixture
def sample_simulation_history():
    """Sample simulation history for metric extraction tests."""
    return [
        {
            'step': 1,
            'cognitive_state': 'CONSTRUCTING',
            'metacognitive_state': 'Planning',
            'code': 'def calculate():\n    pass',
            'output': '',
            'tests_passed': 0,
            'tests_total': 5,
            'success': False
        },
        {
            'step': 2,
            'cognitive_state': 'DEBUGGING',
            'metacognitive_state': 'Monitoring',
            'code': 'def calculate():\n    return 42',
            'output': 'Error: expected float',
            'tests_passed': 1,
            'tests_total': 5,
            'success': False
        },
        {
            'step': 3,
            'cognitive_state': 'ASSISTANCE',
            'metacognitive_state': 'Monitoring',
            'code': 'def calculate():\n    return 42',
            'output': '',
            'tests_passed': 1,
            'tests_total': 5,
            'success': False
        },
        {
            'step': 4,
            'cognitive_state': 'DEBUGGING',
            'metacognitive_state': 'Reflecting',
            'code': 'def calculate():\n    return 42.0',
            'output': 'All tests passed!',
            'tests_passed': 5,
            'tests_total': 5,
            'success': True
        },
    ]


@pytest.fixture
def sample_efi_violation_history():
    """History containing an EFI violation (uses math library)."""
    return [
        {
            'step': 1,
            'cognitive_state': 'CONSTRUCTING',
            'metacognitive_state': 'Planning',
            'code': 'import math\n\ndef calculate(angle):\n    return math.sin(angle)',
            'output': '',
            'tests_passed': 0,
            'tests_total': 5,
            'success': False
        },
        {
            'step': 2,
            'cognitive_state': 'DEBUGGING',
            'metacognitive_state': 'Monitoring',
            'code': 'import math\n\ndef calculate(angle):\n    return math.sin(math.radians(angle))',
            'output': 'All tests passed!',
            'tests_passed': 5,
            'tests_total': 5,
            'success': True
        },
    ]


# =============================================================================
# IDE Oracle Fixtures
# =============================================================================

@pytest.fixture
def mock_ide_oracle():
    """Create a mock IDEOracle for testing."""
    oracle = MagicMock()
    oracle.run_tests.return_value = MagicMock(
        passed=2,
        total=5,
        output="2/5 tests passed\nFailed: test_angle_conversion",
        success=False
    )
    oracle.save_history = True
    return oracle


@pytest.fixture
def mock_problem_def():
    """Create a mock problem definition."""
    problem = MagicMock()
    problem.problem_id = "test_problem"
    problem.description = "Implement projectile motion calculation"
    problem.required_kcs = ["KC_C2_MATH_LIBRARY", "KC_P5_UNIT_RADIANS"]
    problem.starting_code = ""
    return problem


# =============================================================================
# BKT Fixtures
# =============================================================================

@pytest.fixture
def mock_bkt():
    """Create a mock BKT for testing."""
    from beagle.data_generation.studentv2.bkt.bkt import BKT, BKTParameters
    
    bkt = BKT()
    bkt.initialize_kc('KC_C2_MATH_LIBRARY', BKTParameters(p_init=0.5))
    bkt.initialize_kc('KC_P5_UNIT_RADIANS', BKTParameters(p_init=0.3))
    return bkt


# =============================================================================
# Async Test Helpers
# =============================================================================

@pytest.fixture
def run_async():
    """Helper to run async functions in sync tests."""
    import asyncio
    
    def _run(coro):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)
    return _run


# =============================================================================
# Path Fixtures
# =============================================================================

@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory for test results."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def project_root_path():
    """Return the project root path."""
    return Path(__file__).parent.parent
