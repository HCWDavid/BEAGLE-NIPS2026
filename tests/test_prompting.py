"""
Tests for Epistemic Gating Module

Tests the epistemic blindness mechanism that hides error details
from students in "Enacting" state.
"""

import pytest
from beagle.data_generation.studentv2.prompting import (
    EpistemicGating,
    NeuroSymbolicPromptGenerator,  # Legacy alias
    get_generator,
    set_sparse_comments,
)


class TestEpistemicBlindness:
    """Test the epistemic blindness mechanism."""
    
    @pytest.fixture
    def gating(self):
        return EpistemicGating()
    
    
    
    def test_enacting_hides_test_failures(self, gating):
        """In Enacting state, test failures should be hidden."""
        raw_output = "FAILED test_example - Expected 5 but got 10"
        
        filtered = gating.apply_blindness("Enacting", raw_output)
        
        assert "Expected 5 but got 10" not in filtered
        assert "Tests Failed" in filtered
    
    def test_monitoring_shows_full_error(self, gating):
        """In Monitoring state, full error details should be visible."""
        raw_output = "Traceback (most recent call last):\n  TypeError: missing argument"
        
        filtered = gating.apply_blindness("Monitoring", raw_output)
        
        assert "TypeError" in filtered
        assert "missing argument" in filtered
        assert "glanced at the red text" not in filtered
    
    def test_planning_shows_full_error(self, gating):
        """In Planning state, full error details should be visible."""
        raw_output = "SyntaxError: invalid syntax"
        
        filtered = gating.apply_blindness("Planning", raw_output)
        
        assert "SyntaxError" in filtered
    
    def test_reflecting_shows_full_error(self, gating):
        """In Reflecting state, full error details should be visible."""
        raw_output = "ValueError: too many values"
        
        filtered = gating.apply_blindness("Reflecting", raw_output)
        
        assert "ValueError" in filtered
    
    def test_no_output_handled(self, gating):
        """Empty output should be handled gracefully."""
        filtered = gating.apply_blindness("Enacting", "")
        assert "(No output yet)" in filtered
    
    def test_success_output_not_hidden(self, gating):
        """Successful output should not be hidden even in Enacting state."""
        raw_output = "All tests passed!\nScore: 100%"
        
        filtered = gating.apply_blindness("Enacting", raw_output)
        
        assert "All tests passed" in filtered
        assert "Score: 100%" in filtered


class TestFactoryFunction:
    """Test the singleton factory function."""
    
    def test_get_generator_returns_instance(self):
        """get_generator should return an EpistemicGating instance."""
        gen = get_generator()
        assert isinstance(gen, EpistemicGating)
    
    def test_get_generator_returns_singleton(self):
        """get_generator should return the same instance."""
        gen1 = get_generator()
        gen2 = get_generator()
        assert gen1 is gen2
    
    def test_legacy_alias_works(self):
        """NeuroSymbolicPromptGenerator should be an alias for EpistemicGating."""
        gen = get_generator()
        assert isinstance(gen, NeuroSymbolicPromptGenerator)


class TestSparseCommentsFlag:
    """Test the sparse comments flag (legacy, now handled in nodes.py)."""
    
    def test_set_sparse_comments_works(self):
        """set_sparse_comments should not raise."""
        set_sparse_comments(True)
        set_sparse_comments(False)
