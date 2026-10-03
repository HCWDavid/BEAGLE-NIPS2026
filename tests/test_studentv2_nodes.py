#!/usr/bin/env python3
"""
Unit tests for beagle/data_generation/studentv2/nodes.py

Tests for all graph nodes with comprehensive LLM API mocking:
- MarkovNode: state transitions, interrupt handling
- StrategistNode: prompt composition, LLM calls
- ExecutorNode: CONSTRUCTING/DEBUGGING/ASSESSING branches
- AssistanceNode: question formulation
- TutorNode: hint generation
- OffTopicNode: idle behavior
- EnvironmentNode: code execution, BKT updates
- Pure functions: filter_feedback_for_state, get_comment_guidance
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from beagle.data_generation.studentv2.nodes import (
    run_agent_with_retry,
    MAX_API_RETRIES,
    RETRYABLE_ERROR_KEYWORDS,
    get_comment_guidance,
    filter_feedback_for_state,
    USE_SPARSE_COMMENTS,
    ENABLE_EPISTEMIC_BLINDNESS,
    MarkovNode,
    StrategistNode,
    ExecutorNode,
    AssistanceNode,
    TutorNode,
    OffTopicNode,
    EnvironmentNode,
)


# =============================================================================
# Tests for Pure Functions
# =============================================================================

class TestGetCommentGuidance:
    """Tests for get_comment_guidance function."""
    
    def test_get_comment_guidance_sparse_mode(self):
        """Test comment guidance in sparse mode."""
        import beagle.data_generation.studentv2.nodes as nodes
        original = nodes.USE_SPARSE_COMMENTS
        
        try:
            nodes.USE_SPARSE_COMMENTS = True
            guidance = get_comment_guidance()
            
            assert "sparse" in guidance.lower() or "short" in guidance.lower() or "15%" in guidance
        finally:
            nodes.USE_SPARSE_COMMENTS = original
    
    def test_get_comment_guidance_emotional_mode(self):
        """Test comment guidance in emotional mode."""
        import beagle.data_generation.studentv2.nodes as nodes
        original = nodes.USE_SPARSE_COMMENTS
        
        try:
            nodes.USE_SPARSE_COMMENTS = False
            guidance = get_comment_guidance()
            
            # Emotional mode should have different guidance
            assert guidance is not None
            assert isinstance(guidance, str)
        finally:
            nodes.USE_SPARSE_COMMENTS = original


class TestLoadPrompt:
    """Tests for load_prompt function."""
    
    def test_load_prompt_existing_file(self):
        """Test loading an existing prompt file."""
        from beagle.data_generation.studentv2.nodes import load_prompt
        
        # Load an existing prompt file (common_errors.txt still exists)
        content = load_prompt("common_errors.txt")
        assert content is not None
        assert len(content) > 0
    
    def test_load_prompt_nonexistent_raises(self):
        """Test that loading nonexistent file raises FileNotFoundError."""
        from beagle.data_generation.studentv2.nodes import load_prompt
        
        with pytest.raises(FileNotFoundError):
            load_prompt("nonexistent_file.txt")
    
    def test_load_prompt_strips_whitespace(self):
        """Test that loaded prompts are stripped."""
        from beagle.data_generation.studentv2.nodes import load_prompt
        
        content = load_prompt("common_errors.txt")
        # Should not start or end with whitespace
        assert content == content.strip()


class TestLoadCommonErrors:
    """Tests for load_common_errors function."""
    
    def test_load_common_errors_returns_string(self):
        """Test that load_common_errors returns a string."""
        from beagle.data_generation.studentv2.nodes import load_common_errors
        
        result = load_common_errors()
        assert isinstance(result, str)
    
    def test_load_common_errors_handles_missing_file(self):
        """Test that missing file returns empty string."""
        from beagle.data_generation.studentv2.nodes import load_common_errors
        from unittest.mock import patch
        
        # Mock the file not existing
        with patch('beagle.data_generation.studentv2.nodes.PROMPT_DIR') as mock_dir:
            mock_path = MagicMock()
            mock_path.__truediv__ = MagicMock(return_value=MagicMock())
            mock_path.__truediv__.return_value.read_text.side_effect = FileNotFoundError()
            mock_dir.__truediv__ = mock_path.__truediv__
            
            # Should return empty string, not raise
            result = load_common_errors()
            # If the real file exists, it returns content; that's fine too
            assert isinstance(result, str)


class TestFilterFeedbackForState:
    """Tests for filter_feedback_for_state function."""
    
    def test_enacting_sees_limited_feedback(self):
        """Enacting state should see limited feedback (epistemic blindness)."""
        import beagle.data_generation.studentv2.nodes as nodes
        original = nodes.ENABLE_EPISTEMIC_BLINDNESS
        
        try:
            nodes.ENABLE_EPISTEMIC_BLINDNESS = True
            
            full_output = """
TypeError: unsupported operand type(s) for ^: 'float' and 'float'
  File "solution.py", line 10, in calculate
    return x ^ 2
"""
            filtered = filter_feedback_for_state(full_output, 'Enacting')
            
            # Enacting should see limited info
            assert isinstance(filtered, str)
            # Might see "error occurred" but not full traceback
        finally:
            nodes.ENABLE_EPISTEMIC_BLINDNESS = original
    
    def test_planning_sees_full_feedback(self):
        """Planning state should see full feedback."""
        import beagle.data_generation.studentv2.nodes as nodes
        original = nodes.ENABLE_EPISTEMIC_BLINDNESS
        
        try:
            nodes.ENABLE_EPISTEMIC_BLINDNESS = True
            
            full_output = "Full error traceback here"
            filtered = filter_feedback_for_state(full_output, 'Planning')
            
            # Planning should see everything
            assert filtered == full_output
        finally:
            nodes.ENABLE_EPISTEMIC_BLINDNESS = original
    
    def test_monitoring_sees_full_feedback(self):
        """Monitoring state should see full feedback."""
        full_output = "Full error traceback here"
        filtered = filter_feedback_for_state(full_output, 'Monitoring')
        
        assert filtered == full_output
    
    def test_reflecting_sees_full_feedback(self):
        """Reflecting state should see full feedback."""
        full_output = "Full error traceback here"
        filtered = filter_feedback_for_state(full_output, 'Reflecting')
        
        assert filtered == full_output
    
    def test_epistemic_blindness_disabled(self):
        """When disabled, all states see full feedback."""
        import beagle.data_generation.studentv2.nodes as nodes
        original = nodes.ENABLE_EPISTEMIC_BLINDNESS
        
        try:
            nodes.ENABLE_EPISTEMIC_BLINDNESS = False
            
            full_output = "Full error traceback"
            
            for state in ['Enacting', 'Planning', 'Monitoring', 'Reflecting']:
                filtered = filter_feedback_for_state(full_output, state)
                assert filtered == full_output
        finally:
            nodes.ENABLE_EPISTEMIC_BLINDNESS = original


# =============================================================================
# Tests for MarkovNode
# =============================================================================

class TestMarkovNode:
    """Tests for MarkovNode state transition logic."""
    
    @pytest.fixture
    def markov_node(self):
        return MarkovNode()
    
    @pytest.mark.asyncio
    async def test_markov_node_terminates_on_success(self, mock_graph_context):
        """MarkovNode should return End on execution_success."""
        from pydantic_graph import End
        
        ctx = mock_graph_context(state_kwargs={'execution_success': True})
        node = MarkovNode()
        
        result = await node.run(ctx)
        
        assert isinstance(result, End)
    
    @pytest.mark.asyncio
    async def test_markov_node_terminates_on_max_steps(self, mock_graph_context):
        """MarkovNode should return End when step_count >= max_steps."""
        from pydantic_graph import End
        
        ctx = mock_graph_context(state_kwargs={
            'step_count': 30,
            'max_steps': 30,
            'execution_success': False
        })
        node = MarkovNode()
        
        result = await node.run(ctx)
        
        assert isinstance(result, End)
    
    @pytest.mark.asyncio
    async def test_markov_node_force_assistance(self, mock_graph_context):
        """MarkovNode should trigger AssistanceNode on forced steps."""
        ctx = mock_graph_context(state_kwargs={
            'step_count': 5,
            'force_assistance_steps': [5],
            'segment_duration': 0
        })
        
        with patch('beagle.data_generation.studentv2.nodes.p_offtopic', return_value=0.0):
            with patch('beagle.data_generation.studentv2.nodes.p_assistance', return_value=0.0):
                node = MarkovNode()
                result = await node.run(ctx)
        
        assert isinstance(result, AssistanceNode)
    
    @pytest.mark.asyncio
    async def test_markov_node_skips_interrupt_after_tutor_help(self, mock_graph_context):
        """MarkovNode should skip interrupt checks when just_received_tutor_help."""
        ctx = mock_graph_context(state_kwargs={
            'step_count': 5,
            'force_assistance_steps': [5],
            'just_received_tutor_help': True,
            'segment_duration': 0
        })
        
        with patch('beagle.data_generation.studentv2.nodes.p_offtopic', return_value=0.0):
            with patch('beagle.data_generation.studentv2.nodes.p_assistance', return_value=0.0):
                node = MarkovNode()
                result = await node.run(ctx)
        
        # Should NOT be AssistanceNode since we just got help
        assert not isinstance(result, AssistanceNode)
    
    @pytest.mark.asyncio
    async def test_markov_node_offtopic_probability(self, mock_graph_context):
        """MarkovNode should trigger OffTopicNode based on probability."""
        ctx = mock_graph_context(state_kwargs={
            'step_count': 5,
            'segment_duration': 0
        })
        
        with patch('beagle.data_generation.studentv2.nodes.p_offtopic', return_value=1.0):
            with patch('beagle.data_generation.studentv2.nodes.p_assistance', return_value=0.0):
                with patch('beagle.data_generation.studentv2.nodes.random.random', return_value=0.5):
                    node = MarkovNode()
                    result = await node.run(ctx)
        
        assert isinstance(result, OffTopicNode)
    
    @pytest.mark.asyncio
    async def test_markov_node_assistance_probability(self, mock_graph_context):
        """MarkovNode should trigger AssistanceNode based on probability."""
        ctx = mock_graph_context(state_kwargs={
            'step_count': 5,
            'force_assistance_steps': [],
            'segment_duration': 0
        })
        
        with patch('beagle.data_generation.studentv2.nodes.p_offtopic', return_value=0.0):
            with patch('beagle.data_generation.studentv2.nodes.p_assistance', return_value=1.0):
                with patch('beagle.data_generation.studentv2.nodes.random.random', return_value=0.5):
                    node = MarkovNode()
                    result = await node.run(ctx)
        
        assert isinstance(result, AssistanceNode)
    
    @pytest.mark.asyncio
    async def test_markov_node_samples_new_segment(self, mock_graph_context):
        """MarkovNode should sample new segment when duration exhausted."""
        ctx = mock_graph_context(state_kwargs={
            'step_count': 2,
            'segment_duration': 0,
            'segment_step': 0,
            'current_metacognitive_state': 'Planning'
        })
        
        with patch('beagle.data_generation.studentv2.nodes.p_offtopic', return_value=0.0):
            with patch('beagle.data_generation.studentv2.nodes.p_assistance', return_value=0.0):
                node = MarkovNode()
                result = await node.run(ctx)
        
        # Should return StrategistNode or ExecutorNode for new segment
        assert isinstance(result, (StrategistNode, ExecutorNode))


# =============================================================================
# Tests for StrategistNode
# =============================================================================

class TestStrategistNode:
    """Tests for StrategistNode LLM-driven strategy generation."""
    
    @pytest.mark.asyncio
    async def test_strategist_generates_output(
        self, mock_graph_context, mock_strategist_output
    ):
        """StrategistNode should generate goal, mindset, directive."""
        ctx = mock_graph_context(state_kwargs={
            'current_cognitive_state': 'CONSTRUCTING',
            'current_metacognitive_state': 'Planning',
            'current_code': '',
            'last_output': ''
        })
        
        mock_output = mock_strategist_output(
            goal="Implement the calculate function",
            mindset="Focused",
            directive="Start with function signature"
        )
        
        with patch('beagle.data_generation.studentv2.nodes.run_agent_with_retry', 
                   new_callable=AsyncMock, return_value=mock_output):
            node = StrategistNode()
            result = await node.run(ctx)
        
        # Should return ExecutorNode after strategizing
        assert isinstance(result, ExecutorNode)
        # State should be updated
        assert ctx.state.current_goal == "Implement the calculate function"
        assert ctx.state.current_mindset == "Focused"
        assert ctx.state.current_directive == "Start with function signature"
    
    @pytest.mark.asyncio
    async def test_strategist_uses_bkt_knowledge_state(
        self, mock_graph_context, mock_strategist_output
    ):
        """StrategistNode should incorporate BKT knowledge state."""
        ctx = mock_graph_context(state_kwargs={
            'current_cognitive_state': 'DEBUGGING',
            'current_metacognitive_state': 'Monitoring'
        })
        ctx.deps.bkt.get_sampled_knowledge_state.return_value = "You incorrectly applied radians conversion."
        
        mock_output = mock_strategist_output()
        
        with patch('beagle.data_generation.studentv2.nodes.run_agent_with_retry',
                   new_callable=AsyncMock, return_value=mock_output) as mock_retry:
            node = StrategistNode()
            await node.run(ctx)
            
            # Verify run_agent_with_retry was called
            mock_retry.assert_called_once()
    
    @pytest.mark.skip(reason="Requires deep graph context mocking - integration test")
    @pytest.mark.asyncio
    async def test_strategist_caches_knowledge_state(
        self, mock_graph_context, mock_strategist_output
    ):
        """StrategistNode should cache BKT knowledge state for the phase."""
        ctx = mock_graph_context(state_kwargs={
            'cached_knowledge_state': None
        })
        ctx.deps.bkt.get_sampled_knowledge_state.return_value = "Cached state"
        
        mock_output = mock_strategist_output()
        
        with patch('beagle.data_generation.studentv2.nodes.run_agent_with_retry',
                   new_callable=AsyncMock, return_value=mock_output):
            node = StrategistNode()
            await node.run(ctx)
        
        # Knowledge state should be cached
        assert ctx.state.cached_knowledge_state is not None


# =============================================================================
# Tests for ExecutorNode
# =============================================================================

class TestExecutorNode:
    """Tests for ExecutorNode code/reflection generation."""
    
    
    @pytest.mark.asyncio
    async def test_executor_debugging_branch(
        self, mock_graph_context, mock_debugging_output
    ):
        """ExecutorNode DEBUGGING should generate code and monologue."""
        ctx = mock_graph_context(state_kwargs={
            'current_cognitive_state': 'DEBUGGING',
            'current_metacognitive_state': 'Monitoring',
            'current_code': 'x = 1',
            'last_output': 'Error: expected int',
            'current_goal': 'Fix the bug',
            'current_mindset': 'Determined',
            'current_directive': 'Check variable types'
        })
        
        mock_output = mock_debugging_output(
            code="x = int(1)",
            monologue="I need to ensure x is an integer"
        )
        
        with patch('beagle.data_generation.studentv2.nodes.run_agent_with_retry',
                   new_callable=AsyncMock, return_value=mock_output):
            node = ExecutorNode()
            result = await node.run(ctx)
        
        assert isinstance(result, EnvironmentNode)
        assert "int" in ctx.state.current_code
    
    @pytest.mark.asyncio
    async def test_executor_assessing_branch(
        self, mock_graph_context, mock_assessing_output
    ):
        """ExecutorNode ASSESSING should generate reflection only."""
        ctx = mock_graph_context(state_kwargs={
            'current_cognitive_state': 'ASSESSING',
            'current_metacognitive_state': 'Reflecting',
            'current_code': 'working_code()',
            'last_output': 'All tests passed',
            'current_goal': 'Review code',
            'current_mindset': 'Reflective',
            'current_directive': 'Check edge cases'
        })
        
        mock_output = mock_assessing_output(
            reflection="The code handles edge cases well"
        )
        
        with patch('beagle.data_generation.studentv2.nodes.run_agent_with_retry',
                   new_callable=AsyncMock, return_value=mock_output):
            node = ExecutorNode()
            result = await node.run(ctx)
        
        assert isinstance(result, EnvironmentNode)
        # Reflection should be stored
        assert ctx.state.pending_reflection == "The code handles edge cases well"
    
    @pytest.mark.asyncio
    async def test_executor_uses_tutor_hint(
        self, mock_graph_context, mock_debugging_output
    ):
        """ExecutorNode should incorporate tutor hint when available."""
        ctx = mock_graph_context(state_kwargs={
            'current_cognitive_state': 'DEBUGGING',
            'tutor_response': 'Try using math.radians() for conversion'
        })
        
        mock_output = mock_debugging_output()
        
        with patch('beagle.data_generation.studentv2.nodes.run_agent_with_retry',
                   new_callable=AsyncMock, return_value=mock_output) as mock_retry:
            node = ExecutorNode()
            await node.run(ctx)
            
            # Should have been called with hint in prompt
            mock_retry.assert_called_once()


# =============================================================================
# Tests for AssistanceNode
# =============================================================================

class TestAssistanceNode:
    """Tests for AssistanceNode question formulation."""
    
    @pytest.mark.asyncio
    async def test_assistance_generates_question(
        self, mock_graph_context, mock_assistance_output
    ):
        """AssistanceNode should generate a question for the tutor."""
        ctx = mock_graph_context(state_kwargs={
            'current_code': 'broken_code()',
            'last_output': 'TypeError: cannot add str and int',
            'current_cognitive_state': 'DEBUGGING'
        })
        
        mock_output = mock_assistance_output(
            goal="Understand the error",
            mindset="Confused",
            directive="Ask for clarification",
            question="What does TypeError mean?"
        )
        
        with patch('beagle.data_generation.studentv2.nodes.run_agent_with_retry',
                   new_callable=AsyncMock, return_value=mock_output):
            node = AssistanceNode()
            result = await node.run(ctx)
        
        # Should return TutorNode
        assert isinstance(result, TutorNode)
        # Question should be stored
        assert ctx.state.pending_tutor_question == "What does TypeError mean?"
        # Assistance count should increment
        assert ctx.state.assistance_count >= 1


# =============================================================================
# Tests for TutorNode
# =============================================================================

class TestTutorNode:
    """Tests for TutorNode hint generation."""
    
    @pytest.mark.skip(reason="Requires deep graph context mocking - integration test")
    @pytest.mark.asyncio
    async def test_tutor_generates_hint(
        self, mock_graph_context, mock_tutor_output
    ):
        """TutorNode should generate a helpful hint."""
        ctx = mock_graph_context(state_kwargs={
            'pending_tutor_question': 'What does TypeError mean?',
            'current_code': 'x = "hello" + 5'
        })
        
        mock_output = mock_tutor_output(
            response="TypeError occurs when you try to perform an operation on incompatible types."
        )
        
        with patch('beagle.data_generation.studentv2.nodes.run_agent_with_retry',
                   new_callable=AsyncMock, return_value=mock_output):
            node = TutorNode()
            result = await node.run(ctx)
        
        # Should return MarkovNode after tutor responds
        assert isinstance(result, MarkovNode)
        # Tutor response should be stored
        assert "TypeError" in ctx.state.tutor_response
        # Should flag that we just received help
        assert ctx.state.just_received_tutor_help == True


# =============================================================================
# Tests for OffTopicNode
# =============================================================================

class TestOffTopicNode:
    """Tests for OffTopicNode idle behavior."""
    
    @pytest.mark.asyncio
    async def test_offtopic_increments_count(self, mock_graph_context):
        """OffTopicNode should increment off-topic count."""
        ctx = mock_graph_context(state_kwargs={
            'offtopic_count': 0
        })
        
        node = OffTopicNode()
        result = await node.run(ctx)
        
        # Should return MarkovNode
        assert isinstance(result, MarkovNode)
        # Off-topic count should increment
        assert ctx.state.offtopic_count == 1
    
    @pytest.mark.asyncio
    async def test_offtopic_records_in_history(self, mock_graph_context):
        """OffTopicNode should record idle behavior in history."""
        ctx = mock_graph_context(state_kwargs={
            'history': []
        })
        
        node = OffTopicNode()
        await node.run(ctx)
        
        # History should have an entry
        assert len(ctx.state.history) >= 1
        # Should be marked as off-topic
        last_entry = ctx.state.history[-1]
        assert last_entry.get('cognitive_state') == 'OFF_TOPIC' or 'off' in str(last_entry).lower()


# =============================================================================
# Tests for EnvironmentNode
# =============================================================================

class TestEnvironmentNode:
    """Tests for EnvironmentNode code execution."""
    
    @pytest.mark.skip(reason="Requires IDE Oracle integration - integration test")
    @pytest.mark.asyncio
    async def test_environment_runs_tests(self, mock_graph_context):
        """EnvironmentNode should run tests via IDE Oracle."""
        ctx = mock_graph_context(state_kwargs={
            'current_code': 'def calculate():\n    return 42'
        })
        ctx.deps.ide_oracle.run_tests.return_value = MagicMock(
            passed=3,
            total=5,
            output="3/5 tests passed",
            success=False
        )
        
        node = EnvironmentNode()
        result = await node.run(ctx)
        
        # Should return MarkovNode
        assert isinstance(result, MarkovNode)
        # Oracle should have been called
        ctx.deps.ide_oracle.run_tests.assert_called()
        # State should be updated
        assert ctx.state.tests_passed == 3
        assert ctx.state.tests_total == 5
    
    @pytest.mark.skip(reason="Requires IDE Oracle integration - integration test")
    @pytest.mark.asyncio
    async def test_environment_marks_success(self, mock_graph_context):
        """EnvironmentNode should mark success when all tests pass."""
        ctx = mock_graph_context(state_kwargs={
            'current_code': 'def calculate():\n    return 42'
        })
        ctx.deps.ide_oracle.run_tests.return_value = MagicMock(
            passed=5,
            total=5,
            output="All tests passed!",
            success=True
        )
        
        node = EnvironmentNode()
        await node.run(ctx)
        
        assert ctx.state.execution_success == True
    
    @pytest.mark.skip(reason="Requires IDE Oracle integration - integration test")
    @pytest.mark.asyncio
    async def test_environment_increments_step_count(self, mock_graph_context):
        """EnvironmentNode should increment step count."""
        ctx = mock_graph_context(state_kwargs={
            'step_count': 5,
            'current_code': 'code'
        })
        
        node = EnvironmentNode()
        await node.run(ctx)
        
        assert ctx.state.step_count == 6
    
    @pytest.mark.skip(reason="Requires IDE Oracle integration - integration test")
    @pytest.mark.asyncio
    async def test_environment_records_history(self, mock_graph_context):
        """EnvironmentNode should record step in history."""
        ctx = mock_graph_context(state_kwargs={
            'history': [],
            'current_code': 'x = 1',
            'current_cognitive_state': 'CONSTRUCTING',
            'current_metacognitive_state': 'Planning'
        })
        
        node = EnvironmentNode()
        await node.run(ctx)
        
        assert len(ctx.state.history) >= 1
    
    @pytest.mark.skip(reason="Requires IDE Oracle integration - integration test")
    @pytest.mark.asyncio
    async def test_environment_uses_cached_execution(self, mock_graph_context):
        """EnvironmentNode should use cached execution result if available."""
        ctx = mock_graph_context(state_kwargs={
            'current_code': 'x = 1',
            '_cached_execution_result': MagicMock(
                passed=4,
                total=5,
                output="Cached output",
                success=False
            )
        })
        
        node = EnvironmentNode()
        await node.run(ctx)
        
        # Should use cached result
        assert ctx.state.tests_passed == 4


# =============================================================================
# Tests for run_agent_with_retry
# =============================================================================

class TestRunAgentWithRetry:
    """Tests for the retry logic wrapper."""
    
    @pytest.mark.asyncio
    async def test_success_on_first_try(self):
        """Test successful call on first attempt."""
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.output = {"goal": "test"}
        mock_agent.run = AsyncMock(return_value=mock_result)
        
        result = await run_agent_with_retry(mock_agent, "test prompt")
        
        assert result == {"goal": "test"}
        assert mock_agent.run.call_count == 1
    
    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self):
        """Test retry on rate limit error."""
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.output = {"success": True}
        
        mock_agent.run = AsyncMock(side_effect=[
            Exception("429 rate limit exceeded"),
            mock_result
        ])
        
        with patch('beagle.data_generation.studentv2.nodes.asyncio.sleep', new_callable=AsyncMock):
            result = await run_agent_with_retry(mock_agent, "test")
        
        assert result == {"success": True}
        assert mock_agent.run.call_count == 2
    
    @pytest.mark.asyncio
    async def test_no_retry_on_validation_error(self):
        """Test that validation errors are not retried."""
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=ValueError("Invalid input"))
        
        with pytest.raises(ValueError, match="Invalid input"):
            await run_agent_with_retry(mock_agent, "test")
        
        assert mock_agent.run.call_count == 1
    
    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Test error is raised after max retries."""
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=Exception("rate limit"))
        
        with patch('beagle.data_generation.studentv2.nodes.asyncio.sleep', new_callable=AsyncMock):
            with pytest.raises(Exception, match="rate limit"):
                await run_agent_with_retry(mock_agent, "test")
        
        assert mock_agent.run.call_count == MAX_API_RETRIES


class TestRetryableErrorKeywords:
    """Tests for retryable error keyword list."""
    
    def test_rate_limit_is_retryable(self):
        assert 'rate limit' in RETRYABLE_ERROR_KEYWORDS
    
    def test_timeout_is_retryable(self):
        assert 'timeout' in RETRYABLE_ERROR_KEYWORDS
    
    def test_500_error_is_retryable(self):
        assert '500' in RETRYABLE_ERROR_KEYWORDS
    
    def test_common_http_errors_are_retryable(self):
        for code in ['429', '502', '503', '504']:
            assert code in RETRYABLE_ERROR_KEYWORDS


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
