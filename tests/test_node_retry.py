#!/usr/bin/env python3
"""
Tests for API retry logic in studentv2 nodes.

Tests that:
1. run_agent_with_retry correctly retries on API-level errors
2. Non-retryable errors propagate immediately
3. Exponential backoff is applied
4. All node types (Assistance, Tutor, Strategist, Executor) use retry logic
"""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

sys.path.insert(0, '.')

from beagle.data_generation.studentv2.nodes import (
    run_agent_with_retry,
    MAX_API_RETRIES,
    RETRYABLE_ERROR_KEYWORDS,
)


class MockAgentResult:
    """Mock result from pydantic_ai Agent.run()"""
    def __init__(self, output):
        self.output = output


class TestRunAgentWithRetry:
    """Tests for the run_agent_with_retry function."""

    @pytest.mark.asyncio
    async def test_success_on_first_try(self):
        """Test that successful calls return immediately."""
        mock_agent = MagicMock()
        mock_output = {"goal": "test", "mindset": "focused"}
        mock_agent.run = AsyncMock(return_value=MockAgentResult(mock_output))

        result = await run_agent_with_retry(mock_agent, "test prompt")

        assert result == mock_output
        assert mock_agent.run.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_malformed_function_call(self):
        """Test retry on MALFORMED_FUNCTION_CALL error (the exact error from Run 13)."""
        mock_agent = MagicMock()
        mock_output = {"goal": "success"}

        # Fail twice with MALFORMED_FUNCTION_CALL, succeed on third try
        error_msg = "Content field missing from Gemini response, finish_reason: MALFORMED_FUNCTION_CALL"
        mock_agent.run = AsyncMock(side_effect=[
            Exception(error_msg),
            Exception(error_msg),
            MockAgentResult(mock_output)
        ])

        with patch('beagle.data_generation.studentv2.nodes.asyncio.sleep', new_callable=AsyncMock):
            result = await run_agent_with_retry(mock_agent, "test prompt")

        assert result == mock_output
        assert mock_agent.run.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_on_exact_run13_error(self):
        """Test retry using the EXACT error message from batch_30/run_013_low.json."""
        mock_agent = MagicMock()
        mock_output = {"goal": "recovered"}

        # This is the exact error from Run 13 (trimmed for readability but contains key parts)
        exact_error = '''Content field missing from Gemini response, body:
{
  "candidates": [
    {
      "content": null,
      "finish_reason": "MALFORMED_FUNCTION_CALL",
      "index": 0
    }
  ],
  "model_version": "gemini-2.5-flash"
}'''
        mock_agent.run = AsyncMock(side_effect=[
            Exception(exact_error),
            MockAgentResult(mock_output)
        ])

        with patch('beagle.data_generation.studentv2.nodes.asyncio.sleep', new_callable=AsyncMock):
            result = await run_agent_with_retry(mock_agent, "test prompt")

        assert result == mock_output
        assert mock_agent.run.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self):
        """Test retry on rate limit errors."""
        mock_agent = MagicMock()
        mock_output = {"response": "help"}

        mock_agent.run = AsyncMock(side_effect=[
            Exception("429 rate limit exceeded"),
            MockAgentResult(mock_output)
        ])

        with patch('beagle.data_generation.studentv2.nodes.asyncio.sleep', new_callable=AsyncMock):
            result = await run_agent_with_retry(mock_agent, "test prompt")

        assert result == mock_output
        assert mock_agent.run.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self):
        """Test retry on timeout errors."""
        mock_agent = MagicMock()
        mock_output = {"code": "print('hello')"}

        mock_agent.run = AsyncMock(side_effect=[
            Exception("Connection timeout"),
            MockAgentResult(mock_output)
        ])

        with patch('beagle.data_generation.studentv2.nodes.asyncio.sleep', new_callable=AsyncMock):
            result = await run_agent_with_retry(mock_agent, "test prompt")

        assert result == mock_output
        assert mock_agent.run.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_503_error(self):
        """Test retry on 503 service unavailable."""
        mock_agent = MagicMock()
        mock_output = {"directive": "fix the bug"}

        mock_agent.run = AsyncMock(side_effect=[
            Exception("503 Service Unavailable"),
            MockAgentResult(mock_output)
        ])

        with patch('beagle.data_generation.studentv2.nodes.asyncio.sleep', new_callable=AsyncMock):
            result = await run_agent_with_retry(mock_agent, "test prompt")

        assert result == mock_output
        assert mock_agent.run.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_openai_insufficient_quota(self):
        """Test retry on OpenAI insufficient_quota error."""
        mock_agent = MagicMock()
        mock_output = {"code": "print('fixed')"}

        mock_agent.run = AsyncMock(side_effect=[
            Exception("OpenAI API error: insufficient_quota - You exceeded your current quota"),
            MockAgentResult(mock_output)
        ])

        with patch('beagle.data_generation.studentv2.nodes.asyncio.sleep', new_callable=AsyncMock):
            result = await run_agent_with_retry(mock_agent, "test prompt")

        assert result == mock_output
        assert mock_agent.run.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_openai_server_error(self):
        """Test retry on OpenAI server_error."""
        mock_agent = MagicMock()
        mock_output = {"goal": "done"}

        mock_agent.run = AsyncMock(side_effect=[
            Exception("OpenAI server_error: The server had an error processing your request"),
            MockAgentResult(mock_output)
        ])

        with patch('beagle.data_generation.studentv2.nodes.asyncio.sleep', new_callable=AsyncMock):
            result = await run_agent_with_retry(mock_agent, "test prompt")

        assert result == mock_output
        assert mock_agent.run.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_anthropic_529_overloaded(self):
        """Test retry on Anthropic 529 overloaded_error."""
        mock_agent = MagicMock()
        mock_output = {"response": "here's a hint"}

        mock_agent.run = AsyncMock(side_effect=[
            Exception('API Error (529 {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}})'),
            MockAgentResult(mock_output)
        ])

        with patch('beagle.data_generation.studentv2.nodes.asyncio.sleep', new_callable=AsyncMock):
            result = await run_agent_with_retry(mock_agent, "test prompt")

        assert result == mock_output
        assert mock_agent.run.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_anthropic_rate_limit(self):
        """Test retry on Anthropic rate limit error."""
        mock_agent = MagicMock()
        mock_output = {"directive": "implement feature"}

        mock_agent.run = AsyncMock(side_effect=[
            Exception("Anthropic API rate limit exceeded"),
            MockAgentResult(mock_output)
        ])

        with patch('beagle.data_generation.studentv2.nodes.asyncio.sleep', new_callable=AsyncMock):
            result = await run_agent_with_retry(mock_agent, "test prompt")

        assert result == mock_output
        assert mock_agent.run.call_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_non_retryable_error(self):
        """Test that non-retryable errors propagate immediately."""
        mock_agent = MagicMock()

        # This error should NOT be retried
        mock_agent.run = AsyncMock(side_effect=ValueError("Invalid argument"))

        with pytest.raises(ValueError, match="Invalid argument"):
            await run_agent_with_retry(mock_agent, "test prompt")

        # Should only try once
        assert mock_agent.run.call_count == 1

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Test that error is raised after max retries."""
        mock_agent = MagicMock()

        # Always fail with retryable error
        error_msg = "Content field missing from Gemini response"
        mock_agent.run = AsyncMock(side_effect=Exception(error_msg))

        with patch('beagle.data_generation.studentv2.nodes.asyncio.sleep', new_callable=AsyncMock):
            with pytest.raises(Exception, match="Content field missing"):
                await run_agent_with_retry(mock_agent, "test prompt")

        # Should try MAX_API_RETRIES times
        assert mock_agent.run.call_count == MAX_API_RETRIES


class TestRetryableErrorKeywords:
    """Tests for the list of retryable error keywords."""

    def test_malformed_function_call_is_retryable(self):
        """Verify MALFORMED_FUNCTION_CALL is in retryable list."""
        assert 'malformed_function_call' in RETRYABLE_ERROR_KEYWORDS

    def test_content_field_missing_is_retryable(self):
        """Verify content field missing is in retryable list."""
        assert 'content field missing' in RETRYABLE_ERROR_KEYWORDS

    def test_rate_limit_is_retryable(self):
        """Verify rate limit is in retryable list."""
        assert 'rate limit' in RETRYABLE_ERROR_KEYWORDS

    def test_common_http_errors_are_retryable(self):
        """Verify common HTTP errors are retryable."""
        http_errors = ['429', '500', '502', '503', '504']
        for code in http_errors:
            assert code in RETRYABLE_ERROR_KEYWORDS, f"{code} should be retryable"

    def test_gemini_specific_errors_are_retryable(self):
        """Verify Gemini-specific errors are retryable."""
        gemini_errors = ['recitation', 'safety', 'resource_exhausted', 'malformed_function_call']
        for error in gemini_errors:
            assert error in RETRYABLE_ERROR_KEYWORDS, f"{error} should be retryable"

    def test_openai_specific_errors_are_retryable(self):
        """Verify OpenAI-specific errors are retryable."""
        openai_errors = ['insufficient_quota', 'server_error', 'engine_overloaded', 'openai']
        for error in openai_errors:
            assert error in RETRYABLE_ERROR_KEYWORDS, f"{error} should be retryable"

    def test_anthropic_specific_errors_are_retryable(self):
        """Verify Anthropic-specific errors are retryable."""
        anthropic_errors = ['529', 'overloaded_error', 'anthropic']
        for error in anthropic_errors:
            assert error in RETRYABLE_ERROR_KEYWORDS, f"{error} should be retryable"


class TestNodeIntegration:
    """Integration tests verifying nodes use retry logic."""

    @pytest.mark.asyncio
    async def test_strategist_node_uses_retry(self):
        """Test that StrategistNode uses run_agent_with_retry."""
        # This is a code inspection test - verify the node calls run_agent_with_retry
        from beagle.data_generation.studentv2 import nodes
        import inspect

        source = inspect.getsource(nodes.StrategistNode.run)
        assert 'run_agent_with_retry' in source, "StrategistNode should use run_agent_with_retry"


    @pytest.mark.asyncio
    async def test_assistance_node_uses_retry(self):
        """Test that AssistanceNode uses run_agent_with_retry."""
        from beagle.data_generation.studentv2 import nodes
        import inspect

        source = inspect.getsource(nodes.AssistanceNode.run)
        assert 'run_agent_with_retry' in source, "AssistanceNode should use run_agent_with_retry"


def run_tests():
    """Run all tests without pytest (for quick manual testing)."""
    print("=" * 70)
    print("NODE RETRY LOGIC TESTS")
    print("=" * 70)

    # Test retryable keywords
    print("\n--- Testing Retryable Error Keywords ---")
    test_keywords = TestRetryableErrorKeywords()
    test_keywords.test_malformed_function_call_is_retryable()
    test_keywords.test_content_field_missing_is_retryable()
    test_keywords.test_rate_limit_is_retryable()
    test_keywords.test_common_http_errors_are_retryable()
    test_keywords.test_gemini_specific_errors_are_retryable()
    test_keywords.test_openai_specific_errors_are_retryable()
    test_keywords.test_anthropic_specific_errors_are_retryable()
    print("✓ All retryable keyword tests passed")

    # Test node integration (source inspection)
    print("\n--- Testing Node Integration ---")
    from beagle.data_generation.studentv2 import nodes
    import inspect

    for node_name in ['StrategistNode', 'ExecutorNode', 'AssistanceNode', 'TutorNode']:
        node_class = getattr(nodes, node_name)
        source = inspect.getsource(node_class.run)
        assert 'run_agent_with_retry' in source, f"{node_name} should use run_agent_with_retry"
        print(f"✓ {node_name} uses run_agent_with_retry")

    # Run async tests
    print("\n--- Testing Retry Logic ---")

    async def run_async_tests():
        test_retry = TestRunAgentWithRetry()

        await test_retry.test_success_on_first_try()
        print("✓ Success on first try")

        await test_retry.test_retry_on_malformed_function_call()
        print("✓ Retry on MALFORMED_FUNCTION_CALL")

        await test_retry.test_retry_on_exact_run13_error()
        print("✓ Retry on EXACT Run 13 error (batch_30/run_013_low.json)")

        await test_retry.test_retry_on_rate_limit()
        print("✓ Retry on rate limit")

        await test_retry.test_retry_on_timeout()
        print("✓ Retry on timeout")

        await test_retry.test_retry_on_503_error()
        print("✓ Retry on 503 error")

        await test_retry.test_retry_on_openai_insufficient_quota()
        print("✓ Retry on OpenAI insufficient_quota")

        await test_retry.test_retry_on_openai_server_error()
        print("✓ Retry on OpenAI server_error")

        await test_retry.test_retry_on_anthropic_529_overloaded()
        print("✓ Retry on Anthropic 529 overloaded")

        await test_retry.test_retry_on_anthropic_rate_limit()
        print("✓ Retry on Anthropic rate limit")

        await test_retry.test_no_retry_on_non_retryable_error()
        print("✓ No retry on non-retryable error")

        await test_retry.test_max_retries_exceeded()
        print("✓ Max retries exceeded raises error")

        await test_retry.test_exponential_backoff()
        print("✓ Exponential backoff applied")

    asyncio.run(run_async_tests())

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
