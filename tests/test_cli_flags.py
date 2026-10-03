"""
Test CLI flag interactions for run_batch_simulations.py

Tests that these flags work together correctly:
- --force-assistance-steps: Force assistance at specific step indices
- --disable-assistance: Set p_assistance to 0
- --disable-offtopic: Set p_offtopic to 0

Key interaction: force_assistance_steps should OVERRIDE disable_assistance
because forced assistance is checked BEFORE the probabilistic assistance check.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import unittest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict


def create_mock_state(
    step_count: int = 0,
    force_assistance_steps: Optional[List[int]] = None,
    max_steps: int = 30,
    execution_success: bool = False,
    just_received_tutor_help: bool = False,
    tests_total: int = 5,
    tests_passed: int = 2,
    segment_step: int = 0,
    segment_duration: int = 0,
    current_metacognitive_state: str = "Planning",
    pregenerated_sequence: Any = None
):
    """Create a fully mocked StudentState with all required attributes."""
    state = MagicMock()

    # Core attributes
    state.step_count = step_count
    state.force_assistance_steps = force_assistance_steps or []
    state.max_steps = max_steps
    state.execution_success = execution_success
    state.just_received_tutor_help = just_received_tutor_help
    state.tests_total = tests_total
    state.tests_passed = tests_passed

    # Markov state attributes
    state.segment_step = segment_step
    state.segment_duration = segment_duration
    state.current_metacognitive_state = current_metacognitive_state
    state.current_cognitive_state = "CONSTRUCTING"
    state.pregenerated_sequence = pregenerated_sequence
    state.metacognitive_history = ["Planning"]  # For 2nd order Markov

    # Other state attributes that might be accessed
    state.current_goal = ""
    state.current_mindset = ""
    state.current_directive = ""
    state.cached_knowledge_state = None

    return state


def create_mock_deps(performance_level: str = "low"):
    """Create a fully mocked StudentDeps with all required attributes."""
    deps = MagicMock()
    deps.performance_level = performance_level

    # Mock markov model
    deps.markov_model = MagicMock()
    deps.markov_model.sample_next_state.return_value = "Planning"
    deps.markov_model.sample_action.return_value = "CONSTRUCTING"
    deps.markov_model.sample_duration.return_value = 2

    return deps


def create_mock_context(state, deps):
    """Create a mock GraphRunContext."""
    ctx = MagicMock()
    ctx.state = state
    ctx.deps = deps
    return ctx


class TestForceAssistanceWithDisabledAssistance(unittest.TestCase):
    """Test that force_assistance_steps works even when assistance is disabled."""

    def test_force_assistance_checked_before_probability(self):
        """
        Verify code logic: force_assistance_steps is checked BEFORE p_assistance.

        In MarkovNode.run():
        1. Off-topic check (p_offtopic)
        2. Force assistance check (step_count in force_assistance_steps)
        3. Probabilistic assistance check (p_assistance)

        So force_assistance should work even if p_assistance = 0.
        """
        # Read the source to verify the order
        nodes_path = Path(__file__).parent.parent / "beagle/data_generation/studentv2/nodes.py"
        source = nodes_path.read_text()

        # Find the positions of key checks
        force_check_pos = source.find("force_assistance_steps")
        prob_check_pos = source.find("p_assistance(")

        self.assertGreater(force_check_pos, 0, "force_assistance_steps check not found")
        self.assertGreater(prob_check_pos, 0, "p_assistance check not found")
        self.assertLess(force_check_pos, prob_check_pos,
            "force_assistance_steps should be checked BEFORE p_assistance probabilistic check")

        print("✓ Code structure verified: force_assistance_steps checked before p_assistance")

    def test_step_count_in_force_steps_returns_assistance_node(self):
        """Test that when step_count is in force_assistance_steps, AssistanceNode is returned."""
        from beagle.data_generation.studentv2.nodes import MarkovNode, AssistanceNode

        state = create_mock_state(
            step_count=5,
            force_assistance_steps=[5, 10, 15]
        )
        deps = create_mock_deps()
        ctx = create_mock_context(state, deps)

        # Patch p_offtopic and p_assistance to return 0
        with patch('beagle.data_generation.studentv2.nodes.p_offtopic', return_value=0.0):
            with patch('beagle.data_generation.studentv2.nodes.p_assistance', return_value=0.0):
                node = MarkovNode()
                result = asyncio.get_event_loop().run_until_complete(node.run(ctx))

                self.assertIsInstance(result, AssistanceNode,
                    "Should return AssistanceNode when step_count in force_assistance_steps")

        print("✓ force_assistance_steps correctly triggers AssistanceNode")

    def test_step_not_in_force_steps_with_disabled_assistance(self):
        """Test that when step is NOT in force_steps and assistance disabled, no assistance."""
        from beagle.data_generation.studentv2.nodes import MarkovNode, AssistanceNode, StrategistNode

        state = create_mock_state(
            step_count=3,  # NOT in force list
            force_assistance_steps=[5, 10, 15],
            segment_duration=0  # Will trigger new segment sampling
        )
        deps = create_mock_deps()
        ctx = create_mock_context(state, deps)

        with patch('beagle.data_generation.studentv2.nodes.p_offtopic', return_value=0.0):
            with patch('beagle.data_generation.studentv2.nodes.p_assistance', return_value=0.0):
                node = MarkovNode()
                result = asyncio.get_event_loop().run_until_complete(node.run(ctx))

                # Should NOT be AssistanceNode - should fall through to Markov logic
                self.assertNotIsInstance(result, AssistanceNode,
                    "Should NOT return AssistanceNode when step not in force list and p_assistance=0")
                # Should be StrategistNode (new segment) or ExecutorNode (continue segment)
                self.assertIn(type(result).__name__, ['StrategistNode', 'ExecutorNode'],
                    "Should return Markov node (Strategist or Executor) when not forcing assistance")

        print("✓ Correctly skips assistance when step not in force list and assistance disabled")


class TestDisableOfftopic(unittest.TestCase):
    """Test that --disable-offtopic correctly prevents off-topic behavior."""

    def test_offtopic_disabled_prevents_offtopic(self):
        """When p_offtopic returns 0, should never go off-topic."""
        from beagle.data_generation.studentv2.nodes import MarkovNode, OffTopicNode

        state = create_mock_state(
            step_count=5,
            force_assistance_steps=[],
            segment_duration=0
        )
        deps = create_mock_deps()
        ctx = create_mock_context(state, deps)

        # Run 20 times with p_offtopic = 0
        with patch('beagle.data_generation.studentv2.nodes.p_offtopic', return_value=0.0):
            with patch('beagle.data_generation.studentv2.nodes.p_assistance', return_value=0.0):
                node = MarkovNode()
                for _ in range(20):
                    result = asyncio.get_event_loop().run_until_complete(node.run(ctx))
                    self.assertNotIsInstance(result, OffTopicNode,
                        "Should never return OffTopicNode when p_offtopic=0")

        print("✓ disable-offtopic correctly prevents off-topic behavior")

    def test_offtopic_can_trigger_with_high_probability(self):
        """Verify off-topic CAN be triggered when probability is high (control test)."""
        from beagle.data_generation.studentv2.nodes import MarkovNode, OffTopicNode

        state = create_mock_state(
            step_count=5,
            force_assistance_steps=[]
        )
        deps = create_mock_deps()
        ctx = create_mock_context(state, deps)

        offtopic_count = 0
        # Run 50 times with p_offtopic = 1.0 (always)
        with patch('beagle.data_generation.studentv2.nodes.p_offtopic', return_value=1.0):
            with patch('beagle.data_generation.studentv2.nodes.p_assistance', return_value=0.0):
                node = MarkovNode()
                for _ in range(50):
                    result = asyncio.get_event_loop().run_until_complete(node.run(ctx))
                    if isinstance(result, OffTopicNode):
                        offtopic_count += 1

        # Should always trigger off-topic
        self.assertEqual(offtopic_count, 50,
            "Should always return OffTopicNode when p_offtopic=1.0")

        print("✓ Off-topic correctly triggers with high probability (control test passed)")


class TestDisableAssistance(unittest.TestCase):
    """Test that --disable-assistance correctly prevents probabilistic assistance."""

    def test_assistance_disabled_prevents_random_assistance(self):
        """When p_assistance returns 0, should never get random assistance."""
        from beagle.data_generation.studentv2.nodes import MarkovNode, AssistanceNode

        state = create_mock_state(
            step_count=5,
            force_assistance_steps=[],  # No forced steps
            segment_duration=0
        )
        deps = create_mock_deps()
        ctx = create_mock_context(state, deps)

        # Run 20 times with p_assistance = 0
        with patch('beagle.data_generation.studentv2.nodes.p_offtopic', return_value=0.0):
            with patch('beagle.data_generation.studentv2.nodes.p_assistance', return_value=0.0):
                node = MarkovNode()
                for _ in range(20):
                    result = asyncio.get_event_loop().run_until_complete(node.run(ctx))
                    self.assertNotIsInstance(result, AssistanceNode,
                        "Should never return AssistanceNode when p_assistance=0 and no forced steps")

        print("✓ disable-assistance correctly prevents random assistance")

    def test_assistance_can_trigger_with_high_probability(self):
        """Verify assistance CAN be triggered probabilistically (control test)."""
        from beagle.data_generation.studentv2.nodes import MarkovNode, AssistanceNode

        state = create_mock_state(
            step_count=5,
            force_assistance_steps=[]  # No forced steps
        )
        deps = create_mock_deps()
        ctx = create_mock_context(state, deps)

        assistance_count = 0
        # Run 50 times with p_assistance = 1.0 (always)
        with patch('beagle.data_generation.studentv2.nodes.p_offtopic', return_value=0.0):
            with patch('beagle.data_generation.studentv2.nodes.p_assistance', return_value=1.0):
                node = MarkovNode()
                for _ in range(50):
                    result = asyncio.get_event_loop().run_until_complete(node.run(ctx))
                    if isinstance(result, AssistanceNode):
                        assistance_count += 1

        # Should always trigger assistance
        self.assertEqual(assistance_count, 50,
            "Should always return AssistanceNode when p_assistance=1.0")

        print("✓ Assistance correctly triggers with high probability (control test passed)")


class TestFlagsIntegration(unittest.TestCase):
    """Integration test simulating actual CLI flag behavior."""

    def test_all_flags_together(self):
        """
        Test combination: --force-assistance-steps "4" --disable-assistance --disable-offtopic

        Expected behavior:
        - Step 4: SHOULD get assistance (forced)
        - Other steps: NO assistance (disabled)
        - No steps: off-topic (disabled)
        """
        from beagle.data_generation.studentv2.nodes import MarkovNode, AssistanceNode, OffTopicNode

        results = {'assistance': 0, 'offtopic': 0, 'other': 0}
        assistance_steps = []

        for step in range(10):
            state = create_mock_state(
                step_count=step,
                force_assistance_steps=[4],  # Force at step 4 (0-indexed)
                segment_duration=0
            )
            deps = create_mock_deps()
            ctx = create_mock_context(state, deps)

            with patch('beagle.data_generation.studentv2.nodes.p_offtopic', return_value=0.0):
                with patch('beagle.data_generation.studentv2.nodes.p_assistance', return_value=0.0):
                    node = MarkovNode()
                    result = asyncio.get_event_loop().run_until_complete(node.run(ctx))

                    if isinstance(result, AssistanceNode):
                        results['assistance'] += 1
                        assistance_steps.append(step)
                    elif isinstance(result, OffTopicNode):
                        results['offtopic'] += 1
                    else:
                        results['other'] += 1

        self.assertEqual(results['assistance'], 1, "Should have exactly 1 assistance (at step 4)")
        self.assertEqual(assistance_steps, [4], "Assistance should only occur at step 4")
        self.assertEqual(results['offtopic'], 0, "Should have 0 off-topic events")
        self.assertEqual(results['other'], 9, "Should have 9 other steps")

        print(f"✓ All flags work together correctly:")
        print(f"  Assistance: {results['assistance']} (forced at step 4)")
        print(f"  Off-topic: {results['offtopic']} (disabled)")
        print(f"  Other: {results['other']}")

    def test_multiple_forced_assistance_steps(self):
        """Test forcing assistance at multiple specific steps."""
        from beagle.data_generation.studentv2.nodes import MarkovNode, AssistanceNode

        forced_steps = [2, 5, 8]
        actual_assistance_steps = []

        for step in range(10):
            state = create_mock_state(
                step_count=step,
                force_assistance_steps=forced_steps,
                segment_duration=0
            )
            deps = create_mock_deps()
            ctx = create_mock_context(state, deps)

            with patch('beagle.data_generation.studentv2.nodes.p_offtopic', return_value=0.0):
                with patch('beagle.data_generation.studentv2.nodes.p_assistance', return_value=0.0):
                    node = MarkovNode()
                    result = asyncio.get_event_loop().run_until_complete(node.run(ctx))

                    if isinstance(result, AssistanceNode):
                        actual_assistance_steps.append(step)

        self.assertEqual(actual_assistance_steps, forced_steps,
            f"Should get assistance at exactly steps {forced_steps}")

        print(f"✓ Multiple forced assistance steps work correctly: {forced_steps}")

    def test_force_assistance_after_tutor_help_is_skipped(self):
        """Test that force_assistance is skipped right after receiving tutor help."""
        from beagle.data_generation.studentv2.nodes import MarkovNode, AssistanceNode

        # When just_received_tutor_help is True, interrupt checks are skipped
        state = create_mock_state(
            step_count=5,
            force_assistance_steps=[5],  # Would normally force assistance
            just_received_tutor_help=True,  # But we just got help
            segment_duration=0
        )
        deps = create_mock_deps()
        ctx = create_mock_context(state, deps)

        with patch('beagle.data_generation.studentv2.nodes.p_offtopic', return_value=0.0):
            with patch('beagle.data_generation.studentv2.nodes.p_assistance', return_value=0.0):
                node = MarkovNode()
                result = asyncio.get_event_loop().run_until_complete(node.run(ctx))

                # Should NOT be AssistanceNode because just_received_tutor_help skips interrupt checks
                self.assertNotIsInstance(result, AssistanceNode,
                    "Should skip assistance check when just_received_tutor_help is True")

        print("✓ Force assistance correctly skipped when just_received_tutor_help=True")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_empty_force_assistance_steps(self):
        """Test with empty force_assistance_steps list."""
        from beagle.data_generation.studentv2.nodes import MarkovNode, AssistanceNode

        for step in range(5):
            state = create_mock_state(
                step_count=step,
                force_assistance_steps=[],
                segment_duration=0
            )
            deps = create_mock_deps()
            ctx = create_mock_context(state, deps)

            with patch('beagle.data_generation.studentv2.nodes.p_offtopic', return_value=0.0):
                with patch('beagle.data_generation.studentv2.nodes.p_assistance', return_value=0.0):
                    node = MarkovNode()
                    result = asyncio.get_event_loop().run_until_complete(node.run(ctx))
                    self.assertNotIsInstance(result, AssistanceNode,
                        "Should never get assistance with empty force list and p_assistance=0")

        print("✓ Empty force_assistance_steps works correctly")

    def test_force_at_step_zero(self):
        """Test forcing assistance at step 0."""
        from beagle.data_generation.studentv2.nodes import MarkovNode, AssistanceNode

        state = create_mock_state(
            step_count=0,
            force_assistance_steps=[0],
            segment_duration=0
        )
        deps = create_mock_deps()
        ctx = create_mock_context(state, deps)

        with patch('beagle.data_generation.studentv2.nodes.p_offtopic', return_value=0.0):
            with patch('beagle.data_generation.studentv2.nodes.p_assistance', return_value=0.0):
                node = MarkovNode()
                result = asyncio.get_event_loop().run_until_complete(node.run(ctx))
                self.assertIsInstance(result, AssistanceNode,
                    "Should force assistance at step 0")

        print("✓ Force assistance at step 0 works correctly")

    def test_max_steps_terminates_before_force(self):
        """Test that max_steps check happens before force_assistance check."""
        from beagle.data_generation.studentv2.nodes import MarkovNode, AssistanceNode
        from pydantic_graph import End

        state = create_mock_state(
            step_count=30,  # At max_steps
            max_steps=30,
            force_assistance_steps=[30],  # Would force here, but max_steps should end first
        )
        deps = create_mock_deps()
        ctx = create_mock_context(state, deps)

        with patch('beagle.data_generation.studentv2.nodes.p_offtopic', return_value=0.0):
            with patch('beagle.data_generation.studentv2.nodes.p_assistance', return_value=0.0):
                node = MarkovNode()
                result = asyncio.get_event_loop().run_until_complete(node.run(ctx))
                # Should return End, not AssistanceNode
                self.assertIsInstance(result, End,
                    "Should return End at max_steps, even if force_assistance is set for that step")

        print("✓ max_steps correctly takes precedence over force_assistance")


def run_tests():
    """Run all tests and print summary."""
    print("=" * 70)
    print("CLI FLAGS INTERACTION TESTS")
    print("=" * 70)
    print("\nTesting: --force-assistance-steps + --disable-assistance + --disable-offtopic\n")

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestForceAssistanceWithDisabledAssistance))
    suite.addTests(loader.loadTestsFromTestCase(TestDisableOfftopic))
    suite.addTests(loader.loadTestsFromTestCase(TestDisableAssistance))
    suite.addTests(loader.loadTestsFromTestCase(TestFlagsIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if result.wasSuccessful():
        print("\n✓ ALL TESTS PASSED!")
        print("\nConfirmed behavior:")
        print("  - --force-assistance-steps works even with --disable-assistance")
        print("  - --disable-assistance blocks probabilistic assistance")
        print("  - --disable-offtopic blocks off-topic events")
        print("  - All three flags can be used together safely")
        print("  - Edge cases (step 0, max_steps, multiple steps) handled correctly")
        return True
    else:
        print(f"\n✗ {len(result.failures) + len(result.errors)} TESTS FAILED")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
