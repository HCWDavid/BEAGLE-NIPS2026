from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar, Union

from loguru import logger
from pydantic_ai import Agent
from pydantic_graph import BaseNode, End, GraphRunContext

from beagle.data_generation.studentv2.bkt.bkt import Observation
from beagle.data_generation.studentv2.deps import StudentDeps
from beagle.data_generation.studentv2.models import (
    AssistanceOutput, ExecutorOutput, StrategistOutput, TutorOutput,
    ConstructingOutput, AssessingOutput, DebuggingOutput,
    # Ablation: Merged pipeline schemas
    MergedConstructingOutput, MergedDebuggingOutput, MergedAssessingOutput
)
from beagle.data_generation.studentv2.profiles import get_profile
from beagle.data_generation.studentv2.state import (
    StudentState, EpisodicMemory, p_assistance, p_offtopic
)
from beagle.data_generation.studentv2.prompting import get_generator
from beagle.data_generation.studentv2.prompts.assembler import PromptAssembler
from beagle.data_generation.studentv2.output_parser import parse_execution_output

# Define paths to prompts
PROMPT_DIR = Path(__file__).parent / "prompts"

# Retry configuration for API-level errors
MAX_API_RETRIES = 5  # Increased from 3 to handle rate limits
INITIAL_RETRY_DELAY = 5.0  # Increased from 2.0 seconds

# Per-call model settings applied to every pydantic_ai Agent we construct.
# Match the paper's config (no temperature / max_tokens override → use the
# model's default sampling). The only safety net we add is a wall-clock timeout
# so a stuck stream raises an error the retry handler can catch instead of
# hanging the whole simulation. Schema-level max_length (in models.py) gives
# the decode-loop safety net.
from pydantic_ai import ModelSettings as _ModelSettings
DEFAULT_AGENT_MODEL_SETTINGS = _ModelSettings(
    timeout=60.0,
)

# Keywords indicating retryable API errors (not validation errors - pydantic_ai handles those)
# Covers: Gemini, OpenAI, Anthropic, and general HTTP errors
RETRYABLE_ERROR_KEYWORDS = [
    # General HTTP errors
    'rate limit',
    'timeout',
    'connection',
    'network',
    '408',
    '429',
    '500',
    '502',
    '503',
    '504',
    'overloaded',
    'service unavailable',
    'bad gateway',
    # Gemini-specific
    'malformed_function_call',
    'content field missing',
    'unexpected',
    'resource_exhausted',
    'recitation',
    'safety',
    'deadline_exceeded',
    'cancelled',
    # OpenAI-specific
    'insufficient_quota',
    'server_error',
    'engine_overloaded',
    'openai',
    'chatcompletion',  # catch OpenAI generic errors
    # Anthropic-specific (529 = overloaded)
    '529',
    'overloaded_error',
    'anthropic',
    # Generic
    'internal',
    'unavailable',
    'temporarily',
    'retry',
    'clienterror',  # Generic client errors
    'client_error',
]

T = TypeVar('T')


async def run_agent_with_retry(agent: Agent[None, T], prompt: str) -> T:
    """
    Run a pydantic_ai Agent with retry logic for API-level errors.

    Pydantic AI already handles validation retries (when model output doesn't match schema).
    This wrapper handles API-level failures like MALFORMED_FUNCTION_CALL, timeouts, etc.

    Args:
        agent: The pydantic_ai Agent to run
        prompt: The prompt to send

    Returns:
        The agent's output (not the full result, just result.output)

    Raises:
        Exception: If all retries are exhausted
    """
    last_exception = None

    for attempt in range(MAX_API_RETRIES):
        try:
            result = await agent.run(prompt)
            return result.output
        except Exception as e:
            last_exception = e
            error_str = str(e).lower()

            # Check if this is a retryable API error
            is_retryable = any(
                kw in error_str for kw in RETRYABLE_ERROR_KEYWORDS
            )

            if not is_retryable:
                logger.error(f"Non-retryable error: {e}")
                raise

            if attempt < MAX_API_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (2**
                                               attempt) + random.uniform(0, 1)
                logger.warning(
                    f"API error (attempt {attempt + 1}/{MAX_API_RETRIES}): {type(e).__name__}"
                )
                logger.info(f"Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"All {MAX_API_RETRIES} API retry attempts failed"
                )

    raise last_exception if last_exception else Exception("All retries failed")


# Module-level setting for comment style in prompts
# When True, uses sparse comments (realistic - matches real student behavior)
# When False, uses emotional comments (original behavior - "# hope this works", etc.)
# Default is False (emotional comments) - use --disable-emotional-comments flag to switch
# NOTE: This is also propagated to prompting module via set_sparse_comments()
USE_SPARSE_COMMENTS = False


def get_comment_guidance() -> str:
    """Get comment guidance based on USE_SPARSE_COMMENTS setting."""
    if USE_SPARSE_COMMENTS:
        return (
            "    - VERY FEW COMMENTS: Only ~15% of lines should have comments (most lines = NO comment)\n"
            "    - DO NOT comment every line! Real students don't explain every step\n"
            "    - If commenting, use SHORT notes: `# gravity`, `# velocity update`, `# KE = 1/2mv^2`\n"
            "    - NEVER use full sentences in comments"
        )
    else:
        return (
            "    - Minimal comments (maybe one \"# hope this works\")\n"
            "    - Add emotional comments showing doubt: `# idk`, `# ???`"
        )


def load_prompt(filename: str) -> str:
    """Load a prompt template from file."""
    return (PROMPT_DIR / filename).read_text().strip()


def load_common_errors() -> str:
    """Load common beginner errors from DCU dataset patterns."""
    try:
        return (PROMPT_DIR / "common_errors.txt").read_text().strip()
    except FileNotFoundError:
        return ""



# V34: REMOVED add_execution_summary function
# Output formatting is now handled by output_parser.py which provides:
# - Deterministic parsing of crash vs failure vs success
# - Single focused error display to prevent cognitive overload
# - Clear status headers as "Truth Anchors" for the LLM



# Module-level setting for epistemic blindness
# When True, Enacting state sees limited feedback (error summary only)
# When False (default), all states see full output
ENABLE_EPISTEMIC_BLINDNESS = True

# Memory Ablation Flags (for ablation study)
# When True, the corresponding memory system is disabled (empty strings passed to prompts)
DISABLE_MEMORY_STRATEGIST = False  # Disables thought_buffer, episodic_memories
DISABLE_MEMORY_EXECUTOR = False    # Disables monologue_buffer, agent_notes

# ABLATION: Merged Pipeline (tests if Strategist/Executor split prevents psychic debugging)
# When True, merges Strategist + Executor into a SINGLE LLM call per step
ENABLE_MERGED_PIPELINE = False


def filter_feedback_for_state(output: str, metacog_state: str) -> str:
    """
    Filter feedback based on metacognitive state (Epistemic Blindness).
    
    This implements the "Enacting Trap" mechanism:
    - Enacting: Sees only that an error occurred, not the details (forces trial-and-error)
    - Monitoring/Planning/Reflecting: Sees full output (can reason systematically)
    
    This is the key behavioral differentiator - Enacting students can't read
    error messages, so they MUST guess, while Monitoring students can analyze.
    
    Args:
        output: Raw execution output
        metacog_state: Current metacognitive state
    
    Returns:
        Filtered output based on epistemic visibility rules
    """
    if not ENABLE_EPISTEMIC_BLINDNESS:
        return output  # Feature disabled - return full output
    
    # Get the epistemic gating filter
    gen = get_generator()
    return gen.apply_blindness(metacog_state, output)


# =============================================================================
# Context Formatting Helpers for PromptAssembler
# =============================================================================

def _format_strategist_context(
    ctx,
    visible_output: str,
    knowledge_state: str,
    tutor_hint: str,
    thought_buffer: str,
    episodic_memory: str,
    pending_reflection: str
) -> str:
    """
    Formats all context variables into the {HISTORY} string for Strategist prompt.
    
    This consolidates the many context variables into a single string,
    keeping the Assembler focused on architecture while nodes.py handles logic.
    
    V34: Uses parsed execution output with clear status headers to prevent
    the LLM from misreading test results.
    """
    parts = []
    
    # Parse output for clear semantic signal (Truth Anchor)
    parsed = parse_execution_output(visible_output)
    
    parts.append(f"PROBLEM:\n{ctx.state.problem_description}")
    
    # Inject LOUD status header first (Truth Anchor)
    parts.append(f"LATEST EXECUTION STATUS:\n{parsed.status_header}\n{parsed.summary_line}")
    
    parts.append(f"CURRENT CODE:\n{ctx.state.current_code or '(No code yet)'}")
    parts.append(f"KNOWLEDGE STATE:\n{knowledge_state}")
    
    if tutor_hint:
        parts.append(tutor_hint)
    # V34: REMOVED recent_errors - causes zombie context / ghost errors
    # Memory ablation: skip thought_buffer and episodic_memory if disabled
    if not DISABLE_MEMORY_STRATEGIST:
        if thought_buffer:
            parts.append(thought_buffer)
        if episodic_memory:
            parts.append(episodic_memory)
    if pending_reflection:
        parts.append(pending_reflection)
        
    return "\n\n".join(parts)

def _format_executor_context(
    ctx,
    visible_output: str,
    knowledge_state: str,
    tutor_hint: str,
    comment_guidance: str,
    # Optional State-Specific Args
    previous_reflection: str = None,
    monologue_buffer: list = None,
    common_errors: str = None,
    previous_error_type: str = None,  # V34.4: Track error changes for context awareness
    raw_error_type: str = None,  # V34.4: Preserve error type for Enacting (filtered loses it)
    agent_notes: list = None,  # V34.5: Agent's own memory notes [(step, note), ...]
    emotional_expressions: str = None  # V35: Persona-driven emotional vocabulary
) -> str:
    """
    Formats the {SCREEN} context for Executor prompt (Equation 2).
    
    Handles variable inputs depending on Cognitive State:
    - DEBUGGING: previous_reflection, common_errors, monologue_buffer
    - CONSTRUCTING: common_errors only
    - ASSESSING: monologue_buffer only
    
    V34: Uses parsed output with focused single-error display for ALL students.
    V34.4: Added previous_error_type for context change awareness.
    V34.4: Added raw_error_type for Enacting state (filtered output loses type info).
    """
    parts = []
    
    # V34: Parse output for focused error display (Novice Spotlight - applies to ALL)
    # V36: Ensure visible_output is a string
    visible_output = visible_output if isinstance(visible_output, str) else ""
    parsed = parse_execution_output(visible_output)
    
    # V34.4: Use raw_error_type if provided (for Enacting state where filtered loses info)
    effective_error_type = raw_error_type if raw_error_type else parsed.error_type
    
    # 1. TRUTH ANCHOR: Put the PRIMARY ERROR FIRST in big prominent format
    # V34.3: Aggressive error type repetition (A/B test showed 4/4 vs 3/4 accuracy)
    # V34.4: Use effective_error_type to preserve type for Enacting state
    if effective_error_type not in ['success', 'draft', 'unknown', None, '']:
        # Extract error type for repetition
        error_type = effective_error_type.replace('_', ' ').title() if effective_error_type else 'Error'
        # Get primary error from raw if available
        primary_error = parsed.primary_error if parsed.primary_error else f"({error_type} occurred)"
        parts.append(
            f"THE ERROR IS: {error_type}\n"
            f"THE ERROR IS: {error_type}\n"
            f"THE ERROR IS: {error_type}\n"
            f"THE ERROR IS: {error_type}\n"
            f"THE ERROR IS: {error_type}\n"
            f"(Say {error_type} in your response, NOT a different error type)\n\n"
            f"***** THE ERROR YOU MUST REACT TO *****\n"
            f">>> {primary_error} <<<\n"
            f"*****************************************"
        )
        # V34.6: REMOVED error change/repeat alerts - they cause unnatural meta-commentary
        # Let the student react to the current error naturally based on their persona,
        # without forced phrases like "different error" or "this AGAIN!"
    elif parsed.error_type == 'success':
        parts.append(
            f"***** CURRENT STATUS: SUCCESS *****\n"
            f"All tests passed! Your code is correct."
        )
    
    # 2. Code and Console Output
    parts.append(f"CURRENT CODE:\n{ctx.state.current_code or '(No code yet)'}")
    parts.append(f"CONSOLE OUTPUT:\n{parsed.display_log}")
    parts.append(f"KNOWLEDGE STATE:\n{knowledge_state}")
    
    if tutor_hint:
        # V34: Add caveat that tutor may have been responding to a DIFFERENT/OLD error
        parts.append(
            f"{tutor_hint}\n\n"
            f"(NOTE: The tutor responded to a PREVIOUS error. "
            f"You must still react to the CURRENT error shown above in console output.)"
        )
        
    # 3. State-Specific Injections
    if previous_reflection:
        parts.append(f"YOUR PREVIOUS OBSERVATION:\n{previous_reflection}")
        
    if common_errors:
        parts.append(f"COMMON ERRORS TO AVOID:\n{common_errors}")
        
    # V34: REMOVED monologue buffer for DEBUGGING to prevent LLM from copying 
    # previous error patterns. The buffer was causing LLMs to repeat "I see [old error]"
    # instead of reading the current console output.
    # if monologue_buffer:
    #     ... (removed)

    # V34.5: Inject agent's own memory notes (unless ablation is enabled)
    if not DISABLE_MEMORY_EXECUTOR and agent_notes:
        notes_str = "\n".join([f"Step {step}: \"{note}\"" for step, note in agent_notes])
        parts.append(
            f"*** YOUR PAST NOTES (you wrote these) ***\n"
            f"{notes_str}\n"
            f"If you see a similar error, show recognition! Don't act confused."
        )

    # 3. Environmental Constraints
    style_parts = [comment_guidance]
    if emotional_expressions:
        style_parts.append(f"VOICE: {emotional_expressions}")
    parts.append(f"STYLE GUIDANCE:\n" + "\n".join(style_parts))
    
    return "\n\n".join(parts)




@dataclass
class MarkovNode(BaseNode[StudentState, StudentDeps, None]):
    """
    Determines the next high-level state (Cognitive + Metacognitive) using the Markov Model.

    Check order:
    1. Success check → End
    2. Max steps check → End
    3. just_received_tutor_help flag → Skip interrupt checks, reset flag
    4. P(Off-Topic) → OffTopicNode (checked first - full disengagement)
    5. P(Assistance) → AssistanceNode (checked second - partial engagement)
    6. Markov logic (pre-generated or sampled) → StrategistNode

    Pre-generated Sequence Mode:
    When ctx.state.pregenerated_sequence is set, the Markov chain (metacog states,
    cognitive actions) is consumed from the pre-generated sequence instead of sampled.
    However, interrupt decisions (assistance, off-topic) are ALWAYS made in real-time
    based on actual test progress (tests_passed / tests_total).
    """

    async def run(
        self, ctx: GraphRunContext[StudentState, StudentDeps]
    ) -> Union[StrategistNode, ExecutorNode, OffTopicNode, AssistanceNode, End]:
        # 1. Check for success - stop if problem is solved
        if ctx.state.execution_success:
            return End(None)

        # 2. Check for max steps
        if ctx.state.step_count >= ctx.state.max_steps:
            return End(None)

        # 3. Check if we just received tutor help - skip interrupt checks
        if ctx.state.just_received_tutor_help:
            ctx.state.just_received_tutor_help = False  # Reset flag
            # Continue to Markov logic (skip interrupt checks)
        else:
            # Calculate session progress based on unit tests passed
            if ctx.state.tests_total > 0:
                session_progress = ctx.state.tests_passed / ctx.state.tests_total
            else:
                session_progress = 0.0

            # 4. Check for Off-Topic (full disengagement) - checked FIRST
            offtopic_prob = p_offtopic(
                session_progress, ctx.deps.performance_level
            )
            if random.random() < offtopic_prob:
                return OffTopicNode()

            # 5. Check for Assistance (partial engagement) - checked SECOND
            # First check if assistance is forced at this step
            if ctx.state.step_count in ctx.state.force_assistance_steps:
                logger.info(f"Forcing assistance at step {ctx.state.step_count}")
                return AssistanceNode()

            assist_prob = p_assistance(
                session_progress, ctx.deps.performance_level
            )
            if random.random() < assist_prob:
                return AssistanceNode()

        # 6. Markov Logic - either from pre-generated sequence or sampled
        if ctx.state.pregenerated_sequence is not None:
            # Consume from pre-generated Markov sequence
            return self._consume_pregenerated_step(ctx)
        else:
            # Sample from Markov model
            return self._sample_markov_step(ctx)

    def _sample_markov_step(
        self, ctx: GraphRunContext[StudentState, StudentDeps]
    ) -> Union[StrategistNode, ExecutorNode]:
        """
        Sample the next Markov state from the Semi-Markov model.

        Returns:
            - StrategistNode if metacog phase changed (need new Goal/Mindset/Directive)
            - ExecutorNode if continuing same metacog phase (reuse existing directive)
        """
        if ctx.state.segment_step < ctx.state.segment_duration:
            # Continue segment - same metacog, just new cognitive action
            ctx.state.segment_step += 1
            cog = ctx.deps.markov_model.sample_action(
                ctx.state.current_metacognitive_state,
                ctx.deps.performance_level,
                previous_action=ctx.state.current_cognitive_state,
                is_first_step=(ctx.state.step_count == 0)
            )
            ctx.state.current_cognitive_state = cog

            # Skip StrategistNode - reuse existing Goal/Mindset/Directive
            return ExecutorNode()
        else:
            # New segment - sample BKT once and cache for entire metacog phase
            history = ctx.state.metacognitive_history
            new_meta = ctx.deps.markov_model.sample_next_state(
                history, ctx.deps.performance_level
            )
            duration = ctx.deps.markov_model.sample_duration(
                new_meta,
                ctx.deps.performance_level,
                duration_multiplier=ctx.deps.duration_multiplier
            )
            cog = ctx.deps.markov_model.sample_action(
                new_meta,
                ctx.deps.performance_level,
                previous_action=ctx.state.current_cognitive_state,
                is_first_step=(ctx.state.step_count == 0)
            )
            ctx.state.current_metacognitive_state = new_meta
            ctx.state.segment_duration = duration
            ctx.state.segment_step = 1
            ctx.state.metacognitive_history.append(new_meta)

            # Cache BKT sampling for this entire metacog phase
            if ctx.deps.bkt and ctx.state.required_kcs:
                ctx.state.cached_knowledge_state = ctx.deps.bkt.get_sampled_knowledge_state(
                    ctx.state.required_kcs
                )
            else:
                ctx.state.cached_knowledge_state = "No knowledge tracking available."

            ctx.state.current_cognitive_state = cog

            # New metacog phase - need fresh Goal/Mindset/Directive from StrategistNode
            return StrategistNode()

    def _consume_pregenerated_step(
        self, ctx: GraphRunContext[StudentState, StudentDeps]
    ) -> Union[StrategistNode, ExecutorNode, End]:
        """
        Consume the next Markov step from the pre-generated sequence.

        Only consumes metacog states and cognitive actions.
        Interrupt decisions are handled separately (real-time).

        Returns:
            - StrategistNode if metacog phase changed (is_segment_start)
            - ExecutorNode if continuing same metacog phase
            - End if sequence exhausted
        """
        sequence = ctx.state.pregenerated_sequence
        idx = ctx.state.sequence_index

        # Check if we've exhausted the sequence
        if idx >= len(sequence.steps):
            return End(None)

        # Get the next step from the sequence
        step = sequence.steps[idx]
        ctx.state.sequence_index += 1

        # Update state from pre-generated step
        ctx.state.current_metacognitive_state = step.metacog_state
        ctx.state.current_cognitive_state = step.cognitive_action

        # Track segment boundaries and decide next node
        if step.is_segment_start:
            # New metacog phase - need fresh Goal/Mindset/Directive
            ctx.state.metacognitive_history.append(step.metacog_state)
            return StrategistNode()
        else:
            # Continuing same metacog phase - reuse existing directive
            return ExecutorNode()


@dataclass
class OffTopicNode(BaseNode[StudentState, StudentDeps, None]):
    """
    Handles off-topic behavior (student disengages from the task).

    This is a simple idle step - no LLM call, just record the idle behavior
    and return to MarkovNode.

    Based on LAK24 data:
    - Peak late in session (μ=0.73)
    - Low performers go off-topic more (9.2% vs 3.7% peak)
    """

    async def run(
        self, ctx: GraphRunContext[StudentState, StudentDeps]
    ) -> MarkovNode:
        # Record off-topic step in history
        ctx.state.history.append(
            {
                'cognitive_state': 'OFF_TOPIC',
                'metacognitive_state': ctx.state.current_metacognitive_state,
                'goal': 'Disengaged from task',
                'mindset': 'Distracted',
                'directive': 'Not working on the problem',
                'monologue': '(Student went off-topic - idle step)',
                'code': ctx.state.current_code,  # Code unchanged
                'output': '(No execution - off-topic)',
                'error': '',
                'success': None
            }
        )

        # Increment counters
        ctx.state.step_count += 1
        ctx.state.offtopic_count += 1

        return MarkovNode()


@dataclass
class AssistanceNode(BaseNode[StudentState, StudentDeps, None]):
    """
    Student asks the tutor for help.

    This is Turn 1 of the two-turn assistance flow:
    1. AssistanceNode: Student formulates goal, mindset, directive, and question
    2. TutorNode: Tutor responds with a hint
    3. Next turn: Student applies the hint (handled by StrategistNode/ExecutorNode)

    Receives same inputs as StrategistNode for consistency.
    Output: AssistanceOutput (goal, mindset, directive, question)
    """

    async def run(
        self, ctx: GraphRunContext[StudentState, StudentDeps]
    ) -> TutorNode:
        # Get Profile (uses performance_level for behavioral traits)
        profile = get_profile(ctx.deps.performance_level)

        # Load Prompts
        system_prompt = load_prompt("assistance_system.txt")
        user_prompt_template = load_prompt("assistance_user.txt")

        # Get knowledge state (cached or fresh depending on cache_bkt setting)
        if ctx.deps.cache_bkt:
            knowledge_state = ctx.state.cached_knowledge_state or "No knowledge tracking available."
        else:
            # Sample fresh BKT per cognitive step (old behavior)
            if ctx.deps.bkt and ctx.state.required_kcs:
                knowledge_state = ctx.deps.bkt.get_sampled_knowledge_state(
                    ctx.state.required_kcs
                )
            else:
                knowledge_state = "No knowledge tracking available."

        # Format User Prompt (same inputs as StrategistNode)
        prompt = user_prompt_template.format(
            problem_description=ctx.state.problem_description,
            current_metacognitive_state=ctx.state.current_metacognitive_state,
            performance_level=ctx.deps.performance_level,
            current_code=ctx.state.current_code,
            last_output=ctx.state.last_output,
            knowledge_state=knowledge_state,
            planning_style=profile.planning_style,
            reflecting_style=profile.reflecting_style,
            monitoring_style=profile.monitoring_style,
            debugging_style=profile.debugging_style,
            persona_description=profile.persona_description
        )

        # Create agent
        agent = Agent(
            ctx.deps.llm_client.pydantic_model,
            output_type=AssistanceOutput,
            system_prompt=system_prompt,
            model_settings=DEFAULT_AGENT_MODEL_SETTINGS,
        )

        # Run agent with retry for API errors
        output = await run_agent_with_retry(agent, prompt)

        # Store the question for TutorNode
        ctx.state.pending_tutor_question = output.question

        # Record assistance request in history
        ctx.state.history.append(
            {
                'cognitive_state': 'ASSISTANCE',
                'metacognitive_state': ctx.state.current_metacognitive_state,
                'goal': output.goal,
                'mindset': output.mindset,
                'directive': output.directive,
                'question': output.question,
                'code': ctx.state.current_code,  # Code unchanged
                'output': '(Asking tutor for help)',
                'error': '',
                'success': None
            }
        )

        # Increment counters
        ctx.state.step_count += 1
        ctx.state.assistance_count += 1

        return TutorNode()


@dataclass
class TutorNode(BaseNode[StudentState, StudentDeps, None]):
    """
    Tutor responds to the student's question.

    Receives:
    - Student's question (from pending_tutor_question)
    - Current code and output
    - BKT knowledge state

    Output: TutorOutput (response)
    Sets: tutor_response and just_received_tutor_help flag

    Tutor Strategy:
    - If ctx.deps.tutor_strategy is set, uses the pluggable tutor (for controlled experiments)
    - Otherwise, falls back to the default LLM-based tutor
    """

    async def run(
        self, ctx: GraphRunContext[StudentState, StudentDeps]
    ) -> MarkovNode:
        # Determine strategy to use
        strategy = ctx.deps.tutor_strategy
        if strategy is None:
            # Fallback to default LLM tutor (replicates original behavior)
            from beagle.tutor.default_tutor import DefaultLLMTutor
            strategy = DefaultLLMTutor(ctx.deps.llm_client)

        # Generate hint using strategy (returns full TutorResponse)
        tutor_response = await self._run_with_strategy(ctx, strategy)

        # Store tutor hint string for next turn (state.tutor_response is a str)
        ctx.state.tutor_response = tutor_response.hint
        ctx.state.just_received_tutor_help = True

        # Clear the pending question
        ctx.state.pending_tutor_question = ""

        # --- Tutor-mediated EFI unblock ---
        # An EFI-blocked KC is otherwise unrecoverable (sample() always returns
        # WRONG → update() freezes P(L) at the prior). The pedagogically grounded
        # release path: when the tutor delivers a hint, unblock the SPECIFIC KC
        # the tutor targeted (TutorResponse.target_kc, set to the lowest-mastery
        # required KC by default). Falls back to all EFI-active required_kcs if
        # the strategy did not declare a target.
        if ctx.deps.bkt:
            unblocked = []
            target_kc = tutor_response.target_kc
            kcs_to_try = (
                [target_kc] if target_kc else (ctx.state.required_kcs or [])
            )
            for kc_id in kcs_to_try:
                if ctx.deps.bkt.unblock_efi(
                    kc_id, source=f"tutor[{strategy.tutor_type}]"
                ):
                    unblocked.append(kc_id)
            if unblocked and ctx.state.history:
                ctx.state.history[-1]['efi_unblocked_by_tutor'] = unblocked

        # Update history with tutor response
        if ctx.state.history:
            ctx.state.history[-1]['tutor_response'] = tutor_response.hint
            # Track tutor metadata
            ctx.state.history[-1]['tutor_type'] = strategy.tutor_type
            ctx.state.history[-1]['tutor_target_kc'] = tutor_response.target_kc

        # Force transition to CONSTRUCTING to apply the hint
        # (Usually Enacting, but let Markov logic decide next phase)
        return MarkovNode()

    async def _run_with_strategy(
        self, 
        ctx: GraphRunContext[StudentState, StudentDeps],
        strategy: 'TutorStrategy'
    ) -> str:
        """
        Generate hint using the provided tutor strategy.
        """
        from beagle.tutor.base import TutorContext, HintHistoryEntry

        # Build TutorContext from StudentState
        tutor_context = TutorContext(
            problem_description=ctx.state.problem_description,
            problem_id=ctx.state.problem_id,
            current_code=ctx.state.current_code,
            last_output=ctx.state.last_output,
            last_error=ctx.state.last_output if not ctx.state.execution_success else "",
            student_question=ctx.state.pending_tutor_question,
            bkt=ctx.deps.bkt,
            required_kcs=ctx.state.required_kcs,
            performance_level=ctx.deps.performance_level,
            current_step=len(ctx.state.history) + 1,
            tests_passed=ctx.state.tests_passed,
            tests_total=ctx.state.tests_total,
            current_metacognitive_state=ctx.state.current_metacognitive_state,
            current_cognitive_state=ctx.state.current_cognitive_state,
        )

        # Generate hint using strategy
        tutor_response = await strategy.generate_hint(tutor_context)

        logger.debug(
            f"Tutor [{strategy.tutor_type}] "
            f"scaffold={tutor_response.scaffold_level.name} "
            f"target_kc={tutor_response.target_kc}: {tutor_response.hint[:50]}..."
        )

        return tutor_response


@dataclass
class StrategistNode(BaseNode[StudentState, StudentDeps, None]):
    """
    Block-based Strategist using PromptAssembler.
    
    Prompts are assembled from composable blocks in prompts/blocks/:
    - system/static/ (persona, rules)
    - system/dynamic/mandate/ (12 metacog×cog files)
    - user/task/ (strategist.txt)
    - user/context/ (history template)
    
    JIT execution still happens for sighted states (DEBUGGING/ASSESSING).
    """

    async def run(
        self, ctx: GraphRunContext[StudentState, StudentDeps]
    ) -> ExecutorNode:
        metacog = ctx.state.current_metacognitive_state  # Planning/Monitoring/Reflecting/Enacting
        cog = ctx.state.current_cognitive_state  # CONSTRUCTING/DEBUGGING/ASSESSING

        # --- ABLATION: Merged Pipeline ---
        # When enabled, skip Strategist LLM call and let Executor do everything
        if ENABLE_MERGED_PIPELINE:
            # Set placeholder strategy values - Executor will generate its own
            ctx.state.current_goal = "(merged - generate own goal)"
            ctx.state.current_mindset = "(merged - generate own mindset)"
            ctx.state.current_directive = "(merged - no separate directive)"
            return ExecutorNode()

        # --- JIT EXECUTION for Sighted states (DEBUGGING/ASSESSING) ---
        visible_output = ctx.state.last_output
        if cog.lower() in ["debugging", "assessing"]:
            if ctx.state.last_output == "(Code drafted but not executed)":
                if ctx.state.current_code and ctx.deps.ide_oracle and ctx.state.problem_id:
                    result = ctx.deps.ide_oracle.test_code(
                        code=ctx.state.current_code,
                        problem_id=ctx.state.problem_id
                    )
                    visible_output = result.stdout or result.captured_output
                    # V34: No summary here - output_parser handles formatting in context formatters

                    # Update state with execution results
                    ctx.state.last_output = visible_output
                    ctx.state.execution_success = result.passed
                    ctx.state.tests_passed = result.passed_tests
                    ctx.state.tests_total = result.total_tests

                    # Cache result for EnvironmentNode
                    ctx.state._cached_execution_result = result

        # --- KNOWLEDGE STATE ---
        if ctx.deps.cache_bkt:
            knowledge_state = ctx.state.cached_knowledge_state or "No knowledge tracking available."
        else:
            if ctx.deps.bkt and ctx.state.required_kcs:
                knowledge_state = ctx.deps.bkt.get_sampled_knowledge_state(
                    ctx.state.required_kcs
                )
            else:
                knowledge_state = "No knowledge tracking available."

        # --- BUILD CONTEXT COMPONENTS ---
        # Tutor hint
        tutor_hint = ""
        if ctx.state.tutor_response:
            tutor_hint = f"\n\n**TUTOR HINT (from your previous question):**\n{ctx.state.tutor_response}\n\nConsider this hint when forming your strategy."


        # V34: REMOVED "recent_errors" injection (ZOMBIE CONTEXT BUG FIX)
        # Previously, we injected errors from the last 10 history steps here.
        # This caused "ghost errors" where the LLM would react to bugs that 
        # had already been fixed. The LLM should rely on:
        # 1. Episodic Memory (internalized lessons like "I forgot self.x")
        # 2. Current Output (external reality from parse_execution_output)
        # NOT on raw error history injection.

        # Pending reflection
        pending_reflection_context = ""
        if ctx.state.pending_reflection:
            pending_reflection_context = f"\n\nYOUR PREVIOUS OBSERVATION (from when you ran the code):\n{ctx.state.pending_reflection}\n\nUse this observation to inform your next action."

        # Thought buffer (anti-repetition)
        thought_buffer_context = ""
        if ctx.state.thought_buffer:
            recent_thoughts = ctx.state.thought_buffer[-ctx.state.THOUGHT_BUFFER_SIZE:]
            thought_buffer_context = "\n\n*** DO NOT REPEAT THESE RECENT THOUGHTS ***\n"
            for i, thought in enumerate(recent_thoughts, 1):
                thought_buffer_context += f"{i}. {thought}\n"
            thought_buffer_context += "\nYour new goal/mindset MUST be different from the above. Make actual progress!"

        # Episodic memory (anti-amnesia)
        episodic_memory_context = ""
        if ctx.state.episodic_memories and visible_output:
            relevant_memories = []
            for memory in ctx.state.episodic_memories:
                if memory.error_pattern.lower() in visible_output.lower():
                    relevant_memories.append(memory)

            if relevant_memories:
                episodic_memory_context = "\n\n*** YOU REMEMBER THIS ERROR! ***\n"
                for memory in relevant_memories:
                    episodic_memory_context += f"- In Step {memory.step_learned}, you learned: \"{memory.realization}\"\n"
                    if memory.fix_applied:
                        episodic_memory_context += f"  The fix was: {memory.fix_applied}\n"
                episodic_memory_context += "\nYou've seen this before! Apply what you learned."

        # --- ASSEMBLE PROMPT USING BLOCK SYSTEM ---
        # 1. Instantiate assembler with persona_type
        assembler = PromptAssembler(persona_type=ctx.deps.persona_type)
        
        # 2. Format context using helper
        context_str = _format_strategist_context(
            ctx=ctx,
            visible_output=visible_output,
            knowledge_state=knowledge_state,
            tutor_hint=tutor_hint,
            thought_buffer=thought_buffer_context,
            episodic_memory=episodic_memory_context,
            pending_reflection=pending_reflection_context
        )

        # 3. Assemble prompts (Equation 1)
        system_prompt, user_prompt = assembler.assemble_strategist_prompts(
            meta_state=metacog,
            cog_state=cog,
            context_str=context_str
        )

        # 4. Create and run agent
        agent = Agent(
            ctx.deps.llm_client.pydantic_model,
            output_type=StrategistOutput,
            system_prompt=system_prompt,
            model_settings=DEFAULT_AGENT_MODEL_SETTINGS,
        )

        # Run agent with retry for API errors
        output = await run_agent_with_retry(agent, user_prompt)

        # Update state with strategy output
        ctx.state.current_goal = output.goal
        ctx.state.current_mindset = output.mindset
        ctx.state.current_directive = output.directive

        # --- V22: Update Thought Buffer ---
        thought_summary = f"Goal: {output.goal} | Mindset: {output.mindset}"
        ctx.state.thought_buffer.append(thought_summary)
        if len(ctx.state.thought_buffer) > ctx.state.THOUGHT_BUFFER_SIZE:
            ctx.state.thought_buffer.pop(0)

        return ExecutorNode()


@dataclass
class ExecutorNode(BaseNode[StudentState, StudentDeps, None]):
    """
    Generates output based on cognitive state:
    - CONSTRUCTING: Code only (no execution, no monologue)
    - ASSESSING: Reflection only (execution happens, but NO code changes)
    - DEBUGGING: Code + Monologue (execution happens, student sees error and fixes)

    This design prevents "psychic debugging" where the LLM identifies bugs
    before seeing execution output.
    """

    async def run(
        self, ctx: GraphRunContext[StudentState, StudentDeps]
    ) -> EnvironmentNode:
        # Get Profile (uses performance_level for behavioral traits)
        profile = get_profile(ctx.deps.performance_level)

        # Get knowledge state (cached or fresh depending on cache_bkt setting)
        if ctx.deps.cache_bkt:
            knowledge_state = ctx.state.cached_knowledge_state or "No knowledge tracking available."
        else:
            if ctx.deps.bkt and ctx.state.required_kcs:
                knowledge_state = ctx.deps.bkt.get_sampled_knowledge_state(ctx.state.required_kcs)
            else:
                knowledge_state = "No knowledge tracking available."

        # Check for tutor response (from previous assistance flow)
        tutor_hint = ""
        if ctx.state.tutor_response:
            tutor_hint = f"\n\n**TUTOR HINT (from your previous question):**\n{ctx.state.tutor_response}\n\nApply this hint in your code."

        # Dispatch to appropriate handler based on cognitive state
        # Each handler uses PromptAssembler directly
        if ctx.state.current_cognitive_state == "CONSTRUCTING":
            output = await self._run_constructing(ctx, profile, knowledge_state, tutor_hint)
        elif ctx.state.current_cognitive_state == "ASSESSING":
            output = await self._run_assessing(ctx, profile, knowledge_state, tutor_hint)
        elif ctx.state.current_cognitive_state == "DEBUGGING":
            output = await self._run_debugging(ctx, profile, knowledge_state, tutor_hint)
        else:
            # Fallback for unknown cognitive states - FAIL LOUDLY
            output = await self._run_generic(ctx, profile, knowledge_state, tutor_hint)

        # Record history entry
        history_entry = {
            'cognitive_state': ctx.state.current_cognitive_state,
            'metacognitive_state': ctx.state.current_metacognitive_state,
            'goal': ctx.state.current_goal,
            'mindset': ctx.state.current_mindset,
            'directive': ctx.state.current_directive,
            'monologue': ctx.state.current_monologue,
            'code': ctx.state.current_code,
            # V33: Add memory state for debugging/diagnostics
            'episodic_memories': [
                {'error_pattern': m.error_pattern, 'realization': m.realization, 'step_learned': m.step_learned}
                for m in ctx.state.episodic_memories
            ] if ctx.state.episodic_memories else [],
            'thought_buffer': list(ctx.state.thought_buffer) if ctx.state.thought_buffer else [],
            'monologue_buffer': list(ctx.state.monologue_buffer) if ctx.state.monologue_buffer else [],
            # V34.5: Add agent memory notes
            'agent_notes': list(ctx.state.agent_notes),
            # V36: Add error tracking fields
            'last_error_type': ctx.state.last_error_type,
            'last_error_message': ctx.state.last_error_message
        }

        # Add reflection if ASSESSING
        if ctx.state.current_cognitive_state == "ASSESSING" and ctx.state.pending_reflection:
            history_entry['reflection'] = ctx.state.pending_reflection

        # V36: Add error_seen if DEBUGGING
        if ctx.state.current_cognitive_state == "DEBUGGING":
            error_seen = getattr(ctx.state, 'current_error_seen', None)
            if error_seen:
                history_entry['error_seen'] = error_seen

        ctx.state.history.append(history_entry)
        ctx.state.step_count += 1

        # Incremental checkpoint save
        if ctx.deps.checkpoint_path:
            ctx.state.save_checkpoint(ctx.deps.checkpoint_path)

        return EnvironmentNode()

    async def _run_constructing(self, ctx, profile, knowledge_state, tutor_hint):
        """
        CONSTRUCTING: Generate code only. No execution happens.
        No monologue since student is just drafting.
        """
        # --- Context Formatting ---
        # V36: Use last execution output as context (what happened when code last ran)
        prev_output = ctx.state.last_output if ctx.state.last_output else "(No execution - drafting code)"
        
        context_str = _format_executor_context(
            ctx=ctx,
            visible_output=prev_output,  # V36: Show last execution result as context
            knowledge_state=knowledge_state,
            tutor_hint=tutor_hint,
            comment_guidance=get_comment_guidance(),
            common_errors=load_common_errors(),
            previous_error_type=ctx.state.last_error_type,  # V36: Error from last execution
            agent_notes=ctx.state.agent_notes,  # V36: Pass agent notes
            emotional_expressions=profile.emotional_expressions
            # No previous_reflection, no monologue_buffer for constructing
        )

        # --- Assembler Call ---
        assembler = PromptAssembler(persona_type=ctx.deps.persona_type)
        
        strategy_packet = {
            "goal": ctx.state.current_goal,
            "mindset": ctx.state.current_mindset,
            "directive": ctx.state.current_directive
        }

        system_prompt, user_prompt = assembler.assemble_executor_prompts(
            meta_state=ctx.state.current_metacognitive_state,
            cog_state="Constructing",
            strategy_output=strategy_packet,
            context_str=context_str
        )

        # --- ABLATION: Merged Pipeline uses different output schema ---
        if ENABLE_MERGED_PIPELINE:
            # Use merged schema - includes goal/mindset that Strategist would have generated
            agent = Agent(
                ctx.deps.llm_client.pydantic_model,
                output_type=MergedConstructingOutput,
                system_prompt=system_prompt,
                model_settings=DEFAULT_AGENT_MODEL_SETTINGS,
            )
            output = await run_agent_with_retry(agent, user_prompt)
            
            # Update state from merged output
            ctx.state.current_goal = output.goal
            ctx.state.current_mindset = output.mindset
            if output.code:
                ctx.state.current_code = output.code
            ctx.state.current_monologue = output.thinking  # Use 'thinking' field (like baselines)
            ctx.state.pending_reflection = ""
            return output
        
        # --- Normal: Agent Execution with separate schema ---
        agent = Agent(
            ctx.deps.llm_client.pydantic_model,
            output_type=ConstructingOutput,
            system_prompt=system_prompt,
            model_settings=DEFAULT_AGENT_MODEL_SETTINGS,
        )

        output = await run_agent_with_retry(agent, user_prompt)

        # --- State Updates ---
        if output.code:
            ctx.state.current_code = output.code
        ctx.state.current_monologue = output.monologue  # V37: Save monologue for blind error testing
        ctx.state.pending_reflection = ""

        return output

    async def _run_assessing(self, ctx, profile, knowledge_state, tutor_hint):
        """
        ASSESSING: Execute code FIRST, then generate reflection on the output.
        The reflection is stored and carried forward to inform future debugging.
        
        Logic preserved: JIT Exec -> Assembler -> Agent -> Update State
        """
        # --- 1. PRESERVED: JIT Execution ---
        fresh_output = ctx.state.last_output

        if ctx.state.current_code and ctx.deps.ide_oracle and ctx.state.problem_id:
            result = ctx.deps.ide_oracle.test_code(
                code=ctx.state.current_code, problem_id=ctx.state.problem_id
            )
            fresh_output = result.stdout or result.captured_output or ""
            # V34: No summary here - output_parser handles formatting in context formatters

            # Update state with execution results
            ctx.state.last_output = fresh_output
            ctx.state.execution_success = result.passed
            ctx.state.tests_passed = result.passed_tests
            ctx.state.tests_total = result.total_tests
            ctx.state._cached_execution_result = result
            
            # V36: Save parsed error to global state
            if isinstance(fresh_output, str):
                parsed = parse_execution_output(fresh_output)
                ctx.state.last_error_type = parsed.error_type or ""
                ctx.state.last_error_message = parsed.primary_error or ""

        # --- 2. Context Formatting ---
        filtered_output = filter_feedback_for_state(fresh_output, ctx.state.current_metacognitive_state)
        
        context_str = _format_executor_context(
            ctx=ctx,
            visible_output=filtered_output,
            knowledge_state=knowledge_state,
            tutor_hint=tutor_hint,
            comment_guidance=get_comment_guidance(),
            monologue_buffer=ctx.state.monologue_buffer if ctx.state.monologue_buffer else None,
            raw_error_type=ctx.state.last_error_type,  # V36: Pass current error type
            agent_notes=ctx.state.agent_notes,  # V36: Pass agent notes
            emotional_expressions=profile.emotional_expressions
            # No previous_reflection, no common_errors for assessing
        )

        # --- 3. Assembler Call ---
        assembler = PromptAssembler(persona_type=ctx.deps.persona_type)
        
        strategy_packet = {
            "goal": ctx.state.current_goal,
            "mindset": ctx.state.current_mindset,
            "directive": ctx.state.current_directive
        }

        system_prompt, user_prompt = assembler.assemble_executor_prompts(
            meta_state=ctx.state.current_metacognitive_state,
            cog_state="Assessing",
            strategy_output=strategy_packet,
            context_str=context_str
        )

        # --- ABLATION: Merged Pipeline uses different output schema ---
        if ENABLE_MERGED_PIPELINE:
            agent = Agent(
                ctx.deps.llm_client.pydantic_model,
                output_type=MergedAssessingOutput,
                system_prompt=system_prompt,
                model_settings=DEFAULT_AGENT_MODEL_SETTINGS,
            )
            output = await run_agent_with_retry(agent, user_prompt)
            
            # Update state from merged output
            ctx.state.current_goal = output.goal
            ctx.state.current_mindset = output.mindset
            ctx.state.current_monologue = output.thinking
            ctx.state.pending_reflection = output.thinking
            
            if output.thinking:
                ctx.state.monologue_buffer.append(output.thinking)
                if len(ctx.state.monologue_buffer) > ctx.state.MONOLOGUE_BUFFER_SIZE:
                    ctx.state.monologue_buffer.pop(0)
            return output

        # --- 4. Agent Execution ---
        agent = Agent(
            ctx.deps.llm_client.pydantic_model,
            output_type=AssessingOutput,
            system_prompt=system_prompt,
            model_settings=DEFAULT_AGENT_MODEL_SETTINGS,
        )

        output = await run_agent_with_retry(agent, user_prompt)

        # --- 5. PRESERVED: State Updates ---
        ctx.state.current_monologue = output.reflection
        ctx.state.pending_reflection = output.reflection  # Carry forward

        if output.reflection:
            ctx.state.monologue_buffer.append(output.reflection)
            if len(ctx.state.monologue_buffer) > ctx.state.MONOLOGUE_BUFFER_SIZE:
                ctx.state.monologue_buffer.pop(0)

        # NOTE: We do NOT modify current_code here - ASSESSING cannot change code

        return output

    async def _run_debugging(self, ctx, profile, knowledge_state, tutor_hint):
        """
        DEBUGGING: Generate both code and monologue. Code runs and student sees output.
        
        Logic preserved: JIT Exec -> Assembler -> Agent -> Update State
        """
        # --- 1. V34 FIX: ALWAYS re-execute for DEBUGGING ---
        # Previously this only ran if last_output == "(Code drafted but not executed)"
        # But consecutive DEBUGGING steps would use STALE output from previous step!
        # Now we ALWAYS re-execute to get fresh output for the current code.
        fresh_output = ctx.state.last_output
        if ctx.state.current_code and ctx.deps.ide_oracle and ctx.state.problem_id:
            result = ctx.deps.ide_oracle.test_code(
                code=ctx.state.current_code, problem_id=ctx.state.problem_id
            )
            fresh_output = result.stdout or result.captured_output or ""

            # Update state with execution results
            ctx.state.last_output = fresh_output
            ctx.state.execution_success = result.passed
            ctx.state.tests_passed = result.passed_tests
            ctx.state.tests_total = result.total_tests
            ctx.state._cached_execution_result = result
            
            # V36: Save parsed error to global state
            if isinstance(fresh_output, str):
                parsed = parse_execution_output(fresh_output)
                ctx.state.last_error_type = parsed.error_type or ""
                ctx.state.last_error_message = parsed.primary_error or ""

        # --- 2. Context Formatting ---
        # Apply Epistemic Blindness Filter
        filtered_output = filter_feedback_for_state(fresh_output, ctx.state.current_metacognitive_state)
        
        # V36: Use global state for error tracking
        context_str = _format_executor_context(
            ctx=ctx,
            visible_output=filtered_output,
            knowledge_state=knowledge_state,
            tutor_hint=tutor_hint,
            comment_guidance=get_comment_guidance(),
            previous_reflection=ctx.state.pending_reflection,
            common_errors=load_common_errors(),
            monologue_buffer=ctx.state.monologue_buffer if ctx.state.monologue_buffer else None,
            previous_error_type=ctx.state.last_error_type,  # V36: Global state
            raw_error_type=ctx.state.last_error_type,  # V36: Current JIT error
            agent_notes=ctx.state.agent_notes,  # Now explicit in StudentState
            emotional_expressions=profile.emotional_expressions
        )

        # --- 3. Assembler Call ---
        assembler = PromptAssembler(persona_type=ctx.deps.persona_type)
        
        strategy_packet = {
            "goal": ctx.state.current_goal,
            "mindset": ctx.state.current_mindset,
            "directive": ctx.state.current_directive
        }

        system_prompt, user_prompt = assembler.assemble_executor_prompts(
            meta_state=ctx.state.current_metacognitive_state,
            cog_state="Debugging",
            strategy_output=strategy_packet,
            context_str=context_str
        )

        # --- ABLATION: Merged Pipeline uses different output schema ---
        if ENABLE_MERGED_PIPELINE:
            agent = Agent(
                ctx.deps.llm_client.pydantic_model,
                output_type=MergedDebuggingOutput,
                system_prompt=system_prompt,
                model_settings=DEFAULT_AGENT_MODEL_SETTINGS,
            )
            output = await run_agent_with_retry(agent, user_prompt)
            
            # Update state from merged output
            ctx.state.current_goal = output.goal
            ctx.state.current_mindset = output.mindset
            ctx.state.current_monologue = output.thinking
            if output.code:
                ctx.state.current_code = output.code
            ctx.state.current_error_seen = output.error_seen
            
            if output.thinking:
                ctx.state.monologue_buffer.append(output.thinking)
                if len(ctx.state.monologue_buffer) > ctx.state.MONOLOGUE_BUFFER_SIZE:
                    ctx.state.monologue_buffer.pop(0)
            
            # Store memory note
            if hasattr(output, 'memory_note') and output.memory_note:
                step_num = len(ctx.state.history) + 1
                ctx.state.agent_notes.append((step_num, output.memory_note))
                if len(ctx.state.agent_notes) > 10:
                    ctx.state.agent_notes.pop(0)
            
            ctx.state.pending_reflection = ""
            return output

        # --- 4. Agent Execution ---
        agent = Agent(
            ctx.deps.llm_client.pydantic_model,
            output_type=DebuggingOutput,
            system_prompt=system_prompt,
            model_settings=DEFAULT_AGENT_MODEL_SETTINGS,
        )

        output = await run_agent_with_retry(agent, user_prompt)

        # --- 5. PRESERVED: State Updates ---
        ctx.state.current_monologue = output.monologue
        if output.code:
            ctx.state.current_code = output.code

        # V36: Save error_seen for history logging and validation
        ctx.state.current_error_seen = output.error_seen

        if output.monologue:
            ctx.state.monologue_buffer.append(output.monologue)
            if len(ctx.state.monologue_buffer) > ctx.state.MONOLOGUE_BUFFER_SIZE:
                ctx.state.monologue_buffer.pop(0)

        # V34.5: Store agent memory note (V36: agent_notes now explicit in StudentState)
        if hasattr(output, 'memory_note') and output.memory_note:
            step_num = len(ctx.state.history) + 1
            ctx.state.agent_notes.append((step_num, output.memory_note))
            # Keep only last 10 notes to avoid context bloat
            if len(ctx.state.agent_notes) > 10:
                ctx.state.agent_notes.pop(0)

        ctx.state.pending_reflection = ""  # Consumed

        return output

    async def _run_generic(self, ctx, profile, knowledge_state, tutor_hint):
        """
        Fallback for unknown cognitive states - FAIL LOUDLY.
        """
        raise ValueError(
            f"Unknown cognitive state: {ctx.state.current_cognitive_state}. "
            f"Expected one of: CONSTRUCTING, DEBUGGING, ASSESSING"
        )


@dataclass
class EnvironmentNode(BaseNode[StudentState, StudentDeps, None]):
    """
    Executes the code in the environment (if changed/runnable) and updates state.
    Also updates BKT based on assessment.
    """

    async def run(
        self, ctx: GraphRunContext[StudentState, StudentDeps]
    ) -> MarkovNode:

        # Check cognitive state - if CONSTRUCTING, do NOT execute
        if ctx.state.current_cognitive_state == "CONSTRUCTING":
            # Update state to reflect non-execution
            ctx.state.last_output = "(Code drafted but not executed)"

            if ctx.state.history:
                ctx.state.history[-1]['output'] = ctx.state.last_output
                ctx.state.history[-1]['error'] = ""
                ctx.state.history[-1][
                    'success'] = None  # Explicitly None for no execution

            return MarkovNode()

        # Check if ExecutorNode already ran the code (for ASSESSING/DEBUGGING)
        # This avoids double execution when ExecutorNode needs fresh output before LLM call
        if ctx.state._cached_execution_result is not None:
            result = ctx.state._cached_execution_result
            ctx.state._cached_execution_result = None  # Clear cache
        elif ctx.state.current_code and ctx.deps.ide_oracle and ctx.state.problem_id:
            # Run the code (normal path for DEBUGGING and other states)
            result = ctx.deps.ide_oracle.test_code(
                code=ctx.state.current_code, problem_id=ctx.state.problem_id
            )
            
            # V34: Store raw output - output_parser handles formatting in context formatters
            raw_output = result.stdout or result.captured_output

            # Update state with execution results
            ctx.state.last_output = raw_output
            ctx.state.execution_success = result.passed

            # Update test progress tracking
            ctx.state.tests_passed = result.passed_tests
            ctx.state.tests_total = result.total_tests
        else:
            result = None

        # --- V22: EPISODIC MEMORY EXTRACTION (Anti-Amnesia) ---
        # When an error gets fixed, extract what was learned for future recall
        if result is not None and ctx.state.current_cognitive_state == "DEBUGGING":
            self._extract_episodic_memory(ctx, result)

        # --- BKT UPDATE using sampling (only during Reflecting or Monitoring) ---
        # Pedagogical rationale: Learning happens during reflection/monitoring,
        # not during every code execution. This models that students update
        # their understanding when they consciously evaluate outcomes.
        #
        # NEW: Use BKT sampling instead of assessment oracle
        # 1. Sample from P(correct) = P(L)*(1-P(S)) + (1-P(L))*P(G) for each KC
        # 2. Update BKT based on sampled observation
        # 3. If tests passed AND sampled CORRECT, mark KC as demonstrated (no more slipping)
        if result is not None:
            # --- Performance vs Competence Check ---
            # If student used tutor help for this step, they may have "performed"
            # (copied correct answer) without "learning" (internalizing the concept).
            # When skip_bkt_on_assisted is True, we skip BKT updates for assisted steps.
            was_assisted = bool(ctx.state.tutor_response)
            bkt_skipped_reason = None

            if was_assisted and ctx.deps.skip_bkt_on_assisted:
                bkt_skipped_reason = "assisted_performance"
                logger.info(
                    f"BKT Update SKIPPED (Step {ctx.state.step_count}): "
                    "Student used tutor help. Performance != Competence mode active."
                )

            should_update_bkt = ctx.state.current_metacognitive_state in (
                "Reflecting", "Monitoring"
            )

            # Skip BKT update if assisted AND flag is enabled
            if bkt_skipped_reason:
                should_update_bkt = False

            if should_update_bkt and ctx.deps.bkt:
                required_kcs = ctx.state.required_kcs

                if required_kcs:
                    # Sample correct/incorrect for each KC using BKT model
                    sampled_results = ctx.deps.bkt.sample_and_feedback(
                        required_kcs
                    )

                    # Update BKT for each KC based on sampled observation
                    for kc_id, sample_info in sampled_results.items():
                        observation = sample_info['observation']
                        ctx.deps.bkt.update(kc_id, observation)

                        # If tests passed AND sampled CORRECT, mark KC as demonstrated
                        # This locks the KC - student can no longer "slip" on it
                        if result.passed and observation == Observation.CORRECT:
                            ctx.deps.bkt.mark_demonstrated(kc_id)

                    # Add BKT sampling info to history
                    if ctx.state.history:
                        ctx.state.history[-1]['bkt_update'] = {
                            kc_id: sample_info['observation'].value
                            for kc_id, sample_info in sampled_results.items()
                        }
                        ctx.state.history[-1]['bkt_feedback'] = {
                            kc_id: sample_info['feedback']
                            for kc_id, sample_info in sampled_results.items()
                        }

            # Add execution info to the last history item
            if ctx.state.history:
                ctx.state.history[-1]['output'] = ctx.state.last_output
                ctx.state.history[-1][
                    'error'] = result.stderr if result.stderr else ""
                ctx.state.history[-1]['success'] = ctx.state.execution_success

                # Record if BKT update was skipped due to assistance
                # This provides explicit trace evidence for case study analysis
                if bkt_skipped_reason:
                    ctx.state.history[-1]['bkt_interaction'] = {
                        'updated': False,
                        'reason': bkt_skipped_reason,
                        'was_assisted': True,
                        'note': 'Success observed, but mastery not credited due to assistance.'
                    }

                # Add BKT mastery state (P(L) for each KC) for knowledge tracking analysis
                if ctx.deps.bkt and ctx.deps.bkt.kc_states:
                    ctx.state.history[-1]['bkt_mastery'] = {
                        kc_id: {
                            'p_known': round(state.p_known, 4),
                            'mastery_level': state.mastery_level.value,
                            'efi_active': state.efi_active,
                            'observations': state.observation_count
                        }
                        for kc_id, state in ctx.deps.bkt.kc_states.items()
                    }

                # Add episodic memories snapshot (what the student has learned)
                if ctx.state.episodic_memories:
                    ctx.state.history[-1]['episodic_memories'] = [
                        {
                            'error_pattern': mem.error_pattern,
                            'realization': mem.realization,
                            'step_learned': mem.step_learned
                        }
                        for mem in ctx.state.episodic_memories
                    ]

                # Add test progress
                ctx.state.history[-1]['tests_passed'] = ctx.state.tests_passed
                ctx.state.history[-1]['tests_total'] = ctx.state.tests_total

        # Clear tutor response after it has been used
        # (The turn after assistance has now completed)
        ctx.state.tutor_response = ""

        return MarkovNode()

    def _extract_episodic_memory(
        self, ctx: GraphRunContext[StudentState, StudentDeps], result
    ) -> None:
        """
        V22: Extract episodic memory when an error gets fixed.

        Heuristic: Look for common error patterns that were present in previous
        output but are now resolved. Extract the pattern and what the student
        learned (from their monologue/goal).

        Common memorable error patterns:
        - "^" vs "**" for exponentiation (TypeError with ^)
        - Missing self parameter
        - Indentation errors
        - NameError (undefined variable)
        - SyntaxError patterns
        """
        # Skip if no history to compare against
        if len(ctx.state.history) < 2:
            return

        # Get previous output (what error was there before?)
        prev_step = ctx.state.history[-2] if len(ctx.state.history) >= 2 else None
        if not prev_step:
            return

        prev_output = prev_step.get('output', '') or ''
        curr_output = ctx.state.last_output or ''

        # Define error patterns to track (pattern, human-readable realization)
        ERROR_PATTERNS = [
            # Exponentiation error (^ vs **)
            ("unsupported operand type(s) for ^", "Python uses ** for exponentiation, not ^"),
            ("TypeError: unsupported operand type(s) for ^", "Python uses ** for exponentiation, not ^"),
            # Missing self
            ("takes 0 positional arguments but", "Methods need 'self' as first parameter"),
            ("missing 1 required positional argument: 'self'", "Need to call with self"),
            # Indentation
            ("IndentationError", "Python requires proper indentation"),
            ("expected an indented block", "Need to indent the code block"),
            # NameError
            ("NameError: name", "Variable needs to be defined before use"),
            # SyntaxError patterns
            ("SyntaxError: invalid syntax", "Check for typos or missing colons"),
            ("def def", "Don't write 'def' twice"),
            # Attribute errors
            ("AttributeError: 'NoneType'", "Function needs to return a value, not None"),
            ("has no attribute", "Method or attribute doesn't exist on this object"),
        ]

        # Check each pattern
        for error_pattern, realization in ERROR_PATTERNS:
            pattern_lower = error_pattern.lower()

            # Was this error present before but is now gone (or different)?
            was_present = pattern_lower in prev_output.lower()
            still_present = pattern_lower in curr_output.lower()

            if was_present and not still_present:
                # Error was fixed! Extract memory
                # Avoid duplicate memories for same pattern
                already_learned = any(
                    m.error_pattern.lower() == error_pattern.lower()
                    for m in ctx.state.episodic_memories
                )

                if not already_learned:
                    # Extract fix from current monologue/goal if available
                    fix_applied = ""
                    if ctx.state.current_monologue:
                        # Take first sentence as the fix description
                        fix_applied = ctx.state.current_monologue.split('.')[0][:100]

                    memory = EpisodicMemory(
                        error_pattern=error_pattern,
                        realization=realization,
                        step_learned=ctx.state.step_count,
                        fix_applied=fix_applied
                    )
                    ctx.state.episodic_memories.append(memory)
                    logger.debug(f"V22: Extracted episodic memory - {realization}")
