from typing import Optional, List

from pydantic import BaseModel, Field


# =============================================================================
# Field length caps
# =============================================================================
# Smaller models (e.g. gemini-2.5-flash-lite) lose stop-token discipline on
# open-ended free-text fields and decode-loop until they hit max output tokens,
# producing MALFORMED_FUNCTION_CALL. Capping the schema-level length forces
# pydantic-ai to reject overruns and retry with a shorter generation.
#
# Numbers calibrated against the paper's gemini-2.0-flash particle_simulator
# runs (1441 records). Caps are p99 + ~50% headroom, so we should never reject
# a legitimate output from the same prompt design.
#
#   Field           p50    p95    p99    max    cap
#   goal            71     136    169    242    500
#   mindset         10     142    200    240    500
#   directive       142    248    320    365    500
#   monologue       135    207    240    354    800
#   error_seen      65     83     99     152    500
#   memory_note     —      —      —      —      500
#   reflection      112    172    181    181    800
#   question        214    302    361    361    800
#   tutor_response  326    435    518    518    800
#   code            697    1290   1543   1688   8000
_SHORT_FIELD = 500    # goals / mindsets / directives / labels / error_seen / memory_note
_MED_FIELD = 800      # monologues / reflections / questions / responses
_CODE_FIELD = 8000    # full file dumps for the bouncing/inclined/particle problems


class StrategistOutput(BaseModel):
    """
    Output structure for the Strategist agent.
    """
    goal: str = Field(
        ...,
        max_length=_SHORT_FIELD,
        description="The immediate goal for the next step. ONE short sentence (~15 words max)."
    )
    mindset: str = Field(
        ...,
        max_length=_SHORT_FIELD,
        description="One label for the mindset to adopt (e.g., 'Careful', 'Exploratory'). ONE-TWO words."
    )
    directive: str = Field(
        ...,
        max_length=_SHORT_FIELD,
        description="A specific directive for the executor. ONE short sentence (~20 words max)."
    )


class ExecutorOutput(BaseModel):
    """
    Output structure for the Executor agent.
    """
    monologue: str = Field(
        ...,
        max_length=_MED_FIELD,
        description="Internal monologue (1-2 sentences, ~30 words MAX). Stop after the second sentence."
    )
    code: str = Field(
        ...,
        max_length=_CODE_FIELD,
        description="The code to write. Empty if just thinking or planning."
    )


class ConstructingOutput(BaseModel):
    """
    Output structure for CONSTRUCTING cognitive state.
    Produces code and internal monologue about the coding process.

    NOTE: The monologue should reflect the student's thinking WHILE writing code,
    NOT predictions about what errors might occur (which would be psychic debugging).
    """
    monologue: str = Field(
        ...,
        max_length=_MED_FIELD,
        description="Internal thoughts as you write code (1-2 sentences, ~30 words MAX). Stop after the second sentence. Don't predict errors."
    )
    code: str = Field(
        ...,
        max_length=_CODE_FIELD,
        description="The code to draft. Focus on one small incremental change."
    )


class AssessingOutput(BaseModel):
    """
    Output structure for ASSESSING cognitive state.
    Only produces reflection/monologue - NO code modifications allowed.
    The reflection carries forward to inform the next cognitive turn.
    """
    reflection: str = Field(
        ...,
        max_length=_MED_FIELD,
        description="Reflection on the code output (1-2 sentences, ~30 words MAX). What worked, what didn't. Stop after the second sentence."
    )


class DebuggingOutput(BaseModel):
    """
    Output structure for DEBUGGING cognitive state.
    Produces both code changes AND monologue - student sees error and fixes it.
    V34.5: Added memory_note for agent-driven memory system.
    """
    error_seen: str = Field(
        ...,
        max_length=_SHORT_FIELD,
        description="Copy the EXACT error type and message from the console (e.g., 'TypeError: __init__() takes 4 args'). ONE line, no commentary."
    )
    monologue: str = Field(
        ...,
        max_length=_MED_FIELD,
        description="Internal reaction to the error you just stated (1-2 sentences, ~30 words MAX). Stop after the second sentence."
    )
    memory_note: str = Field(
        ...,
        max_length=_SHORT_FIELD,
        description="Short note to your future self. Format: '[ErrorType] - short phrase'. ~10 words MAX."
    )
    code: str = Field(
        ...,
        max_length=_CODE_FIELD,
        description="The modified code with the fix applied."
    )


class AssistanceOutput(BaseModel):
    """
    Output structure for the Assistance agent (student asking for help).

    Same as StrategistOutput + question field. The question is generated last
    to leverage autoregressive LLM reasoning (better question quality when
    goal/mindset/directive are generated first).
    """
    goal: str = Field(
        ...,
        max_length=_SHORT_FIELD,
        description="What the student is trying to achieve. ONE short sentence (~15 words max)."
    )
    mindset: str = Field(
        ...,
        max_length=_SHORT_FIELD,
        description="The student's emotional/cognitive state (e.g., 'Confused', 'Frustrated'). ONE-TWO words."
    )
    directive: str = Field(
        ...,
        max_length=_SHORT_FIELD,
        description="What the student was trying to do before asking. ONE short sentence (~20 words max)."
    )
    question: str = Field(
        ...,
        max_length=_MED_FIELD,
        description="The specific question the student asks the tutor (1-2 sentences, ~30 words MAX)."
    )


class TutorOutput(BaseModel):
    """
    Output structure for the Tutor agent (responding to student questions).
    """
    response: str = Field(
        ...,
        max_length=_MED_FIELD,
        description="The tutor's hint or guidance (1-2 sentences, ~30 words MAX). Stop after the second sentence."
    )


# =============================================================================
# ABLATION: Merged Pipeline Output Schemas
# =============================================================================
# These schemas merge Strategist + Executor into a SINGLE LLM call.
# Used to test if the two-stage pipeline prevents psychic debugging.

class MergedConstructingOutput(BaseModel):
    """
    ABLATION: Merged output for CONSTRUCTING (single LLM call).

    Combines Strategist (goal/mindset/directive) + Executor (monologue/code).
    The thinking field is where psychic debugging might leak.
    """
    goal: str = Field(..., max_length=_SHORT_FIELD, description="The immediate goal. ONE short sentence (~15 words max).")
    mindset: str = Field(..., max_length=_SHORT_FIELD, description="The mindset (e.g., 'Careful', 'Exploratory'). ONE-TWO words.")
    thinking: str = Field(
        ...,
        max_length=_MED_FIELD,
        description="Internal monologue as you work (1-2 sentences, ~30 words MAX). Stop after the second sentence."
    )
    code: str = Field(..., max_length=_CODE_FIELD, description="The code to draft.")


class MergedDebuggingOutput(BaseModel):
    """
    ABLATION: Merged output for DEBUGGING (single LLM call).

    Combines Strategist + Executor for debugging.
    """
    goal: str = Field(..., max_length=_SHORT_FIELD, description="What you're trying to fix. ONE short sentence (~15 words max).")
    mindset: str = Field(..., max_length=_SHORT_FIELD, description="Your mindset (e.g., 'Frustrated', 'Analytical'). ONE-TWO words.")
    error_seen: str = Field(..., max_length=_SHORT_FIELD, description="Copy the EXACT error from the console. ONE line.")
    thinking: str = Field(..., max_length=_MED_FIELD, description="Reaction to the error (1-2 sentences, ~30 words MAX). Stop after the second sentence.")
    memory_note: str = Field(..., max_length=_SHORT_FIELD, description="Short note: '[ErrorType] - short phrase'. ~10 words MAX.")
    code: str = Field(..., max_length=_CODE_FIELD, description="The modified code with the fix applied.")


class MergedAssessingOutput(BaseModel):
    """
    ABLATION: Merged output for ASSESSING (single LLM call).

    Combines Strategist + Executor for assessment.
    """
    goal: str = Field(..., max_length=_SHORT_FIELD, description="What you were checking. ONE short sentence (~15 words max).")
    mindset: str = Field(..., max_length=_SHORT_FIELD, description="Your mindset after seeing results. ONE-TWO words.")
    thinking: str = Field(..., max_length=_MED_FIELD, description="Reflection on the output (1-2 sentences, ~30 words MAX).")

