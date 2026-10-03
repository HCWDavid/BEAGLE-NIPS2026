"""
Shared schemas and prompts for metacog-aware baselines.

Option C+ Implementation: Stateful Prompting
- LLM decides metacog state (Planning, Enacting, Monitoring, Reflecting)
- Previous metacog state passed back into prompt
- Step-by-step execution (no sequences)
- Let LLM reveal its competence bias through short durations
"""

from pydantic import BaseModel, Field
from typing import Literal


# =============================================================================
# Output Schemas
# =============================================================================

class MetacogAwareStepOutput(BaseModel):
    """Output for metacog-aware baseline step."""

    metacognitive_state: Literal["Planning", "Enacting", "Monitoring", "Reflecting"] = Field(
        ...,
        description="Your current Self-Regulated Learning phase. Planning=thinking about approach, Enacting=writing code, Monitoring=checking output, Reflecting=analyzing errors."
    )
    metacog_reasoning: str = Field(
        ...,
        max_length=500,
        description="Why are you in this phase? ONE sentence (e.g., 'I need to check what went wrong' -> Reflecting)."
    )
    cognitive_action: Literal["CONSTRUCTING", "DEBUGGING", "ASSESSING"] = Field(
        ...,
        description="CONSTRUCTING=write/edit code, DEBUGGING=run and fix errors, ASSESSING=run and check results"
    )
    thinking: str = Field(
        ...,
        max_length=800,
        description="Your internal monologue as a confused/learning student (2-3 sentences, ~50 words MAX)."
    )
    code: str = Field(
        ...,
        max_length=8000,
        description="The complete updated code. If just planning, output current code unchanged."
    )


# =============================================================================
# SRL Prompts
# =============================================================================

SRL_INSTRUCTION = """
=== SELF-REGULATED LEARNING (SRL) ===

You must explicitly track your learning phase. Real students follow these patterns:

1. **Planning**: Strategy formulation, thinking about approach BEFORE coding.
   - "What do I need to build?"
   - "How should I structure this?"
   - Often lasts 3-5 steps for beginners.

2. **Enacting**: Actively writing or modifying code.
   - "Let me try adding this..."
   - "I'll write the class now."
   - Students often stay here for many steps.

3. **Monitoring**: Running code and checking output.
   - "Let me see if this works..."
   - "What does the error say?"
   - Quick checks between coding bouts.

4. **Reflecting**: Analyzing errors, understanding what went wrong.
   - "Why did that fail?"
   - "I see, the problem is..."
   - Often leads back to Planning or Enacting.

IMPORTANT: Real students often STAY in the same phase for MULTIPLE steps. 
Do NOT switch phases every step - that's unrealistic.
Only transition when there's a clear reason (e.g., code is done -> run it).
"""


SRL_PREVIOUS_STATE_PROMPT = """
YOUR PREVIOUS PHASE: {previous_metacog}

Consider: Should you stay in {previous_metacog} or transition to a new phase?
- Stay if you're not done with the current task
- Transition only if there's a clear reason
"""
