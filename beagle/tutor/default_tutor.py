"""
Default LLM Tutor

The proven default tutor using simple prompt files.
This is the tutor that has been used in all BEAGLE simulations.

Characteristics:
- Simple, encouraging language for middle school students
- Focuses on one concept at a time
- Gives hints without giving away solutions
- Uses BKT knowledge state to inform hints
"""

import logging
from pathlib import Path
from typing import Optional

from beagle.tutor.base import (
    TutorStrategy,
    TutorContext,
    TutorResponse,
    ScaffoldLevel,
)
from beagle.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Path to prompt files
PROMPT_DIR = Path(__file__).parent.parent / "data_generation" / "studentv2" / "prompts"


class DefaultLLMTutor(TutorStrategy):
    """
    Default LLM-based tutor using simple prompt files.
    
    This is the proven tutor implementation that has been used
    in all BEAGLE simulations. It uses:
    - tutor_system.txt: System prompt defining tutor persona
    - tutor_user.txt: User prompt template with context
    
    The prompts are designed for middle school students learning Python,
    with encouraging language and hint-based guidance.
    """
    
    SYSTEM_PROMPT = """You are a helpful and patient tutor for a middle school student learning Python.

Your role is to provide hints and guidance without giving away the complete solution. You should:
1. Be encouraging and supportive
2. Give hints that guide the student toward the solution
3. Use simple language appropriate for a beginner
4. Focus on one concept at a time
5. Avoid giving the complete code - let the student figure it out
6. Keep responses CONCISE (3-4 sentences) - be helpful but direct

Remember: The student is learning. Help them understand, don't just solve the problem for them."""

    USER_PROMPT_TEMPLATE = """A middle school student is learning Python and has asked you for help.

Problem they are working on:
{problem_description}

Student's Question:
"{student_question}"

Student's Current Code:
```python
{current_code}
```

Last Output:
{last_output}

Student's Knowledge State:
{knowledge_state}

Based on the student's question and their current code, provide a helpful hint.

Guidelines:
- Be encouraging but brief ("Good thinking!" + one hint)
- Give ONE specific hint that addresses their question
- Don't give away the full solution
- Keep your response to 3-4 sentences
- Use simple language appropriate for a middle schooler

Provide your hint/guidance as a brief, natural tutor response."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        """
        Initialize default LLM tutor.
        
        Args:
            llm_client: LLMClient instance for LLM calls.
                       If None, will need to be set before use.
        """
        self.llm_client = llm_client

    @property
    def tutor_type(self) -> str:
        return "default"

    async def generate_hint(self, context: TutorContext) -> TutorResponse:
        """
        Generate hint using the default prompts.
        
        This replicates the original _run_default_llm_tutor behavior
        from nodes.py but as a proper TutorStrategy.
        """
        if self.llm_client is None:
            return TutorResponse(
                hint="Keep trying! Think about what the code should do step by step.",
                scaffold_level=ScaffoldLevel.GUIDING,
                target_kc=None,
                tutor_type=self.tutor_type,
                reasoning="No LLM client available, used fallback hint",
            )

        # Get knowledge state description
        knowledge_state = "No knowledge tracking available."
        if context.bkt:
            knowledge_state = context.bkt.get_knowledge_description(
                include_mastered=True
            )

        # Build prompt
        prompt = self.USER_PROMPT_TEMPLATE.format(
            problem_description=context.problem_description,
            student_question=context.student_question or "Can you help me?",
            current_code=context.current_code or "# No code yet",
            last_output=context.last_output or context.last_error or "No output yet",
            knowledge_state=knowledge_state
        )

        try:
            # Call LLM asynchronously
            result = await self.llm_client.generate_async(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT,
                temperature=0.7
            )
            
            response_text = result.content
            logger.debug(f"Tutor raw response: {response_text}")
            
            if response_text:
                hint = response_text.strip()
                reasoning = "Generated using default prompt templates"
            else:
                hint = "That's a good question! Let's think about what your code should do."
                reasoning = "Empty LLM response, used fallback"
                
        except Exception as e:
            logger.error(f"Error generating hint: {e}")
            hint = "Keep trying! Think about what the code should do step by step."
            reasoning = f"LLM error: {str(e)}"

        # Determine scaffold level based on context
        target_kc = context.get_lowest_mastery_kc()
        scaffold_level = self.determine_scaffold_level(context)

        return TutorResponse(
            hint=hint,
            scaffold_level=scaffold_level,
            target_kc=target_kc,
            tutor_type=self.tutor_type,
            reasoning=reasoning,
            no_hint_needed=False,
        )

    def set_llm_client(self, llm_client: LLMClient) -> None:
        """Set the LLM client after initialization."""
        self.llm_client = llm_client
