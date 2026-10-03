"""
ZPD Tutor (Zone of Proximal Development)

Expert pedagogical hint generation using ZPD principles:
- Socratic questioning to promote self-discovery
- Progressive scaffolding (minimal → strategic → corrective → specific)
- Error-specific guidance when needed
- Conceptual over procedural guidance

Based on the original TutorAgent implementation which proved most effective.
"""

import logging
from typing import Optional

from beagle.tutor.base import (
    TutorStrategy,
    TutorContext,
    TutorResponse,
    ScaffoldLevel,
)
from beagle.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


class ZPDTutor(TutorStrategy):
    """
    ZPD-based tutor using expert LLM hint generation.
    
    Follows ZPD principles to provide appropriate scaffolding:
    - Minimal intervention (Socratic questions first)
    - Progressive disclosure (reveal more only if needed)
    - Error-specific guidance (address actual mistakes)
    - Conceptual over procedural (guide thinking, not just steps)
    
    This implementation is based on the original TutorAgent which
    proved most effective in experiments.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        """
        Initialize ZPD tutor.
        
        Args:
            llm_client: LLMClient instance for LLM calls.
                       If None, will need to be set before use.
        """
        self.llm_client = llm_client

    @property
    def tutor_type(self) -> str:
        return "zpd"

    async def generate_hint(self, context: TutorContext) -> TutorResponse:
        """
        Generate appropriate pedagogical hint using ZPD principles.
        
        Process:
        1. Determine hint level based on context (error, BKT mastery)
        2. Generate hint using appropriate method
        3. Return structured response
        """
        if self.llm_client is None:
            return TutorResponse(
                hint="Review your code and think about what each part should do.",
                scaffold_level=ScaffoldLevel.GUIDING,
                target_kc=None,
                tutor_type=self.tutor_type,
                reasoning="No LLM client available, used fallback hint",
            )

        # Determine hint level and target KC
        target_kc = context.get_lowest_mastery_kc()
        scaffold_level = self.determine_scaffold_level(context)
        
        # Route to appropriate hint generation method
        if context.last_error:
            hint_text, reasoning = await self._generate_corrective_hint(context)
            hint_type = "corrective"
        elif scaffold_level in [ScaffoldLevel.NONE, ScaffoldLevel.MINIMAL]:
            hint_text, reasoning = await self._generate_socratic_hint(context)
            hint_type = "socratic"
        elif scaffold_level == ScaffoldLevel.GUIDING:
            hint_text, reasoning = await self._generate_strategic_hint(context)
            hint_type = "strategic"
        else:
            hint_text, reasoning = await self._generate_specific_hint(context)
            hint_type = "specific"

        return TutorResponse(
            hint=hint_text,
            scaffold_level=scaffold_level,
            target_kc=target_kc,
            tutor_type=self.tutor_type,
            reasoning=f"{hint_type}: {reasoning}",
            no_hint_needed=(scaffold_level == ScaffoldLevel.NONE),
        )

    async def _generate_socratic_hint(self, context: TutorContext) -> tuple[str, str]:
        """Generate Socratic question to guide thinking."""
        
        # Build test context
        test_context = ""
        if context.last_error:
            test_context = f"\n**Output/Error:**\n{context.last_error}"
        
        # Student question
        student_q = ""
        if context.student_question:
            student_q = f"\n**Student's Question:**\n{context.student_question}\n\n⚠️ Address their question directly while guiding them to discover the answer."
        
        # Knowledge state
        knowledge_state = self._format_knowledge_state(context)

        prompt = f"""You are an expert programming tutor using Socratic questioning.

**Problem:**
{context.problem_description}

**Student's Current Code:**
```python
{context.current_code or "# No code yet"}
```
{test_context}{student_q}

{knowledge_state}

**Your task:**
Generate a Socratic question that helps the student discover the issue themselves.

**Priority:**
1. If student asked a question → Address it with a counter-question
2. If function signature is wrong → Ask about required signature
3. If missing critical steps → Ask what step comes next
4. If logic error → Ask them to trace their logic
5. Minor style issues → Ignore, focus on functionality

The question should:
- Guide their thinking process
- Help them discover the issue themselves
- Be specific to their current code
- NOT provide the solution directly

Respond with ONLY the question, no explanation."""

        try:
            result = await self.llm_client.generate_async(prompt)
            if result.content:
                return result.content.strip(), "Socratic questioning"
        except Exception as e:
            logger.error(f"Error generating Socratic hint: {e}")
        
        return "What is the first step you need to take to solve this problem?", "Fallback"

    async def _generate_strategic_hint(self, context: TutorContext) -> tuple[str, str]:
        """Generate strategic hint about approach."""
        
        knowledge_state = self._format_knowledge_state(context)
        
        prompt = f"""You are an expert programming tutor providing strategic guidance.

**Problem:**
{context.problem_description}

**Student's Current Code:**
```python
{context.current_code or "# No code yet"}
```

{knowledge_state}

**Your task:**
Provide a HIGH-LEVEL strategic hint about what the student should focus on next.

**Priority:**
1. Function signature issues (wrong name/parameters) are CRITICAL
2. Missing major steps
3. Conceptual misunderstandings
4. Implementation approach

Guidelines:
- Do NOT write code for them
- Guide their strategic thinking
- Keep it to 1-2 sentences
- Focus on CRITICAL issues first

Respond with ONLY the hint, no explanation."""

        try:
            result = await self.llm_client.generate_async(prompt)
            if result.content:
                return result.content.strip(), "Strategic guidance"
        except Exception as e:
            logger.error(f"Error generating strategic hint: {e}")
        
        return "Break down the problem into smaller steps.", "Fallback"

    async def _generate_corrective_hint(self, context: TutorContext) -> tuple[str, str]:
        """Generate corrective hint for error."""
        
        prompt = f"""You are an expert programming tutor helping debug code.

**Problem:**
{context.problem_description}

**Student's Code:**
```python
{context.current_code}
```

**Error/Output:**
{context.last_error or context.last_output}

**Your task:**
Provide corrective feedback that addresses the MOST CRITICAL issue.

**Priority:**
1. Function signature mismatch (wrong name/parameters) → #1 issue
2. Missing required imports
3. Logic errors (wrong formula, missing conversions)
4. Syntax errors
5. Style issues → IGNORE if functionality works

Your hint should:
1. Identify the HIGHEST priority issue
2. Explain WHAT the error means
3. Give a HINT about where to look
4. NOT fix the code directly
5. Be specific and actionable (2-3 sentences)

Respond with ONLY the hint, no explanation."""

        try:
            result = await self.llm_client.generate_async(prompt)
            if result.content:
                return result.content.strip(), "Error-specific guidance"
        except Exception as e:
            logger.error(f"Error generating corrective hint: {e}")
        
        return "Look carefully at the error message and think about what it's telling you.", "Fallback"

    async def _generate_specific_hint(self, context: TutorContext) -> tuple[str, str]:
        """Generate specific hint with more detail."""
        
        knowledge_state = self._format_knowledge_state(context)
        
        prompt = f"""You are an expert programming tutor providing detailed guidance.

**Problem:**
{context.problem_description}

**Student's Current Code:**
```python
{context.current_code or "# No code yet"}
```

{knowledge_state}

**Your task:**
Provide a SPECIFIC hint that:
1. Identifies what's missing or wrong
2. Suggests a concrete next step
3. Still requires the student to implement it themselves
4. Keeps to 2-3 sentences

Respond with ONLY the hint, no explanation."""

        try:
            result = await self.llm_client.generate_async(prompt)
            if result.content:
                return result.content.strip(), "Specific guidance"
        except Exception as e:
            logger.error(f"Error generating specific hint: {e}")
        
        return "Review the requirements and make sure your implementation covers all necessary parts.", "Fallback"

    def _format_knowledge_state(self, context: TutorContext) -> str:
        """Format knowledge state for prompts."""
        if context.bkt is None:
            return ""
        
        summary = context.get_knowledge_state_summary()
        if not summary.get("available", False):
            return ""
        
        kcs = summary.get("kcs", {})
        if not kcs:
            return ""
        
        lines = ["**Knowledge State:**"]
        for kc_id, info in kcs.items():
            kc_name = self.format_kc_name(kc_id)
            mastery = info["p_known"]
            level = info["mastery_level"]
            lines.append(f"  - {kc_name}: {mastery:.2f} ({level})")
        
        return "\n".join(lines)

    def set_llm_client(self, llm_client: LLMClient) -> None:
        """Set the LLM client after initialization."""
        self.llm_client = llm_client


# Backwards compatibility alias
LLMBasedTutor = ZPDTutor
