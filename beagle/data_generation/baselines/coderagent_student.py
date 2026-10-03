"""
CoderAgent Student Baseline

Implements the CoderAgent architecture from "CoderAgent: Simulating Programming 
Student Behavior for Personalized Learning" which uses:

1. ACT-R-inspired Memory Module
   - Long-term Memory: Persistent knowledge proficiency tracking
   - Short-term Memory: Recent observations, current task context

2. Programming Tree of Thought (PTOT)
   - WHY: Understand the goal and requirements
   - HOW: Determine approach/algorithm strategy
   - WHERE: Identify which code section needs changes
   - WHAT: Generate the actual code

Key difference from other baselines:
- Structured cognitive architecture inspired by ACT-R
- Multi-phase reasoning (not just chain-of-thought)
- Explicit knowledge proficiency modeling

Reference: arXiv (2024) CoderAgent paper
"""

import asyncio
import logging
import time
import re
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from beagle.data_generation.ide_oracle.ide_oracle import IDEOracle
from beagle.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)

# =============================================================================
# ACT-R Inspired Memory Module
# =============================================================================

@dataclass
class ShortTermMemory:
    """
    Short-term memory (working memory) - task-specific, resets frequently.
    
    In ACT-R, this corresponds to the goal buffer and imaginal buffer.
    Holds current task context and recent observations.
    """
    current_goal: str = ""
    current_subgoal: str = ""
    recent_errors: List[str] = field(default_factory=list)
    last_execution_output: str = ""
    current_approach: str = ""
    
    def update(self, goal: str = None, subgoal: str = None, 
               error: str = None, output: str = None, approach: str = None):
        if goal:
            self.current_goal = goal
        if subgoal:
            self.current_subgoal = subgoal
        if error:
            self.recent_errors.append(error)
            # Keep only last 3 errors
            self.recent_errors = self.recent_errors[-3:]
        if output:
            self.last_execution_output = output
        if approach:
            self.current_approach = approach
    
    def to_prompt(self) -> str:
        parts = []
        if self.current_goal:
            parts.append(f"Current Goal: {self.current_goal}")
        if self.current_subgoal:
            parts.append(f"Current Subgoal: {self.current_subgoal}")
        if self.recent_errors:
            parts.append(f"Recent Errors: {'; '.join(self.recent_errors[-2:])}")
        if self.current_approach:
            parts.append(f"Current Approach: {self.current_approach}")
        return "\n".join(parts) if parts else "(No active task context)"


@dataclass
class LongTermMemory:
    """
    Long-term memory - persistent knowledge proficiency.
    
    In ACT-R, this corresponds to declarative memory with activation levels.
    Models knowledge components and their mastery probability.
    """
    # Knowledge proficiency (0.0 to 1.0)
    proficiency: Dict[str, float] = field(default_factory=lambda: {
        'class_definition': 0.5,
        'method_definition': 0.5,
        'self_usage': 0.5,
        'power_operator': 0.3,  # Common struggle
        'physics_formulas': 0.3,
        'debugging': 0.4,
        'tuple_return': 0.5,
        'variable_naming': 0.6,
    })
    
    # Error history (for learning)
    error_counts: Dict[str, int] = field(default_factory=dict)
    success_counts: Dict[str, int] = field(default_factory=dict)
    
    def update_from_result(self, success: bool, error_type: str = None):
        """Update proficiency based on execution result (reinforcement learning)."""
        learning_rate = 0.1
        
        if success:
            # Boost all proficiencies slightly
            for key in self.proficiency:
                self.proficiency[key] = min(1.0, self.proficiency[key] + learning_rate * 0.5)
        
        if error_type:
            # Map error to knowledge component
            error_mapping = {
                'TypeError': ['power_operator', 'self_usage'],
                'NameError': ['variable_naming', 'self_usage'],
                'AttributeError': ['self_usage', 'method_definition'],
                'SyntaxError': ['class_definition', 'method_definition'],
                'AssertionError': ['physics_formulas', 'debugging'],
            }
            
            affected_kcs = error_mapping.get(error_type, ['debugging'])
            for kc in affected_kcs:
                if kc in self.proficiency:
                    # Decrease proficiency on error, but learn from it
                    self.proficiency[kc] = max(0.1, self.proficiency[kc] - learning_rate * 0.3)
                    self.error_counts[kc] = self.error_counts.get(kc, 0) + 1
    
    def get_weakness(self) -> str:
        """Return the knowledge component with lowest proficiency."""
        return min(self.proficiency, key=self.proficiency.get)
    
    def to_prompt(self) -> str:
        weak_areas = [k for k, v in self.proficiency.items() if v < 0.5]
        strong_areas = [k for k, v in self.proficiency.items() if v >= 0.7]
        
        parts = []
        if weak_areas:
            parts.append(f"Struggling with: {', '.join(weak_areas[:3])}")
        if strong_areas:
            parts.append(f"Comfortable with: {', '.join(strong_areas[:2])}")
        return "\n".join(parts) if parts else "(Novice level across all areas)"


# =============================================================================
# PTOT Output Models
# =============================================================================

class PTOTStepOutput(BaseModel):
    """
    Programming Tree of Thought (PTOT) output structure.

    Breaks down reasoning into: WHY -> HOW -> WHERE -> WHAT
    """
    # WHY: Goal understanding
    why_goal: str = Field(
        ...,
        max_length=500,
        description="WHY: What am I trying to achieve? ONE short sentence."
    )
    why_knowledge: str = Field(
        ...,
        max_length=500,
        description="WHY: What knowledge/concepts do I need to apply? ONE short sentence."
    )

    # HOW: Strategy planning
    how_approach: str = Field(
        ...,
        max_length=500,
        description="HOW: What's my strategy or approach? ONE short sentence."
    )
    how_steps: str = Field(
        ...,
        max_length=500,
        description="HOW: What steps do I need to take? ONE short sentence."
    )

    # WHERE: Location identification
    where_focus: str = Field(
        ...,
        max_length=500,
        description="WHERE: Which part of the code needs attention? ONE short sentence."
    )

    # WHAT: Action execution
    what_action: str = Field(
        ...,
        max_length=50,
        description="WHAT: The cognitive action - CONSTRUCTING, DEBUGGING, or ASSESSING"
    )
    what_thinking: str = Field(
        ...,
        max_length=800,
        description="WHAT: My messy internal monologue (2-3 sentences, ~50 words MAX, with 'idk', 'ugh', 'maybe')."
    )
    what_code: str = Field(
        ...,
        max_length=8000,
        description="WHAT: The complete updated Python code, or 'NO_CHANGE' if not modifying"
    )


# =============================================================================
# CoderAgent Prompts
# =============================================================================

SYSTEM_PROMPT = """You are simulating a novice Python student (12-13 years old) using the CoderAgent cognitive architecture.

PERFORMANCE LEVEL: {performance_level}

{persona_description}

MEMORY CONTEXT:
{memory_context}

You must think through problems using the Programming Tree of Thought (PTOT):

1. **WHY Phase**: Understand the goal
   - What am I trying to achieve right now?
   - What knowledge/concepts do I need?

2. **HOW Phase**: Plan your approach
   - What strategy should I use?
   - What steps do I need?

3. **WHERE Phase**: Locate the focus
   - Which part of code needs work?

4. **WHAT Phase**: Execute
   - Take action (CONSTRUCTING/DEBUGGING/ASSESSING)
   - Write code reflecting your novice understanding

IMPORTANT: You are a REAL STUDENT who makes real mistakes. Your code should reflect 
your current knowledge proficiency, not expert-level programming.

{extra_instructions}
"""

USER_PROMPT = """PROBLEM: {problem_description}

CURRENT CODE:
```python
{current_code}
```

LAST OUTPUT:
{last_output}

STEP {step_number} of {max_steps}:

SHORT-TERM MEMORY (Current Task):
{short_term_memory}

LONG-TERM MEMORY (Knowledge State):
{long_term_memory}

Use PTOT (Programming Tree of Thought) to reason through this step:
- WHY: Goal and knowledge requirements
- HOW: Strategy and steps
- WHERE: Code location to focus on
- WHAT: Action and code

Provide your response.
"""


# =============================================================================
# CoderAgent Student Class
# =============================================================================

@dataclass
class CoderAgentStepResult:
    """Result of a single CoderAgent simulation step."""
    step: int
    action: str
    ptot_why: str
    ptot_how: str
    ptot_where: str
    thinking: str
    code: str
    output: str
    tests_passed: int
    tests_total: int
    success: bool
    error: str = ""


@dataclass
class CoderAgentSimulationResult:
    """Result of a complete CoderAgent simulation."""
    run_id: int
    performance_level: str
    problem_id: str
    solved: bool
    total_steps: int
    action_counts: Dict[str, int]
    final_tests_passed: int
    final_tests_total: int
    duration_seconds: float
    timestamp: str
    history: List[Dict[str, Any]] = field(default_factory=list)
    baseline_type: str = "coderagent"

    def to_dict(self) -> dict:
        return asdict(self)


class CoderAgentStudent:
    """
    CoderAgent student baseline with ACT-R memory and PTOT reasoning.
    """

    def __init__(
        self,
        performance_level: Literal["low", "high"] = "low",
        llm_model: str = "google-gla:gemini-2.0-flash",
    ):
        self.performance_level = performance_level
        self.llm_client = LLMClient(model=llm_model)
        
        # Initialize memory modules
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory()
        
        # Adjust initial proficiency based on performance level
        if performance_level == "low":
            for key in self.long_term.proficiency:
                self.long_term.proficiency[key] *= 0.6
            self.persona = """As a LOW performer, you:
- Get confused easily and make many conceptual errors
- Often try random changes without understanding why
- Miss obvious errors and make the same mistakes repeatedly
- Your mindset: "I don't really get this..." or "Maybe if I change this..."
"""
            self.extra_instructions = "Make REAL mistakes reflecting your low proficiency areas."
        else:
            for key in self.long_term.proficiency:
                self.long_term.proficiency[key] = min(1.0, self.long_term.proficiency[key] * 1.3)
            self.persona = """As a HIGH performer, you:
- Learn from mistakes but still make novice errors
- Try to understand error messages but sometimes misinterpret
- More systematic but still inexperienced
- Your mindset: "Let me think about this..." or "I think I see the problem..."
"""
            self.extra_instructions = "You're capable but still make beginner mistakes."

    async def _run_ptot_step(
        self,
        problem_description: str,
        current_code: str,
        last_output: str,
        step_number: int,
        max_steps: int,
    ) -> PTOTStepOutput:
        """Run a single PTOT-guided simulation step."""
        
        # Build memory context
        memory_context = f"""Long-term Knowledge:
{self.long_term.to_prompt()}

Short-term Context:
{self.short_term.to_prompt()}
"""
        
        system_prompt = SYSTEM_PROMPT.format(
            performance_level=self.performance_level.upper(),
            persona_description=self.persona,
            memory_context=memory_context,
            extra_instructions=self.extra_instructions
        )
        
        user_prompt = USER_PROMPT.format(
            problem_description=problem_description,
            current_code=current_code if current_code else "# No code yet",
            last_output=last_output if last_output else "(No output yet)",
            step_number=step_number,
            max_steps=max_steps,
            short_term_memory=self.short_term.to_prompt(),
            long_term_memory=self.long_term.to_prompt(),
        )
        
        agent = Agent(
            self.llm_client.pydantic_model,
            output_type=PTOTStepOutput,
            system_prompt=system_prompt,
            model_settings={"timeout": 60.0},
        )
        
        result = await agent.run(user_prompt)
        return result.output

    def _extract_error_type(self, output: str) -> Optional[str]:
        """Extract error type from execution output."""
        error_patterns = ['TypeError', 'NameError', 'AttributeError', 
                         'SyntaxError', 'AssertionError', 'ValueError', 'IndexError']
        for pattern in error_patterns:
            if pattern in output:
                return pattern
        return None

    def solve_problem(
        self,
        problem_description: str,
        problem_id: str,
        max_steps: int = 30,
    ) -> List[Dict[str, Any]]:
        """Run the CoderAgent simulation."""
        oracle = IDEOracle(save_history=True)
        
        current_code = ""
        last_output = "(No output yet)"
        history = []
        solved = False
        
        # Initialize short-term memory with problem goal
        self.short_term.update(goal="Complete the Particle class implementation")
        
        for step in range(1, max_steps + 1):
            try:
                output = asyncio.get_event_loop().run_until_complete(
                    self._run_ptot_step(
                        problem_description=problem_description,
                        current_code=current_code,
                        last_output=last_output,
                        step_number=step,
                        max_steps=max_steps,
                    )
                )
                
                action = output.what_action.upper()
                if action not in ["CONSTRUCTING", "DEBUGGING", "ASSESSING"]:
                    action = "CONSTRUCTING"
                
                new_code = output.what_code
                if new_code and new_code != "NO_CHANGE":
                    # Clean up code
                    new_code = re.sub(r'^```python\s*', '', new_code.strip())
                    new_code = re.sub(r'\s*```$', '', new_code.strip())
                    current_code = new_code
                
                # Update short-term memory with current approach
                self.short_term.update(
                    subgoal=output.where_focus,
                    approach=output.how_approach
                )
                
                # Execute code for DEBUGGING/ASSESSING
                tests_passed = 0
                tests_total = 0
                success = False
                
                if action in ["DEBUGGING", "ASSESSING"] and current_code:
                    exec_result = oracle.test_code(
                        code=current_code, problem_id=problem_id
                    )
                    last_output = exec_result.stdout or exec_result.captured_output or "(No output)"
                    tests_passed = exec_result.passed_tests
                    tests_total = exec_result.total_tests
                    success = exec_result.passed
                    
                    # Update memories based on result
                    error_type = self._extract_error_type(last_output)
                    self.long_term.update_from_result(success, error_type)
                    
                    if error_type:
                        self.short_term.update(error=f"{error_type}: {last_output[:100]}")
                        
                elif action == "CONSTRUCTING":
                    last_output = "(Code drafted but not executed)"
                
                # Build step dict (PTOT info collapsed into monologue)
                ptot_reasoning = f"[WHY] {output.why_goal} [HOW] {output.how_approach} [WHERE] {output.where_focus}"
                
                step_dict = {
                    'step': step,
                    'cognitive_state': action,
                    'action': action,
                    'monologue': output.what_thinking,
                    'ptot_reasoning': ptot_reasoning,
                    'code': current_code,
                    'output': last_output,
                    'tests_passed': tests_passed,
                    'tests_total': tests_total,
                    'success': success,
                    'knowledge_proficiency': dict(self.long_term.proficiency),
                }
                
                history.append(step_dict)
                
                logger.info(
                    f"  Step {step}: {action} - {'SOLVED!' if success else f'{tests_passed}/{tests_total} tests'}"
                )
                
                if success:
                    solved = True
                    break
                    
            except Exception as e:
                logger.error(f"Step {step} failed: {e}")
                history.append({
                    "step": step,
                    "cognitive_state": "ERROR",
                    "action": "ERROR",
                    "monologue": f"[CoderAgent Error: {e}]",
                    "error": str(e),
                    "code": current_code,
                    "output": last_output,
                    "success": False,
                })
                continue
        
        return history


# =============================================================================
# Simulation Runner
# =============================================================================

def run_coderagent_simulation(
    run_id: int,
    performance_level: str,
    problem_id: str,
    max_steps: int,
    llm_model: str = "google-gla:gemini-2.0-flash",
) -> CoderAgentSimulationResult:
    """Run a single CoderAgent simulation."""
    logger.info(f"Starting CODERAGENT run {run_id} ({performance_level} performer)")
    
    start_time = time.time()
    
    # Load problem
    problem_def = IDEOracle.load_problem(problem_id)
    
    student = CoderAgentStudent(
        performance_level=performance_level,
        llm_model=llm_model,
    )
    
    history = student.solve_problem(
        problem_description=problem_def.description,
        problem_id=problem_id,
        max_steps=max_steps,
    )
    
    duration = time.time() - start_time
    
    # Compute stats
    solved = any(h.get("success", False) for h in history)
    action_counts = {"CONSTRUCTING": 0, "DEBUGGING": 0, "ASSESSING": 0}
    for h in history:
        action = h.get("action", h.get("cognitive_state", "UNKNOWN"))
        if action in action_counts:
            action_counts[action] += 1
    
    final_step = history[-1] if history else {}
    
    result = CoderAgentSimulationResult(
        run_id=run_id,
        performance_level=performance_level,
        problem_id=problem_id,
        solved=solved,
        total_steps=len(history),
        action_counts=action_counts,
        final_tests_passed=final_step.get("tests_passed", 0),
        final_tests_total=final_step.get("tests_total", 0),
        duration_seconds=duration,
        timestamp=datetime.now().isoformat(),
        history=history,
        baseline_type="coderagent",
    )
    
    logger.info(
        f"  Run {run_id} complete: {'SOLVED' if solved else 'NOT SOLVED'} in {result.total_steps} steps ({duration:.1f}s)"
    )
    
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Running CoderAgent baseline test...")
    result = run_coderagent_simulation(
        run_id=1,
        performance_level="low",
        problem_id="particle_simulator",
        max_steps=15,
    )
    
    print(f"\nResult: {'SOLVED' if result.solved else 'NOT SOLVED'} in {result.total_steps} steps")
    print(f"Actions: {result.action_counts}")
    print(f"Duration: {result.duration_seconds:.1f}s")
    
    # Print sample step
    if result.history:
        print(f"\nSample step monologue:")
        print(f"  {result.history[0].get('monologue', 'N/A')[:200]}...")
