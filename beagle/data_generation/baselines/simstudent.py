"""
SimStudent-Inspired Baseline

SimStudent is a rule-based cognitive tutoring system that uses inductive logic
programming to learn production rules. This baseline mimics SimStudent's approach
by using predetermined production rules rather than LLM generation.

Key characteristics:
- NO LLM calls (purely rule-based)
- Deterministic behavior based on knowledge state
- Models knowledge acquisition through observed errors
- High behavioral fidelity (structured) but low perceptual fidelity (no natural language)

Reference: Matsuda et al. (2007) "Learning by Doing vs. Learning by Being Told"
https://www.cs.cmu.edu/~mazda/SimStudent/

Used for comparison: "How does a classical cognitive model compare to hybrid approaches?"
"""

import logging
import random
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal

from beagle.data_generation.ide_oracle.ide_oracle import IDEOracle

logger = logging.getLogger(__name__)

# =============================================================================
# SimStudent Production Rules (Knowledge Components)
# =============================================================================

# Each rule represents a piece of knowledge the student may or may not have
PRODUCTION_RULES = {
    'class_definition': {
        'pattern': r'class\s+\w+:',
        'description': 'Can define a class',
        'prerequisite': None,
        'difficulty': 0.1,
    },
    'init_method': {
        'pattern': r'def\s+__init__\s*\(',
        'description': 'Can write __init__ method',
        'prerequisite': 'class_definition',
        'difficulty': 0.2,
    },
    'self_attribute': {
        'pattern': r'self\.\w+\s*=',
        'description': 'Can assign to self.attribute',
        'prerequisite': 'init_method',
        'difficulty': 0.3,
    },
    'method_definition': {
        'pattern': r'def\s+\w+\s*\(self',
        'description': 'Can define instance methods',
        'prerequisite': 'class_definition',
        'difficulty': 0.3,
    },
    'power_operator': {
        'pattern': r'\*\*',
        'description': 'Knows Python uses ** for exponents',
        'prerequisite': None,
        'difficulty': 0.5,  # Common error
    },
    'sqrt_import': {
        'pattern': r'(import\s+math|from\s+math)',
        'description': 'Knows to import math module',
        'prerequisite': None,
        'difficulty': 0.4,
    },
    'math_sqrt': {
        'pattern': r'math\.sqrt|sqrt',
        'description': 'Can use sqrt function',
        'prerequisite': 'sqrt_import',
        'difficulty': 0.3,
    },
    'tuple_return': {
        'pattern': r'return\s+\(.+,.+\)|return\s+\w+,\s*\w+',
        'description': 'Can return tuples',
        'prerequisite': 'method_definition',
        'difficulty': 0.2,
    },
}

# Code templates representing different skill levels  
CODE_TEMPLATES = {
    'low_init': '''class Particle:
    def __init__(self, x, y, vx, vy, mass):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.mass = mass
''',
    'low_get_position': '''    def get_position(self):
        return (self.x, self.y)
    
    def get_velocity(self):
        return (self.vx, self.vy)
''',
    'low_update_wrong_power': '''    def update(self, other, dt):
        dx = other.x - self.x
        dy = other.y - self.y
        r = (dx^2 + dy^2)^0.5  # wrong power operator
        G = 6.674e-11
        a = G * other.mass / r^2
        ax = a * dx / r
        ay = a * dy / r
        self.vx = self.vx + ax * dt
        self.vy = self.vy + ay * dt
        self.x = self.x + self.vx * dt
        self.y = self.y + self.vy * dt
''',
    'low_update_fixed': '''    def update(self, other, dt):
        dx = other.x - self.x
        dy = other.y - self.y
        r = (dx**2 + dy**2)**0.5
        if r == 0:
            return
        G = 6.674e-11
        a = G * other.mass / (r**2)
        ax = a * dx / r
        ay = a * dy / r
        self.vx = self.vx + ax * dt
        self.vy = self.vy + ay * dt
        self.x = self.x + self.vx * dt
        self.y = self.y + self.vy * dt
''',
    'high_complete': '''class Particle:
    def __init__(self, x, y, vx, vy, mass):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.mass = mass
    
    def get_position(self):
        return (self.x, self.y)
    
    def get_velocity(self):
        return (self.vx, self.vy)
    
    def update(self, other, dt):
        G = 6.674e-11
        dx = other.x - self.x
        dy = other.y - self.y
        r = (dx**2 + dy**2)**0.5
        if r == 0:
            return
        a = G * other.mass / (r**2)
        ax = a * dx / r
        ay = a * dy / r
        self.vx += ax * dt
        self.vy += ay * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
''',
}


@dataclass
class SimStudentStepResult:
    """Result of a single simulation step."""
    step: int
    action: str
    rule_applied: str
    knowledge_state: Dict[str, bool]
    code: str
    output: str
    tests_passed: int
    tests_total: int
    success: bool


@dataclass
class SimStudentSimulationResult:
    """Result of a complete SimStudent simulation."""
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
    baseline_type: str = "simstudent"

    def to_dict(self) -> dict:
        return asdict(self)


class SimStudent:
    """
    SimStudent-inspired baseline using production rules.
    
    Unlike LLM baselines, this uses deterministic rules to generate code.
    Simulates knowledge acquisition through error observation.
    """

    def __init__(
        self,
        performance_level: Literal["low", "high"] = "low",
    ):
        self.performance_level = performance_level
        
        # Initialize knowledge state based on performance level
        self.knowledge = {}
        for rule_name, rule_data in PRODUCTION_RULES.items():
            if performance_level == "high":
                # High performers know more initially
                self.knowledge[rule_name] = random.random() > rule_data['difficulty']
            else:
                # Low performers know less
                self.knowledge[rule_name] = random.random() > (rule_data['difficulty'] + 0.3)
        
        # Track which errors have been seen (for learning)
        self.errors_seen = set()
        
    def _learn_from_error(self, error_msg: str):
        """Update knowledge based on observed errors (inductive learning)."""
        if 'unsupported operand type(s) for ^' in error_msg:
            self.knowledge['power_operator'] = True
            self.errors_seen.add('power_operator')
        if "NameError: name 'math'" in error_msg:
            self.knowledge['sqrt_import'] = True
            self.errors_seen.add('sqrt_import')
        if "AttributeError: 'Particle'" in error_msg:
            self.errors_seen.add('missing_method')

    def _generate_code(self, step: int) -> str:
        """Generate code based on current knowledge state."""
        if self.performance_level == "low":
            # Low performer: incremental construction with errors
            if step <= 2:
                return CODE_TEMPLATES['low_init']
            elif step <= 4:
                return CODE_TEMPLATES['low_init'] + CODE_TEMPLATES['low_get_position']
            elif step <= 8 and not self.knowledge.get('power_operator'):
                # Don't know ** yet, use ^
                return (CODE_TEMPLATES['low_init'] + 
                        CODE_TEMPLATES['low_get_position'] +
                        CODE_TEMPLATES['low_update_wrong_power'])
            else:
                # Eventually learn correct syntax
                return (CODE_TEMPLATES['low_init'] + 
                        CODE_TEMPLATES['low_get_position'] +
                        CODE_TEMPLATES['low_update_fixed'])
        else:
            # High performer: faster but still makes some mistakes
            if step <= 1:
                return CODE_TEMPLATES['low_init']
            elif step <= 2:
                return CODE_TEMPLATES['low_init'] + CODE_TEMPLATES['low_get_position']
            elif step <= 4 and not self.knowledge.get('power_operator'):
                return (CODE_TEMPLATES['low_init'] + 
                        CODE_TEMPLATES['low_get_position'] +
                        CODE_TEMPLATES['low_update_wrong_power'])
            else:
                return CODE_TEMPLATES['high_complete']

    def _decide_action(self, step: int, last_output: str) -> str:
        """Decide cognitive action based on state."""
        if step == 1:
            return "CONSTRUCTING"
        elif "Error" in last_output or "FAILED" in last_output:
            return "DEBUGGING"
        elif random.random() < 0.3:
            return "ASSESSING"
        else:
            return "CONSTRUCTING"

    def solve_problem(
        self,
        problem_description: str,
        problem_id: str,
        max_steps: int = 30,
    ) -> List[Dict[str, Any]]:
        """Run the SimStudent simulation."""
        oracle = IDEOracle(save_history=True)
        
        current_code = ""
        last_output = "(No output yet)"
        history = []
        solved = False
        
        for step in range(1, max_steps + 1):
            try:
                action = self._decide_action(step, last_output)
                current_code = self._generate_code(step)
                
                # Execute for DEBUGGING/ASSESSING
                tests_passed = 0
                tests_total = 0
                
                if action in ["DEBUGGING", "ASSESSING"] and current_code:
                    exec_result = oracle.test_code(
                        code=current_code, problem_id=problem_id
                    )
                    last_output = exec_result.stdout or exec_result.captured_output or "(No output)"
                    tests_passed = exec_result.passed_tests
                    tests_total = exec_result.total_tests
                    solved = exec_result.passed
                    
                    # Learn from errors
                    self._learn_from_error(last_output)
                elif action == "CONSTRUCTING":
                    last_output = "(Code drafted but not executed)"
                
                step_result = {
                    "step": step,
                    "cognitive_state": action,
                    "action": action,
                    "monologue": f"[SimStudent Rule-Based: Applying production rules with knowledge state]",
                    "rule_applied": "knowledge_acquisition",
                    "knowledge_state": dict(self.knowledge),
                    "code": current_code,
                    "output": last_output,
                    "tests_passed": tests_passed,
                    "tests_total": tests_total,
                    "success": solved,
                }
                
                history.append(step_result)
                
                logger.info(
                    f"  Step {step}: {action} - {'SOLVED!' if solved else f'{tests_passed}/{tests_total} tests'}"
                )
                
                if solved:
                    break
                    
            except Exception as e:
                logger.error(f"Step {step} failed: {e}")
                history.append({
                    "step": step,
                    "cognitive_state": "ERROR",
                    "action": "ERROR",
                    "monologue": f"[SimStudent Error: {e}]",
                    "error": str(e),
                    "code": current_code,
                    "output": last_output,
                    "success": False,
                })
                
        return history


def run_simstudent_simulation(
    run_id: int,
    performance_level: str,
    problem_id: str,
    max_steps: int,
    llm_model: str = None,  # Not used, but kept for interface consistency
) -> SimStudentSimulationResult:
    """Run a single SimStudent simulation."""
    logger.info(f"Starting SIMSTUDENT run {run_id} ({performance_level} performer)")
    
    start_time = time.time()
    
    student = SimStudent(performance_level=performance_level)
    
    # Load problem
    problem_def = IDEOracle.load_problem(problem_id)
    
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
        action = h.get("action", "UNKNOWN")
        if action in action_counts:
            action_counts[action] += 1
    
    final_step = history[-1] if history else {}
    
    result = SimStudentSimulationResult(
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
        baseline_type="simstudent",
    )
    
    logger.info(
        f"  Run {run_id} complete: {'SOLVED' if solved else 'NOT SOLVED'} in {result.total_steps} steps ({duration:.1f}s)"
    )
    
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Running SimStudent baseline test...")
    result = run_simstudent_simulation(
        run_id=1,
        performance_level="low",
        problem_id="particle_simulator",
        max_steps=30,
    )
    
    print(f"\nResult: {'SOLVED' if result.solved else 'NOT SOLVED'} in {result.total_steps} steps")
    print(f"Actions: {result.action_counts}")
    print(f"Duration: {result.duration_seconds:.1f}s")
