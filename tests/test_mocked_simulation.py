#!/usr/bin/env python
"""
Mocked 20-Step Simulation Test for V34 Output Parser

This script runs a comprehensive simulation that:
1. Uses the REAL IDE oracle to generate actual pytest output
2. Mocks the LLM calls with predetermined code/monologue
3. Tests the full flow: raw output → blindness filter → parser → context
4. Shows exactly what the LLM would see at each step

Run with: python tests/test_v34_mocked_simulation.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from beagle.data_generation.ide_oracle.ide_oracle import IDEOracle
from beagle.data_generation.studentv2.nodes import (
    filter_feedback_for_state,
    _format_executor_context,
    _format_strategist_context
)
from beagle.data_generation.studentv2.output_parser import parse_execution_output


# ==============================================================================
# MOCK DATA: Simulated student code progression over 20 steps
# ==============================================================================

MOCK_CODE_PROGRESSION = [
    # Step 1: Empty Particle class
    '''class Particle:
    pass
''',
    # Step 2: Add __init__ but wrong signature
    '''class Particle:
    def __init__(self, x, y, mass):
        self.x = x
        self.y = y
        self.mass = mass
''',
    # Step 3: Add vx, vy to __init__
    '''class Particle:
    def __init__(self, x, y, vx, vy, mass):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.mass = mass
''',
    # Step 4: Add get_position
    '''class Particle:
    def __init__(self, x, y, vx, vy, mass):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.mass = mass
    
    def get_position(self):
        return (self.x, self.y)
''',
    # Step 5: Add get_velocity
    '''class Particle:
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
''',
    # Step 6: Add update (with bug - missing acceleration)
    '''class Particle:
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
    
    def update(self, dt, ax, ay):
        self.x += self.vx * dt
        self.y += self.vy * dt
''',
    # Step 7: Fix update to include acceleration
    '''class Particle:
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
    
    def update(self, dt, ax, ay):
        self.vx += ax * dt
        self.vy += ay * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
''',
    # Step 8: Add apply_force (with bug)
    '''class Particle:
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
    
    def update(self, dt, ax, ay):
        self.vx += ax * dt
        self.vy += ay * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
    
    def apply_force(self, fx, fy, dt):
        ax = fx / self.mass
        ay = fy / self.mass
        self.update(dt, ax, ay)
''',
    # Step 9: Add get_kinetic_energy
    '''class Particle:
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
    
    def update(self, dt, ax, ay):
        self.vx += ax * dt
        self.vy += ay * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
    
    def apply_force(self, fx, fy, dt):
        ax = fx / self.mass
        ay = fy / self.mass
        self.update(dt, ax, ay)
    
    def get_kinetic_energy(self):
        speed_sq = self.vx**2 + self.vy**2
        return 0.5 * self.mass * speed_sq
''',
    # Step 10-20: Keep same working code or minor variations
]

# Extend to 20 steps by repeating the best version
while len(MOCK_CODE_PROGRESSION) < 20:
    MOCK_CODE_PROGRESSION.append(MOCK_CODE_PROGRESSION[-1])


# Metacognitive state sequence (mix of states)
METACOG_SEQUENCE = [
    'Planning', 'Enacting', 'Monitoring', 'Enacting', 'Planning',
    'Enacting', 'Monitoring', 'Reflecting', 'Enacting', 'Planning',
    'Monitoring', 'Enacting', 'Reflecting', 'Enacting', 'Monitoring',
    'Planning', 'Enacting', 'Monitoring', 'Reflecting', 'Monitoring'
]


class MockCtx:
    """Mock context for testing context formatters."""
    def __init__(self):
        self.state = MockState()
        
class MockState:
    current_code = ""
    problem_description = "Implement a Particle class for physics simulation"


def run_mocked_simulation():
    """Run 20-step mocked simulation with real IDE oracle."""
    
    print("=" * 70)
    print("V34 MOCKED SIMULATION TEST - 20 STEPS")
    print("=" * 70)
    print()
    print("This test uses REAL pytest execution but MOCKS the LLM calls.")
    print("It shows exactly what the LLM would see at each step.")
    print()
    
    # Initialize real IDE oracle
    oracle = IDEOracle()
    problem_id = 'particle_simulator'
    
    results = []
    
    for step in range(1, 21):
        code = MOCK_CODE_PROGRESSION[step - 1]
        metacog = METACOG_SEQUENCE[step - 1]
        
        print(f"\n{'=' * 70}")
        print(f"STEP {step} | Metacog: {metacog}")
        print("=" * 70)
        
        # --- 1. Execute code with REAL oracle ---
        result = oracle.test_code(code=code, problem_id=problem_id)
        raw_output = result.stdout or result.captured_output
        
        print(f"\n[1] REAL EXECUTION:")
        print(f"    Passed: {result.passed_tests}/{result.total_tests}")
        print(f"    Success: {result.passed}")
        
        # --- 2. Apply epistemic blindness based on metacog state ---
        filtered_output = filter_feedback_for_state(raw_output, metacog)
        
        print(f"\n[2] AFTER BLINDNESS FILTER ({metacog}):")
        if metacog == 'Enacting':
            print(f"    (Blurred): {filtered_output[:80]}...")
        else:
            print(f"    (Full access): {len(raw_output)} chars")
        
        # --- 3. Parse the filtered output ---
        parsed = parse_execution_output(filtered_output)
        
        print(f"\n[3] AFTER PARSER:")
        print(f"    Status: {parsed.status_header}")
        print(f"    Summary: {parsed.summary_line}")
        print(f"    Error Type: {parsed.error_type}")
        if parsed.primary_error and parsed.error_type != 'success':
            print(f"    Primary Error: {parsed.primary_error[:80]}...")
        
        # --- 4. What would go into context formatter ---
        ctx = MockCtx()
        ctx.state.current_code = code
        
        # The display_log is what goes into the context
        print(f"\n[4] WHAT LLM SEES (display_log preview):")
        display_preview = parsed.display_log[:200]
        for line in display_preview.split('\n')[:5]:
            print(f"    {line}")
        if len(parsed.display_log) > 200:
            print(f"    ...")
        
        # Store results
        results.append({
            'step': step,
            'metacog': metacog,
            'passed': result.passed_tests,
            'total': result.total_tests,
            'blindness_applied': metacog == 'Enacting',
            'parsed_status': parsed.status_header,
            'error_type': parsed.error_type
        })
    
    # --- Summary ---
    print("\n" + "=" * 70)
    print("SIMULATION SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Step':<6} {'Metacog':<12} {'Tests':<10} {'Blind?':<8} {'Status':<40}")
    print("-" * 70)
    
    for r in results:
        blind = "YES" if r['blindness_applied'] else "NO"
        tests = f"{r['passed']}/{r['total']}"
        status = r['parsed_status'][:38]
        print(f"{r['step']:<6} {r['metacog']:<12} {tests:<10} {blind:<8} {status}")
    
    print()
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    
    # Validate key behaviors
    enacting_steps = [r for r in results if r['blindness_applied']]
    full_vision_steps = [r for r in results if not r['blindness_applied']]
    
    print()
    print("VALIDATION:")
    print(f"  - Enacting steps (blindness applied): {len(enacting_steps)}")
    print(f"  - Full-vision steps: {len(full_vision_steps)}")
    
    # Check that Enacting steps got fallback parsing
    enacting_with_unknown = [r for r in enacting_steps if r['error_type'] == 'unknown']
    print(f"  - Enacting steps with 'unknown' error_type: {len(enacting_with_unknown)}")
    
    # Check that full-vision steps got proper parsing
    full_vision_with_parsing = [r for r in full_vision_steps if r['error_type'] in ['crash', 'failure', 'success']]
    print(f"  - Full-vision steps with proper parsing: {len(full_vision_with_parsing)}")
    
    if len(enacting_with_unknown) == len(enacting_steps):
        print("  [OK] All Enacting steps correctly received fallback parsing")
    else:
        print("  [WARN] Some Enacting steps may have unexpected parsing")
    
    if len(full_vision_with_parsing) == len(full_vision_steps):
        print("  [OK] All full-vision steps correctly received detailed parsing")
    else:
        print("  [WARN] Some full-vision steps may have parsing issues")


if __name__ == "__main__":
    run_mocked_simulation()
