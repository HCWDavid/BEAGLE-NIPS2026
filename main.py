import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from beagle.data_generation.ide_oracle.ide_oracle import IDEOracle
from beagle.data_generation.studentv2.student import Student
from beagle.utils.display_utils import (
    print_code, print_execution_result, print_student_state
)

# Add project root to path
# This allows for importing modules relative to the project root.
_project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if str(_project_root) not in sys.path:
    sys.path.insert(0, _project_root)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)  # Correctly initialize the logger instance

# Suppress noisy google_genai logs (AFC enabled messages)
logging.getLogger("google_genai").setLevel(logging.WARNING)

# Setup Logfire (optional)
try:
    import logfire
    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False
    logfire = None  # Ensure logfire is None if not available


def parse_args():
    parser = argparse.ArgumentParser(description="BEAGLE V2 Simulation")

    parser.add_argument(
        '--project-path',
        type=str,
        default=str(Path(__file__).resolve().parent),
        help='Path to the project to work on. Defaults to current directory.'
    )

    parser.add_argument(
        '--problem',
        type=str,
        default='particle_simulator',
        help='Problem ID (folder name in beagle/data_generation/problems)'
    )

    parser.add_argument(
        '--performance',
        type=str,
        default='low',
        choices=['low', 'high'],
        help='Performance level of the student'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='google-gla:gemini-2.5-flash',  # Reverted default model
        help='LLM model to use'
    )

    parser.add_argument(
        '--max-steps', type=int, default=20, help='Maximum number of steps'
    )

    parser.add_argument(
        '--enable-logfire',
        action='store_true',
        help='Enable Logfire tracing for detailed observability.'
    )

    parser.add_argument(
        '--duration-multiplier',
        type=float,
        default=0.5,
        help='Multiplier for metacognitive segment durations (default: 0.5 = half duration)'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    os.chdir(args.project_path)  # Change working directory if specified

    print(f"\n{'-'*70}")
    print(f"BEAGLE V2 SIMULATION")
    print(f"{'-'*70}")
    print(f"Problem: {args.problem}")
    print(f"Performance: {args.performance}")
    print(f"Model: {args.model}")
    print(f"{'-'*70}\n")

    # Logfire initialization
    logfire_project = None
    logfire_username = None
    if args.enable_logfire and LOGFIRE_AVAILABLE:
        try:
            logfire.configure(
                service_name="beagle_v2_simulation",
                # Removed console_exporter_on as it's not a valid argument
            )
            logfire.instrument_pydantic_ai(
            )  # Corrected instrumentation method
            logfire_project = os.getenv('LOGFIRE_PROJECT', 'beagle')
            logfire_token = os.getenv('LOGFIRE_TOKEN')
            logfire_username = os.getenv('LOGFIRE_USERNAME')

            print(f"\n✓ Logfire tracing enabled: {logfire_project}")
            print("  → LLM calls will show Messages/Tool calls/Files in UI")
            if logfire_token:
                print(f"  → Sending traces to the Logfire dashboard.")

        except Exception as e:
            print(
                f"\n⚠️  Logfire initialization failed: {e} - tracing disabled"
            )
            log.error(f"Logfire initialization failed: {e}")  # Log the error
            args.enable_logfire = False
    elif args.enable_logfire and not LOGFIRE_AVAILABLE:
        print(
            "\n⚠️  Logfire not installed - tracing disabled (install with: pip install logfire)"
        )

    # 1. Load Problem
    try:
        problem_def = IDEOracle.load_problem(
            args.problem
        )  # Corrected to use IDEOracle.load_problem
        print(f"✓ Problem loaded: {problem_def.title}")
    except Exception as e:
        log.error(f"Failed to load problem: {e}")
        if args.enable_logfire and logfire:  # Check if logfire is available before using
            logfire.error(f"Failed to load the problem: {e}")
        return

    # 2. Initialize Oracle
    oracle = IDEOracle(save_history=True)
    print(f"✓ Oracle initialized")

    # 3. Initialize Student
    model_path = "beagle/data_generation/studentv2/semi_markov_model.joblib"  # Updated to Semi-Markov
    # Adjust model_path to be relative to the project root
    abs_model_path = Path(os.getcwd()) / model_path

    if not abs_model_path.exists():
        log.error(f"Markov model not found at {abs_model_path}")
        if args.enable_logfire and logfire:  # Check if logfire is available before using
            logfire.error(f"Markov model not found at {abs_model_path}")
        return

    student = Student(
        performance_level=args.performance,
        model_path=str(abs_model_path),
        llm_model=args.model,
        ide_oracle=oracle,
        duration_multiplier=args.duration_multiplier
    )
    print(f"✓ Student V2 initialized (duration_multiplier={args.duration_multiplier})")

    # 4. Run Simulation
    print(f"\nStarting simulation (max {args.max_steps} steps)...")
    if args.enable_logfire and logfire:  # Check if logfire is available before using
        with logfire.span(
            "beagle_v2_simulation_run",
            problem=args.problem,
            performance=args.performance,
            model=args.model,
            max_steps=args.max_steps
        ):
            history = student.solve_problem(
                problem_description=problem_def.description,
                problem_id=problem_def.problem_id,
                required_kcs=problem_def.required_kcs,
                max_steps=args.max_steps
            )
    else:
        history = student.solve_problem(
            problem_description=problem_def.description,
            problem_id=problem_def.problem_id,
            required_kcs=problem_def.required_kcs,
            max_steps=args.max_steps
        )

    # 5. Report Results (verbose history printing already there)

    print(f"\n{'-'*70}")
    print(f"SIMULATION RESULTS")
    print(f"{'-'*70}")

    solved = False
    final_code = ""

    if history:
        for i, step in enumerate(history):
            print(f"\n--- Step {i+1} ---")

            # Use display_utils for nicer formatting
            print_student_state(
                metacog_state=step.get('metacognitive_state', 'N/A'),
                action=step.get('cognitive_state', 'N/A'
                                ),  # Using cognitive state as 'action' proxy
                intent=step.get('goal', 'N/A'),
                monitor_signal=step.get(
                    'mindset', 'N/A'
                ),  # Using mindset as monitor signal proxy
                inner_iteration=i + 1
            )

            if step.get('code'):
                print_code(step.get('code'), title=f"Step {i+1} Code")

            # Handle success being None, True, or False
            success = step.get('success')
            if success is not None:
                print_execution_result(
                    success=success,
                    output=step.get('output', ''),
                    error=step.get('error', '')
                )
            elif step.get('output'):
                print(
                    f"  Output: {step.get('output')[:100]}..."
                    if len(step.get('output', '')) >
                    100 else f"  Output: {step.get('output')}"
                )

        last_step = history[-1]
        solved = last_step.get('success', False)
        final_code = last_step.get('code', "")

        print(f"\nSteps taken: {len(history)}")
        print(f"Problem Solved: {'✓ YES' if solved else '✗ NO'}")

        print("\nFinal Code:")
        print("-" * 40)
        print(final_code)
        print("-" * 40)

        if solved:
            print("\nSUCCESS! The student solved the problem.")
        else:
            print("\nStudent failed to solve the problem within step limit.")
            print(f"Last Error: {last_step.get('error', 'None')}")
    else:
        print("No history generated.")

    # 6. Export History to JSON
    if history:
        output_file = Path("simulation_history.json")
        with open(output_file, "w") as f:
            json.dump(history, f, indent=2, default=str)
        print(f"\n✓ Simulation history exported to {output_file.absolute()}")


if __name__ == "__main__":
    main()
