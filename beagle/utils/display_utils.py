"""
Display utilities for colored and formatted console output.

Provides Rich-based formatting for student simulation visualization.
"""

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.text import Text
from rich import box
from typing import Dict, Any, List, Optional

console = Console()


def print_attempt_header(attempt: int, max_attempts: int) -> None:
    """Print a colorful attempt header."""
    text = Text()
    text.append("═" * 70, style="bold blue")
    text.append("\n")
    text.append(f"  ATTEMPT {attempt}/{max_attempts}", style="bold cyan")
    text.append("\n")
    text.append("═" * 70, style="bold blue")
    console.print(text)


def print_student_state(
    metacog_state: str,
    action: str,
    intent: str,
    monitor_signal: str,
    inner_iteration: int = 0,
    hints_received: int = 0,
    markov_top1: Optional[str] = None,
    matched_markov: bool = True
) -> None:
    """Print formatted student state information."""

    # Color coding for metacognitive states
    metacog_colors = {
        "Planning": "cyan",
        "Enacting": "green",
        "Monitoring": "yellow",
        "Reflecting": "magenta",
        "Assistance": "red",
        "Planning and Enacting": "blue",
        "Enacting and Monitoring": "bright_green",
    }
    metacog_color = metacog_colors.get(metacog_state, "white")

    # Color coding for monitor signals
    signal_color = "green" if monitor_signal == "CONTINUE" else "red"

    # Create table
    table = Table(
        show_header=False,
        box=box.ROUNDED,
        border_style="bright_blue",
        pad_edge=False,
        collapse_padding=True
    )
    table.add_column("Key", style="bold cyan", width=20)
    table.add_column("Value", style="white")

    table.add_row(
        "🧠 Metacog State",
        f"[bold {metacog_color}]{metacog_state}[/bold {metacog_color}]"
    )

    # Show action with Markov override indicator if applicable
    action_display = f"[bold yellow]{action}[/bold yellow]"
    if not matched_markov and markov_top1:
        action_display += f" [dim](markov=[magenta]{markov_top1}[/magenta])[/dim]"

    table.add_row("🎯 Action", action_display)
    table.add_row(
        "💭 Intent",
        f"[dim]{intent[:80]}{'...' if len(intent) > 80 else ''}[/dim]"
    )
    table.add_row(
        "📊 Monitor",
        f"[bold {signal_color}]{monitor_signal}[/bold {signal_color}]"
    )
    table.add_row("🔢 Inner Loop", f"[cyan]{inner_iteration}[/cyan]")
    if hints_received > 0:
        table.add_row("💡 Hints Received", f"[yellow]{hints_received}[/yellow]")

    console.print(
        Panel(
            table,
            title="[bold white]👤 STUDENT STATE[/bold white]",
            border_style="bright_blue",
            padding=(0, 1)
        )
    )


def print_reasoning(reasoning: str) -> None:
    """Print student reasoning."""
    console.print(
        Panel(
            f"[italic]{reasoning}[/italic]",
            title="[bold magenta]💬 STUDENT REASONING[/bold magenta]",
            border_style="magenta",
            padding=(0, 1)
        )
    )


def print_code(code: str, title: str = "📝 STUDENT CODE") -> None:
    """Print code with syntax highlighting."""
    if code:
        console.print(
            Panel(
                Syntax(
                    code,
                    "python",
                    theme="native",
                    line_numbers=True,
                    background_color="default"
                ),
                title=f"[bold cyan]{title}[/bold cyan]",
                border_style="cyan",
                padding=(0, 1)
            )
        )


def print_execution_result(
    success: bool,
    output: str = "",
    error: str = "",
    test_results: List[Dict[str, Any]] = None
) -> None:
    """Print execution results with color coding."""

    if success:
        # Success case
        console.print(
            Panel(
                Text("✓ All tests passed!", style="bold green"),
                title="[bold green]✅ EXECUTION SUCCESS[/bold green]",
                border_style="green",
                padding=(0, 1)
            )
        )
    else:
        # Failure case - show details
        content = []

        if test_results:
            # Show test results table
            table = Table(show_header=True, box=box.SIMPLE, border_style="red")
            table.add_column("Test", style="white", width=25)
            table.add_column("Status", style="white", width=10)
            table.add_column("Error", style="dim white")

            for test in test_results:
                status = "✓ PASS" if test.get('passed', False) else "✗ FAIL"
                status_style = "green" if test.get('passed', False) else "red"

                # Handle None error values
                error_raw = test.get('error') or ''
                error_msg = error_raw[:60] if error_raw else ''
                if len(error_raw) > 60:
                    error_msg += "..."

                table.add_row(
                    test.get('name', 'Unknown'),
                    f"[{status_style}]{status}[/{status_style}]", error_msg
                )

            content.append(table)

        if error and not test_results:
            # Show raw error if no test results
            content.append(Text(error, style="red"))

        console.print(
            Panel(
                Group(*content) if content else Text("Execution failed", style="red"),
                title="[bold red]❌ EXECUTION FAILED[/bold red]",
                border_style="red",
                padding=(0, 1)
            )
        )


def print_hint(hint_text: str, hint_type: str = "corrective") -> None:
    """Print tutor hint."""
    icon = "💡" if hint_type == "corrective" else "❓"
    color = "yellow" if hint_type == "corrective" else "bright_yellow"

    console.print(
        Panel(
            Text(hint_text, style=f"{color}"),
            title=
            f"[bold {color}]{icon} TUTOR HINT ({hint_type.upper()})[/bold {color}]",
            border_style=color,
            padding=(0, 1)
        )
    )


def print_metacog_transition(
    from_state: str, to_state: str, reason: str
) -> None:
    """Print metacognitive state transition."""
    arrow = Text(" → ", style="bold white")

    content = Text()
    content.append(f"{from_state}", style="bold cyan")
    content.append(" → ", style="bold white")
    content.append(f"{to_state}", style="bold green")
    content.append("\n\n")
    content.append(f"Reason: {reason}", style="dim white")

    console.print(
        Panel(
            content,
            title="[bold magenta]🔄 METACOGNITIVE REGULATION[/bold magenta]",
            border_style="magenta",
            padding=(0, 1)
        )
    )


def print_session_summary(
    solved: bool, attempts: int, hints_used: int, final_metacog_state: str
) -> None:
    """Print session completion summary."""
    status = "✅ SOLVED" if solved else "❌ UNSOLVED"
    status_color = "green" if solved else "red"

    table = Table(
        show_header=False, box=box.ROUNDED, border_style=status_color
    )
    table.add_column("Metric", style="bold white")
    table.add_column("Value", style="white")

    table.add_row(
        "Final Status", f"[bold {status_color}]{status}[/bold {status_color}]"
    )
    table.add_row("Total Attempts", str(attempts))
    table.add_row("Hints Used", str(hints_used))
    table.add_row("Final State", final_metacog_state)

    console.print(
        Panel(
            table,
            title="[bold white]📊 SESSION SUMMARY[/bold white]",
            border_style=status_color,
            padding=(0, 1)
        )
    )


def print_separator() -> None:
    """Print a visual separator."""
    console.print()


def print_cavs_status(
    exec_state: Any,  # ExecutionState object
    current_turn: int
) -> None:
    """
    Print CAVS execution state status in a formatted box.
    
    Args:
        exec_state: ExecutionState object from beagle.data_generation.verification
        current_turn: Current turn number
    """
    # Extract values from ExecutionState
    exec_status = exec_state.status.value
    last_test_turn = exec_state.last_test_turn if exec_state.last_test_turn >= 0 else None
    untested_mods = exec_state.untested_modifications_count
    last_result = exec_state.last_run_result.value
    is_stale = exec_state.is_stale()

    # Color coding for execution status
    status_colors = {
        "NEVER_RUN": "dim white",
        "TESTED_PASS": "green",
        "TESTED_FAIL": "red",
        "EXECUTION_ERROR": "bright_red",
        "UNTESTED_MODIFICATION": "yellow"
    }
    status_color = status_colors.get(exec_status, "white")

    # Color for staleness indicator
    stale_indicator = "🚨 STALE" if is_stale else "✓ Fresh"
    stale_color = "red" if is_stale else "green"

    # Create table
    table = Table(
        show_header=False,
        box=box.ROUNDED,
        border_style="bright_magenta",
        pad_edge=False,
        collapse_padding=True
    )
    table.add_column("Parameter", style="bold magenta", width=22)
    table.add_column("Value", style="white")

    # Add rows
    table.add_row("📍 Current Turn", f"[cyan]{current_turn}[/cyan]")
    table.add_row(
        "🔍 Exec Status",
        f"[bold {status_color}]{exec_status}[/bold {status_color}]"
    )
    table.add_row(
        "🧪 Last Test Turn",
        f"[cyan]{last_test_turn if last_test_turn else 'Never'}[/cyan]"
    )
    table.add_row("📝 Untested Mods", f"[yellow]{untested_mods}[/yellow]")
    table.add_row(
        "📊 Last Result", f"[{status_color}]{last_result}[/{status_color}]"
    )
    table.add_row(
        "⏱️  Freshness", f"[{stale_color}]{stale_indicator}[/{stale_color}]"
    )

    console.print(
        Panel(
            table,
            title="[bold white]🛡️  CAVS EXECUTION STATE[/bold white]",
            border_style="bright_magenta",
            padding=(0, 1)
        )
    )


def print_cavs_override(
    decision: Any,  # CAVSDecision object
    intended_action: str,
    profile: str
) -> None:
    """
    Print CAVS override decision in a formatted box.
    
    Args:
        decision: CAVSDecision object from beagle.data_generation.verification
        intended_action: Action the student intended to take
        profile: Student profile (STRUGGLING, AVERAGE, ADVANCED)
    """
    # Extract values from CAVSDecision
    trigger = decision.override_trigger.value if decision.override_trigger else "UNKNOWN"
    reason = decision.override_reason or "No reason provided"
    forced_action = "run_code"  # CAVS always forces run_code on override

    # Create content
    content = Text()

    # Trigger
    content.append("⚠️  TRIGGER: ", style="bold red")
    content.append(f"{trigger}\n", style="bold yellow")

    # Profile
    content.append("👤 PROFILE: ", style="bold cyan")
    content.append(f"{profile}\n\n", style="cyan")

    # Actions
    content.append("Original Action: ", style="dim white")
    content.append(f"{intended_action}\n", style="yellow")
    content.append("Forced Action: ", style="bold white")
    content.append(f"{forced_action}\n\n", style="bold green")

    # Reason
    content.append("Reason:\n", style="bold white")
    content.append(f"{reason}", style="white")

    console.print(
        Panel(
            content,
            title="[bold red]🛡️  CAVS OVERRIDE ACTIVE[/bold red]",
            border_style="red",
            padding=(1, 2)
        )
    )


def print_cavs_guidance(
    decision: Any,  # CAVSDecision object
    exec_state: Any,  # ExecutionState object
    policy: Any,  # VerificationPolicy object
    profile: str
) -> None:
    """
    Print CAVS guidance nudge in a formatted box.
    
    Args:
        decision: CAVSDecision object from beagle.data_generation.verification
        exec_state: ExecutionState object
        policy: VerificationPolicy object from CAVSController
        profile: Student profile string
    """
    # Extract values
    message = decision.guidance_message or "Consider running tests to verify your changes."
    untested_mods = exec_state.untested_modifications_count
    threshold = policy.get_max_untested_mods(exec_state)

    content = Text()

    # Warning level indicator
    content.append("💡 GUIDANCE: ", style="bold yellow")
    content.append(f"Approaching verification threshold\n\n", style="yellow")

    # Stats
    content.append("Profile: ", style="bold white")
    content.append(f"{profile}\n", style="cyan")
    content.append("Untested Modifications: ", style="bold white")
    content.append(f"{untested_mods}/{threshold}\n\n", style="yellow")

    # Message
    content.append(message, style="white")

    console.print(
        Panel(
            content,
            title="[bold yellow]💡 CAVS GUIDANCE[/bold yellow]",
            border_style="yellow",
            padding=(1, 2)
        )
    )
