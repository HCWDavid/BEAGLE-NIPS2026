"""
Automated Assessment Oracle (AAO)

Provides fine-grained, KC-level assessment of student code.

Instead of binary pass/fail, the AAO analyzes:
1. Did the student correctly apply KC_C2_MATH_LIBRARY? (import math present)
2. Did the student correctly apply KC_P2_TRIG_APPLICATION? (cos for horizontal, sin for vertical)
3. Did the student correctly apply KC_C1_FUNCTION_DEF_RETURN? (function definition with return)
...and so on for each KC.

This breaks the infinite loop problem where:
- Student writes "import math" (correct for KC_C2)
- But code fails for other reasons
- BKT sees overall failure and doesn't credit the correct import
- Director keeps activating M4_NO_IMPORT misconception
- Cycle repeats!

With AAO:
- Student writes "import math" (correct for KC_C2)
- Code fails for other reasons
- AAO recognizes: "KC_C2 = CORRECT, other KCs = INCORRECT"
- BKT updates KC_C2 as correct (mastery increases)
- Director stops activating M4_NO_IMPORT
- Student can move forward!
"""

import ast
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# KC constants for assessment
# Conceptual KCs (C1-C12)
KC_C1 = "KC_C1_FUNCTION_DEF_RETURN"
KC_C2 = "KC_C2_MATH_LIBRARY"
KC_C4 = "KC_C4_ARITHMETIC_IMPLEMENTATION"
KC_C9 = "KC_C9_CLASS_DEFINITION"
KC_C10 = "KC_C10_INIT_METHOD"
KC_C11 = "KC_C11_INSTANCE_VARIABLES"
KC_C12 = "KC_C12_METHOD_DEFINITION"

# Additional Conceptual KCs for gradient descent (C13-C16)
KC_C13 = "KC_C13_LOOP_CONSTRUCT"
KC_C14 = "KC_C14_CONDITIONAL_LOGIC"
KC_C15 = "KC_C15_VARIABLE_UPDATE"
KC_C16 = "KC_C16_FUNCTION_CALL"

# Physics KCs (P1-P11)
KC_P1 = "KC_P1_VECTOR_DECOMPOSITION"
KC_P2 = "KC_P2_TRIG_APPLICATION"
KC_P5 = "KC_P5_UNIT_RADIANS"
KC_P9 = "KC_P9_NUMERICAL_INTEGRATION"
KC_P10 = "KC_P10_FORCE_ACCELERATION"
KC_P11 = "KC_P11_KINETIC_ENERGY"

# Additional Physics KCs (P12-P16) - new problems
KC_P14 = "KC_P14_COLLISION_REFLECTION"
KC_P16 = "KC_P16_FRICTION"

# Math KCs for gradient descent (M1-M5)
KC_M1 = "KC_M1_DERIVATIVE_CONCEPT"
KC_M2 = "KC_M2_NUMERICAL_DIFFERENTIATION"
KC_M3 = "KC_M3_ITERATIVE_ALGORITHM"
KC_M4 = "KC_M4_CONVERGENCE_CRITERIA"
KC_M5 = "KC_M5_LEARNING_RATE"


@dataclass
class KCAssessment:
    """Assessment of a single KC."""
    kc_id: str
    is_correct: bool
    evidence: str  # Why we think it's correct/incorrect
    confidence: float  # 0.0-1.0, how confident we are in this assessment


@dataclass
class CodeAssessment:
    """Complete assessment of student code."""
    kc_assessments: Dict[str, KCAssessment]  # KC_ID -> Assessment
    overall_success: bool
    execution_error: Optional[str]
    ast_tree: Optional[ast.AST]


class AutomatedAssessmentOracle:
    """
    Analyzes student code to provide KC-level feedback.

    Uses AST analysis + error interpretation to determine
    which KCs were correctly applied, independent of overall pass/fail.
    """

    @staticmethod
    def _remove_comments(code: str) -> str:
        """Remove Python comments to avoid false positives from comment text."""
        return "\n".join(line.split("#")[0] for line in code.split("\n"))

    def assess_code(
        self, code: str, execution_result: Dict, required_kcs: List[str]
    ) -> CodeAssessment:
        """
        Assess student code at the KC level.

        Args:
            code: The generated Python code
            execution_result: Result from code executor
            required_kcs: KCs that should be present in this step

        Returns:
            CodeAssessment with per-KC feedback
        """
        # Parse AST
        ast_tree = None
        try:
            ast_tree = ast.parse(code)
        except SyntaxError:
            # Can't parse, all KCs fail
            return self._all_kcs_failed(
                required_kcs, "Syntax error prevents analysis"
            )

        # Assess each KC
        kc_assessments = {}

        for kc_id in required_kcs:
            assessment = self._assess_kc(
                kc_id, code, ast_tree, execution_result
            )
            kc_assessments[kc_id] = assessment

        return CodeAssessment(
            kc_assessments=kc_assessments,
            overall_success=execution_result.get("success", False),
            execution_error=execution_result.get("error"),
            ast_tree=ast_tree
        )

    def _assess_kc(
        self, kc_id: str, code: str, ast_tree: ast.AST, execution_result: Dict
    ) -> KCAssessment:
        """Assess a single KC."""

        # Remove comments once to avoid false positives across all assessments
        code_no_comments = self._remove_comments(code)

        # Route to specific assessment function
        # Conceptual KCs
        if kc_id == KC_C1:
            return self._assess_function_def(code_no_comments, ast_tree)
        elif kc_id == KC_C2:
            return self._assess_math_import(
                code_no_comments, ast_tree, execution_result
            )
        elif kc_id == KC_C4:
            return self._assess_arithmetic(code_no_comments, ast_tree)
        elif kc_id == KC_C9:
            return self._assess_class_definition(code_no_comments, ast_tree)
        elif kc_id == KC_C10:
            return self._assess_init_method(code_no_comments, ast_tree)
        elif kc_id == KC_C11:
            return self._assess_instance_variables(code_no_comments, ast_tree)
        elif kc_id == KC_C12:
            return self._assess_method_definition(code_no_comments, ast_tree)
        # Physics KCs
        elif kc_id == KC_P1:
            return self._assess_vector_decomposition(
                code_no_comments, ast_tree
            )
        elif kc_id == KC_P2:
            return self._assess_trig_application(code_no_comments, ast_tree)
        elif kc_id == KC_P5:
            return self._assess_radian_conversion(code_no_comments, ast_tree)
        elif kc_id == KC_P9:
            return self._assess_numerical_integration(
                code_no_comments, ast_tree
            )
        elif kc_id == KC_P10:
            return self._assess_force_acceleration(code_no_comments, ast_tree)
        elif kc_id == KC_P11:
            return self._assess_kinetic_energy(code_no_comments, ast_tree)
        # Additional Physics KCs (P12-P16) - new problems
        elif kc_id == KC_P14:
            return self._assess_collision_reflection(code_no_comments, ast_tree)
        elif kc_id == KC_P16:
            return self._assess_friction(code_no_comments, ast_tree)
        # Additional Conceptual KCs for gradient descent (C13-C16)
        elif kc_id == KC_C13:
            return self._assess_loop_construct(code_no_comments, ast_tree)
        elif kc_id == KC_C14:
            return self._assess_conditional_logic(code_no_comments, ast_tree)
        elif kc_id == KC_C15:
            return self._assess_variable_update(code_no_comments, ast_tree)
        elif kc_id == KC_C16:
            return self._assess_function_call(code_no_comments, ast_tree)
        # Math KCs for gradient descent (M1-M5)
        elif kc_id == KC_M1:
            return self._assess_derivative_concept(code_no_comments, ast_tree)
        elif kc_id == KC_M2:
            return self._assess_numerical_differentiation(code_no_comments, ast_tree)
        elif kc_id == KC_M3:
            return self._assess_iterative_algorithm(code_no_comments, ast_tree)
        elif kc_id == KC_M4:
            return self._assess_convergence_criteria(code_no_comments, ast_tree)
        elif kc_id == KC_M5:
            return self._assess_learning_rate(code_no_comments, ast_tree)
        else:
            # Hard error: KC must be defined with an assessment method
            defined_kcs = [
                KC_C1, KC_C2, KC_C4, KC_C9, KC_C10, KC_C11, KC_C12,
                KC_C13, KC_C14, KC_C15, KC_C16,
                KC_P1, KC_P2, KC_P5, KC_P9, KC_P10, KC_P11,
                KC_P14, KC_P16,
                KC_M1, KC_M2, KC_M3, KC_M4, KC_M5
            ]
            raise ValueError(
                f"Unknown KC '{kc_id}' has no assessment method defined in AutomatedAssessmentOracle. "
                f"Defined KCs: {defined_kcs}. "
                f"Please add an _assess_* method for this KC."
            )

    # ========== KC Assessment Methods ==========

    def _assess_function_def(
        self, code: str, ast_tree: ast.AST
    ) -> KCAssessment:
        """KC_C1: Function definition with return statement."""
        # Check for function definition
        has_function = False
        has_return = False
        correct_signature = False

        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef):
                has_function = True
                if node.name == "projectile_range":
                    correct_signature = len(node.args.args) == 3
                # Check for return statement
                for subnode in ast.walk(node):
                    if isinstance(
                        subnode, ast.Return
                    ) and subnode.value is not None:
                        has_return = True
                        break

        if correct_signature and has_return:
            return KCAssessment(
                KC_C1, True,
                "Function defined with correct signature and return", 1.0
            )
        elif has_function and has_return:
            return KCAssessment(
                KC_C1, True,
                "Function defined with return (signature may vary)", 0.8
            )
        elif has_function:
            return KCAssessment(
                KC_C1, False, "Function defined but missing return", 0.3
            )
        else:
            return KCAssessment(
                KC_C1, False, "No function definition found", 1.0
            )

    def _assess_math_import(
        self, code: str, ast_tree: ast.AST, execution_result: Dict
    ) -> KCAssessment:
        """KC_C2: Import math library."""
        # Check AST for import
        has_math_import = False
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "math":
                        has_math_import = True
                        break
            elif isinstance(node, ast.ImportFrom):
                if node.module == "math":
                    has_math_import = True
                    break

        # Check execution error
        error = execution_result.get("error", "")
        has_math_error = "name 'math' is not defined" in (error or "").lower()

        if has_math_import and not has_math_error:
            return KCAssessment(KC_C2, True, "math imported correctly", 1.0)
        elif has_math_import and has_math_error:
            # Import present but still error - might be scope issue
            return KCAssessment(
                KC_C2, True, "math imported (error may be unrelated)", 0.7
            )
        elif not has_math_import and has_math_error:
            return KCAssessment(
                KC_C2, False, "math not imported (confirmed by error)", 1.0
            )
        else:
            return KCAssessment(KC_C2, False, "math not imported", 0.9)


    def _assess_arithmetic(self, code: str, ast_tree: ast.AST) -> KCAssessment:
        """KC_C4: Arithmetic operations."""
        has_arithmetic = False

        for node in ast.walk(ast_tree):
            if isinstance(node, (ast.BinOp, ast.UnaryOp)):
                has_arithmetic = True
                break

        if has_arithmetic:
            return KCAssessment(
                KC_C4, True, "Arithmetic operations present", 0.8
            )
        else:
            return KCAssessment(KC_C4, False, "No arithmetic operations", 0.9)

    def _assess_vector_decomposition(
        self, code: str, ast_tree: ast.AST
    ) -> KCAssessment:
        """KC_P1: Vector decomposition concept."""
        # Look for two separate velocity components with various naming conventions
        code_lower = code.lower()
        has_v_x = (
            "v_x" in code or "vx" in code or "v_horizontal" in code
            or "horizontal" in code_lower or "h_" in code or "vel_x" in code
        )
        has_v_y = (
            "v_y" in code or "vy" in code or "v_vertical" in code
            or "vertical" in code_lower or "v_" in code or "vel_y" in code
        )

        if has_v_x and has_v_y:
            return KCAssessment(
                KC_P1, True, "Both velocity components calculated", 0.9
            )
        elif has_v_x or has_v_y:
            return KCAssessment(
                KC_P1, True, "One velocity component (partial)", 0.5
            )
        else:
            return KCAssessment(
                KC_P1, False, "No velocity component decomposition", 0.8
            )

    def _assess_trig_application(
        self, code: str, ast_tree: ast.AST
    ) -> KCAssessment:
        """KC_P2: Trigonometry application (cos for horizontal, sin for vertical)."""
        # Check for cos and sin usage
        uses_cos = "cos" in code.lower()
        uses_sin = "sin" in code.lower()

        # Ideally, check if cos is used for v_x and sin for v_y
        # Simple heuristic: both should be present
        if uses_cos and uses_sin:
            # Check for correct usage (rough heuristic)
            cos_before_sin = code.lower().find("cos"
                                               ) < code.lower().find("sin")
            if cos_before_sin:
                return KCAssessment(
                    KC_P2, True, "cos and sin used (likely correct order)", 0.9
                )
            else:
                return KCAssessment(
                    KC_P2, False, "cos and sin present but possibly swapped",
                    0.6
                )
        elif uses_cos or uses_sin:
            return KCAssessment(KC_P2, True, "Partial trig usage", 0.5)
        else:
            return KCAssessment(KC_P2, False, "No trigonometry used", 1.0)

    def _assess_radian_conversion(
        self, code: str, ast_tree: ast.AST
    ) -> KCAssessment:
        """KC_P5: Angle conversion to radians."""

        # Check if math.radians() is called or radians() from 'from math import radians'
        has_radians_call = False
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Call):
                # Check for math.radians()
                if isinstance(node.func, ast.Attribute):
                    if (
                        isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "math"
                        and node.func.attr == "radians"
                    ):
                        has_radians_call = True
                        break
                # Check for radians() (from math import radians)
                elif isinstance(
                    node.func, ast.Name
                ) and node.func.id == "radians":
                    has_radians_call = True
                    break

        # Check for manual conversion: angle * pi / 180 or angle * math.pi / 180
        has_pi_conversion = "*" in code and "pi" in code.lower(
        ) and "180" in code

        if has_radians_call:
            return KCAssessment(
                KC_P5, True, "Uses radians conversion function", 1.0
            )
        elif has_pi_conversion:
            return KCAssessment(
                KC_P5, True, "Manual radian conversion (theta * pi / 180)", 0.9
            )
        else:
            return KCAssessment(
                KC_P5, False, "No radian conversion found", 0.9
            )


    # ========== OOP KC Assessment Methods (C9-C12) ==========

    def _assess_class_definition(
        self, code: str, ast_tree: ast.AST
    ) -> KCAssessment:
        """KC_C9: Class definition."""
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                return KCAssessment(
                    KC_C9, True, f"Class '{class_name}' defined", 1.0
                )
        return KCAssessment(KC_C9, False, "No class definition found", 1.0)

    def _assess_init_method(
        self, code: str, ast_tree: ast.AST
    ) -> KCAssessment:
        """KC_C10: __init__ method with parameters."""
        has_class = False
        has_init = False
        has_params = False

        for node in ast.walk(ast_tree):
            if isinstance(node, ast.ClassDef):
                has_class = True
                for item in node.body:
                    if isinstance(
                        item, ast.FunctionDef
                    ) and item.name == "__init__":
                        has_init = True
                        # Check for parameters beyond 'self'
                        if len(item.args.args) > 1:
                            has_params = True
                        break

        if has_init and has_params:
            return KCAssessment(
                KC_C10, True, "__init__ method with parameters defined", 1.0
            )
        elif has_init:
            return KCAssessment(
                KC_C10, False, "__init__ defined but missing parameters", 0.5
            )
        elif has_class:
            return KCAssessment(
                KC_C10, False, "Class exists but no __init__ method", 0.8
            )
        else:
            return KCAssessment(KC_C10, False, "No class definition", 1.0)

    def _assess_instance_variables(
        self, code: str, ast_tree: ast.AST
    ) -> KCAssessment:
        """KC_C11: Instance variables (self.x = ...)."""
        has_self_assignment = False
        instance_var_count = 0

        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute):
                        if isinstance(
                            target.value, ast.Name
                        ) and target.value.id == "self":
                            has_self_assignment = True
                            instance_var_count += 1

        if instance_var_count >= 3:
            return KCAssessment(
                KC_C11, True,
                f"Multiple instance variables defined ({instance_var_count})",
                1.0
            )
        elif has_self_assignment:
            return KCAssessment(
                KC_C11, True,
                f"Some instance variables ({instance_var_count})", 0.6
            )
        else:
            return KCAssessment(
                KC_C11, False, "No instance variables (self.x) found", 0.9
            )

    def _assess_method_definition(
        self, code: str, ast_tree: ast.AST
    ) -> KCAssessment:
        """KC_C12: Class methods beyond __init__."""
        method_count = 0
        method_names = []

        for node in ast.walk(ast_tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        if item.name != "__init__":
                            method_count += 1
                            method_names.append(item.name)

        if method_count >= 3:
            return KCAssessment(
                KC_C12, True,
                f"Multiple methods defined: {', '.join(method_names[:3])}", 1.0
            )
        elif method_count > 0:
            return KCAssessment(
                KC_C12, True, f"Method(s) defined: {', '.join(method_names)}",
                0.7
            )
        else:
            return KCAssessment(
                KC_C12, False, "No methods defined beyond __init__", 0.9
            )

    # ========== Advanced Physics KC Assessment Methods (P9-P11) ==========

    def _assess_numerical_integration(
        self, code: str, ast_tree: ast.AST
    ) -> KCAssessment:
        """KC_P9: Numerical integration (updating position/velocity with dt)."""
        # Look for velocity/position updates with time step
        has_dt = (
            "dt" in code or "time_step" in code or "timestep" in code
            or "delta_t" in code or "time" in code.lower()
        )
        has_velocity_update = "+=" in code and (
            "v" in code or "velocity" in code or "vel" in code
        )
        has_position_update = "+=" in code and (
            "x" in code or "y" in code or "position" in code or "pos" in code
        )

        if has_dt and has_velocity_update and has_position_update:
            return KCAssessment(
                KC_P9, True,
                "Numerical integration: position and velocity updated with dt",
                0.9
            )
        elif has_dt and (has_velocity_update or has_position_update):
            return KCAssessment(
                KC_P9, True, "Partial numerical integration (dt used)", 0.6
            )
        else:
            return KCAssessment(
                KC_P9, False, "No numerical integration pattern found", 0.8
            )

    def _assess_force_acceleration(
        self, code: str, ast_tree: ast.AST
    ) -> KCAssessment:
        """KC_P10: Force and acceleration calculation (F = ma, a = F/m)."""
        has_force = "force" in code.lower() or "f_" in code.lower(
        ) or "fx" in code.lower() or "fy" in code.lower()
        has_acceleration = "accel" in code.lower() or "ax" in code.lower(
        ) or "ay" in code.lower()
        has_mass = "mass" in code.lower() or "m" in code and "self.m" in code

        # Look for division by mass (F/m) or multiplication (m*a)
        has_division = "/" in code and has_mass

        if has_force and has_acceleration and has_division:
            return KCAssessment(
                KC_P10, True, "Force and acceleration relationship (F=ma)", 0.9
            )
        elif has_acceleration and has_mass and has_division:
            return KCAssessment(
                KC_P10, True, "Acceleration calculation present", 0.7
            )
        else:
            return KCAssessment(
                KC_P10, False, "No force/acceleration calculation", 0.8
            )

    def _assess_kinetic_energy(
        self, code: str, ast_tree: ast.AST
    ) -> KCAssessment:
        """KC_P11: Kinetic energy calculation (KE = 0.5 * m * v^2)."""
        # Remove comments to avoid false positives from comment text
        code_no_comments = "\n".join(
            line.split("#")[0] for line in code.split("\n")
        )

        has_energy = (
            "energy" in code.lower() or "ke" in code.lower()
            or "kinetic" in code.lower()
        )
        # Check for 0.5 or 1/2 factor in various formats
        has_half = (
            "0.5" in code or "0.50" in code or ".5" in code or ".50" in code
            or "1/2" in code or "1/ 2" in code or "1 /2" in code
            or "1 / 2" in code
        )
        has_mass = "mass" in code.lower() or "m" in code
        # Check for velocity squared: v**2 or velocity * velocity
        code_lower = code.lower()
        has_velocity_squared = (
            "**2" in code
            or (code_lower.count("velocity") >= 2 and "*" in code)
            or ("v**2" in code_lower) or ("v **2" in code_lower)
            or ("vel_x**2" in code_lower) or ("vel_y**2" in code_lower)
        )

        # Must have BOTH half factor AND velocity squared for correct formula
        if has_energy and has_half and has_mass and has_velocity_squared:
            return KCAssessment(
                KC_P11, True, "Kinetic energy formula (0.5*m*v²)", 1.0
            )
        elif has_energy and has_mass and has_velocity_squared and not has_half:
            # Has energy calculation but missing 0.5 factor
            return KCAssessment(
                KC_P11, False, "Energy calculation missing 0.5 factor", 0.8
            )
        elif has_energy and has_mass:
            return KCAssessment(
                KC_P11, False, "Energy calculation incomplete", 0.7
            )
        else:
            return KCAssessment(
                KC_P11, False, "No kinetic energy calculation", 0.9
            )

    # ========== Additional Physics KC Assessment Methods (P12-P16) ==========


    def _assess_collision_reflection(
        self, code: str, ast_tree: ast.AST
    ) -> KCAssessment:
        """KC_P14: Velocity reversal on collision/bounce."""
        code_lower = code.lower()
        # Look for velocity reversal patterns
        has_velocity_reversal = (
            "-self.velocity" in code or "self.velocity = -" in code
            or "-restitution" in code_lower or "restitution *" in code_lower
            or "-self.v" in code or "self.v = -" in code
            or "= -" in code and ("velocity" in code_lower or " v " in code)
        )
        has_bounce_logic = (
            "bounce" in code_lower or "collision" in code_lower
            or "hit" in code_lower or "<= 0" in code or "< 0" in code
        )
        has_restitution = "restitution" in code_lower or "coefficient" in code_lower

        if has_velocity_reversal and has_bounce_logic:
            return KCAssessment(
                KC_P14, True, "Velocity reversal on collision", 1.0
            )
        elif has_velocity_reversal and has_restitution:
            return KCAssessment(
                KC_P14, True, "Bounce with restitution coefficient", 0.9
            )
        elif has_velocity_reversal:
            return KCAssessment(
                KC_P14, True, "Velocity reversal present (partial)", 0.6
            )
        else:
            return KCAssessment(
                KC_P14, False, "No collision/velocity reversal found", 0.8
            )


    def _assess_friction(
        self, code: str, ast_tree: ast.AST
    ) -> KCAssessment:
        """KC_P16: Friction force calculation."""
        code_lower = code.lower()
        # Look for friction patterns
        has_friction_var = (
            "friction" in code_lower or "mu" in code_lower
            or "μ" in code or "coefficient" in code_lower
        )
        has_friction_force = (
            "f_friction" in code_lower or "friction_force" in code_lower
            or "a_friction" in code_lower
        )
        # Check for cos pattern (for inclined plane: μ * m * g * cos(θ))
        has_cos_pattern = "cos" in code_lower and has_friction_var

        if has_friction_force and has_friction_var:
            return KCAssessment(
                KC_P16, True, "Friction force calculation present", 1.0
            )
        elif has_friction_var and has_cos_pattern:
            return KCAssessment(
                KC_P16, True, "Friction with angle consideration", 0.9
            )
        elif has_friction_var:
            return KCAssessment(
                KC_P16, True, "Friction coefficient used (partial)", 0.6
            )
        else:
            return KCAssessment(
                KC_P16, False, "No friction calculation found", 0.8
            )

    # ========== Gradient Descent KC Assessment Methods (C13-C16, M1-M5) ==========

    def _assess_loop_construct(
        self, code: str, ast_tree: ast.AST
    ) -> KCAssessment:
        """KC_C13: Loop construct (for/while)."""
        has_for = False
        has_while = False

        for node in ast.walk(ast_tree):
            if isinstance(node, ast.For):
                has_for = True
            elif isinstance(node, ast.While):
                has_while = True

        if has_while:
            return KCAssessment(
                KC_C13, True, "While loop present (common for iterative algorithms)", 1.0
            )
        elif has_for:
            return KCAssessment(
                KC_C13, True, "For loop present", 0.9
            )
        else:
            return KCAssessment(KC_C13, False, "No loop construct found", 1.0)

    def _assess_conditional_logic(
        self, code: str, ast_tree: ast.AST
    ) -> KCAssessment:
        """KC_C14: Conditional logic (if statements)."""
        has_if = False
        has_comparison = False

        for node in ast.walk(ast_tree):
            if isinstance(node, ast.If):
                has_if = True
            if isinstance(node, ast.Compare):
                has_comparison = True

        if has_if and has_comparison:
            return KCAssessment(
                KC_C14, True, "Conditional with comparison present", 1.0
            )
        elif has_if:
            return KCAssessment(KC_C14, True, "If statement present", 0.8)
        elif has_comparison:
            return KCAssessment(KC_C14, True, "Comparison present (partial)", 0.5)
        else:
            return KCAssessment(KC_C14, False, "No conditional logic found", 1.0)

    def _assess_variable_update(
        self, code: str, ast_tree: ast.AST
    ) -> KCAssessment:
        """KC_C15: Variable update (assignment with +=, -=, or x = x op val)."""
        has_aug_assign = False
        has_self_update = False

        for node in ast.walk(ast_tree):
            if isinstance(node, ast.AugAssign):
                has_aug_assign = True
            # Check for x = x - something pattern
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and isinstance(node.value, ast.BinOp):
                        if isinstance(node.value.left, ast.Name):
                            if target.id == node.value.left.id:
                                has_self_update = True

        if has_aug_assign:
            return KCAssessment(
                KC_C15, True, "Augmented assignment (+=, -=, etc.)", 1.0
            )
        elif has_self_update:
            return KCAssessment(
                KC_C15, True, "Variable self-update pattern (x = x op val)", 0.9
            )
        else:
            return KCAssessment(KC_C15, False, "No variable update pattern found", 0.9)

    def _assess_function_call(
        self, code: str, ast_tree: ast.AST
    ) -> KCAssessment:
        """KC_C16: Function call (calling other functions)."""
        function_calls = []

        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    function_calls.append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    function_calls.append(node.func.attr)

        # Filter out built-ins we don't care about
        meaningful_calls = [c for c in function_calls if c not in ['print', 'range', 'len', 'abs']]

        if len(meaningful_calls) >= 2:
            return KCAssessment(
                KC_C16, True, f"Multiple function calls: {', '.join(meaningful_calls[:3])}", 1.0
            )
        elif meaningful_calls:
            return KCAssessment(
                KC_C16, True, f"Function call: {meaningful_calls[0]}", 0.8
            )
        elif 'abs' in function_calls:
            return KCAssessment(
                KC_C16, True, "abs() used (for convergence check)", 0.7
            )
        else:
            return KCAssessment(KC_C16, False, "No meaningful function calls", 0.9)

    def _assess_derivative_concept(
        self, code: str, ast_tree: ast.AST
    ) -> KCAssessment:
        """KC_M1: Derivative/gradient concept (understanding rate of change)."""
        code_lower = code.lower()
        has_grad = "grad" in code_lower or "derivative" in code_lower
        has_delta = "delta" in code_lower or "dx" in code_lower or "h" in code

        # Look for gradient calculation pattern: f(x+d) - f(x-d) or similar
        has_difference = ("+delta" in code_lower or "-delta" in code_lower or
                         "+ delta" in code_lower or "- delta" in code_lower or
                         "+h" in code or "-h" in code)

        if has_grad and has_difference:
            return KCAssessment(
                KC_M1, True, "Gradient/derivative with difference calculation", 1.0
            )
        elif has_grad:
            return KCAssessment(KC_M1, True, "Gradient function defined", 0.8)
        elif has_delta and has_difference:
            return KCAssessment(KC_M1, True, "Derivative pattern (delta usage)", 0.7)
        else:
            return KCAssessment(KC_M1, False, "No derivative concept found", 0.9)

    def _assess_numerical_differentiation(
        self, code: str, ast_tree: ast.AST
    ) -> KCAssessment:
        """KC_M2: Numerical differentiation (f(x+h) - f(x-h)) / (2h)."""
        code_lower = code.lower()

        # Central difference: (f(x+h) - f(x-h)) / (2*h)
        has_central = ("2 *" in code or "2*" in code or "/ 2" in code) and "delta" in code_lower
        # Forward difference: (f(x+h) - f(x)) / h
        has_forward = "delta" in code_lower and "/" in code

        # Look for division pattern
        has_division = False
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                has_division = True

        if has_central and has_division:
            return KCAssessment(
                KC_M2, True, "Central difference formula", 1.0
            )
        elif has_forward and has_division:
            return KCAssessment(
                KC_M2, True, "Forward/backward difference formula", 0.8
            )
        elif has_division and "delta" in code_lower:
            return KCAssessment(
                KC_M2, True, "Numerical differentiation pattern", 0.7
            )
        else:
            return KCAssessment(KC_M2, False, "No numerical differentiation found", 0.9)

    def _assess_iterative_algorithm(
        self, code: str, ast_tree: ast.AST
    ) -> KCAssessment:
        """KC_M3: Iterative algorithm (repeated updates toward solution)."""
        has_loop = False
        has_update = False
        has_iteration_limit = "max" in code.lower() or "iteration" in code.lower() or "step" in code.lower()

        for node in ast.walk(ast_tree):
            if isinstance(node, (ast.For, ast.While)):
                has_loop = True
            if isinstance(node, ast.AugAssign):
                has_update = True

        if has_loop and has_update and has_iteration_limit:
            return KCAssessment(
                KC_M3, True, "Iterative algorithm with update and limit", 1.0
            )
        elif has_loop and has_update:
            return KCAssessment(KC_M3, True, "Iterative updates in loop", 0.8)
        elif has_loop:
            return KCAssessment(KC_M3, True, "Loop structure (partial)", 0.5)
        else:
            return KCAssessment(KC_M3, False, "No iterative pattern found", 0.9)

    def _assess_convergence_criteria(
        self, code: str, ast_tree: ast.AST
    ) -> KCAssessment:
        """KC_M4: Convergence criteria (stopping when gradient is small)."""
        code_lower = code.lower()
        has_abs = "abs(" in code
        has_threshold = any(t in code_lower for t in ["0.001", "0.01", "1e-", "epsilon", "tol", "threshold"])
        has_break = "break" in code or "return" in code

        # Look for comparison with small number
        has_small_comparison = False
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Compare):
                for comp in node.comparators:
                    if isinstance(comp, ast.Constant):
                        if isinstance(comp.value, (int, float)) and comp.value < 1:
                            has_small_comparison = True

        if has_abs and has_small_comparison and has_break:
            return KCAssessment(
                KC_M4, True, "Convergence check with abs() and threshold", 1.0
            )
        elif has_small_comparison and has_break:
            return KCAssessment(KC_M4, True, "Convergence check with threshold", 0.8)
        elif has_threshold:
            return KCAssessment(KC_M4, True, "Threshold defined (partial)", 0.5)
        else:
            return KCAssessment(KC_M4, False, "No convergence criteria found", 0.9)

    def _assess_learning_rate(
        self, code: str, ast_tree: ast.AST
    ) -> KCAssessment:
        """KC_M5: Learning rate (step size for gradient descent)."""
        code_lower = code.lower()
        has_eta = "eta" in code_lower
        has_lr = "lr" in code_lower or "learning" in code_lower or "rate" in code_lower
        has_step = "step" in code_lower and "size" in code_lower
        has_alpha = "alpha" in code_lower

        # Look for multiplication with gradient: x - eta * grad or x -= eta * grad
        has_grad_mult = "*" in code and ("grad" in code_lower or "f_grad" in code_lower)

        if (has_eta or has_lr or has_alpha) and has_grad_mult:
            return KCAssessment(
                KC_M5, True, "Learning rate used with gradient", 1.0
            )
        elif has_eta or has_lr or has_alpha:
            return KCAssessment(KC_M5, True, "Learning rate parameter present", 0.7)
        elif has_grad_mult:
            return KCAssessment(KC_M5, True, "Gradient multiplication (implicit rate)", 0.6)
        else:
            return KCAssessment(KC_M5, False, "No learning rate found", 0.9)

    def _all_kcs_failed(
        self, required_kcs: List[str], reason: str
    ) -> CodeAssessment:
        """Return assessment where all KCs failed."""
        kc_assessments = {
            kc: KCAssessment(kc, False, reason, 1.0)
            for kc in required_kcs
        }
        return CodeAssessment(
            kc_assessments=kc_assessments,
            overall_success=False,
            execution_error=reason,
            ast_tree=None
        )


def demo_aao():
    """Demonstrate the Assessment Oracle."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # Example code with import but other errors
    code = """
import math

def projectile_range(initial_velocity, angle_degrees, gravity):
    theta_rad = math.radians(angle_degrees)
    v_x = initial_velocity * math.sin(theta_rad)  # WRONG: should be cos
    v_y = initial_velocity * math.cos(theta_rad)  # WRONG: should be sin
    return v_x  # WRONG: incomplete
"""

    execution_result = {
        "success": False,
        "error": "Wrong output: expected 40.82, got 14.14"
    }

    required_kcs = [KC_C1, KC_C2, KC_P1, KC_P2, KC_P5]

    aao = AutomatedAssessmentOracle()
    assessment = aao.assess_code(code, execution_result, required_kcs)

    console.print("\n[bold cyan]Assessment Oracle Demo[/bold cyan]\n")
    console.print(
        f"[bold]Overall Success:[/bold] {assessment.overall_success}"
    )
    console.print(
        f"[bold]Execution Error:[/bold] {assessment.execution_error}\n"
    )

    table = Table(title="KC-Level Assessment")
    table.add_column("KC", style="cyan")
    table.add_column("Correct?", justify="center")
    table.add_column("Evidence", style="dim")
    table.add_column("Confidence", justify="right")

    for kc_id, kc_assess in assessment.kc_assessments.items():
        status = "[green]✓[/green]" if kc_assess.is_correct else "[red]✗[/red]"
        table.add_row(
            kc_id, status, kc_assess.evidence, f"{kc_assess.confidence:.2f}"
        )

    console.print(table)

    console.print("\n[bold green]Key Insight:[/bold green]")
    console.print("Even though the code FAILED overall,")
    console.print("the AAO correctly identifies that:")
    console.print("  • [green]KC_C2 (import math) was CORRECT[/green]")
    console.print("  • [green]KC_C1 (function def) was CORRECT[/green]")
    console.print(
        "  • [red]KC_P2 (trig) was INCORRECT (swapped sin/cos)[/red]"
    )
    console.print("\nThis allows BKT to give credit where due!")


if __name__ == "__main__":
    demo_aao()
