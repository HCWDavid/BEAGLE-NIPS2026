"""
Block-Based Prompt Assembler for BEAGLE Student Simulation

================================================================================
ARCHITECTURE OVERVIEW
================================================================================

This module implements composable prompts based on empirical LAK24 analysis.
Prompts are assembled from text blocks stored in the `blocks/` directory.

The prompts follow a SYSTEM + USER structure:
    SYSTEM PROMPT = Static Core + Dynamic Mandate
    USER PROMPT   = Directive + Task + Context

STRATEGIST PROMPT:
    SYSTEM: Persona(ρ) + Rules + Mandate(M_t, C_t)
    USER:   Task_Strat + Context_Strat

EXECUTOR PROMPT:
    SYSTEM: Persona(ρ) + Rules + Mandate(M_t, C_t)
    USER:   Directive(M_t) + Task_Exec(C_t) + Context_Exec

================================================================================
BLOCK DIRECTORY STRUCTURE
================================================================================

blocks/
├── system/
│   ├── static/
│   │   ├── base.txt            # "You are a middle school student"
│   │   ├── low_performer.txt   # Reactive, confused style
│   │   ├── high_performer.txt  # Systematic, analytical style
│   │   └── rules.txt           # No psychic debugging, memory rule, grounding
│   │
│   └── dynamic/
│       └── mandate/            # Behavioral frames (12 files)
│           ├── planning_constructing.txt
│           ├── planning_debugging.txt
│           ├── enacting_debugging.txt    # EPISTEMIC FILTER
│           ├── monitoring_debugging.txt  # BKT UPDATE
│           └── ...
│
└── user/
    ├── directive/              # LAK24-derived behavioral instructions
    │   ├── directive_planning.txt   # "Strategist - Think before acting"
    │   ├── directive_enacting.txt   # "Actor - Verbalize then do"
    │   ├── directive_monitoring.txt # "Spotter - Check values"
    │   └── directive_reflecting.txt # "Critic - Understand why"
    │
    ├── task/                   # Output format instructions
    │   ├── strategist.txt
    │   ├── executor_constructing.txt
    │   ├── executor_debugging.txt
    │   └── executor_assessing.txt
    │
    └── context/                # Dynamic context templates
        ├── strategist.txt      # {HISTORY} placeholder
        └── executor.txt        # {SCREEN} placeholder

================================================================================
EXAMPLE: Strategist SYSTEM + USER prompts for Planning+Debugging
================================================================================

--- SYSTEM PROMPT (assembled from 4 blocks) ---

┌─ FROM: system/static/base.txt ─────────────────────────────────────────────┐
│ [PERSONA: BASE]                                                            │
│ You are a middle school student learning to code in Python...              │
└────────────────────────────────────────────────────────────────────────────┘

┌─ FROM: system/static/low_performer.txt ────────────────────────────────────┐
│ [STYLE MODIFIER: STRUGGLING]                                               │
│ You are feeling uncertain and reactive...                                  │
└────────────────────────────────────────────────────────────────────────────┘

┌─ FROM: system/static/rules.txt ────────────────────────────────────────────┐
│ [UNIVERSAL RULES]                                                          │
│ *** NO PSYCHIC DEBUGGING ***                                               │
│ *** MEMORY RULE ***                                                        │
│ *** GROUNDING RULE ***                                                     │
└────────────────────────────────────────────────────────────────────────────┘

┌─ FROM: system/dynamic/mandate/planning_debugging.txt ──────────────────────┐
│ [MANDATE: DIAGNOSTIC PLANNING]                                             │
│ Current State: PLANNING (Deciding what to do)                              │
│ Context: DEBUGGING (Fixing an error)                                       │
│ EMPIRICAL LANGUAGE PATTERNS (from LAK24): ...                              │
└────────────────────────────────────────────────────────────────────────────┘

--- USER PROMPT (assembled from 2 blocks) ---

┌─ FROM: user/task/strategist.txt ───────────────────────────────────────────┐
│ [TASK: STRATEGIST]                                                         │
│ Generate: GOAL, MINDSET, DIRECTIVE                                         │
└────────────────────────────────────────────────────────────────────────────┘

┌─ FROM: user/context/strategist.txt (with {HISTORY} replaced) ──────────────┐
│ [CONTEXT]                                                                  │
│ Previous code: ...                                                         │
│ Output: NameError on line 5                                                │
└────────────────────────────────────────────────────────────────────────────┘

================================================================================
EXAMPLE: Executor SYSTEM + USER prompts for Enacting+Debugging  
================================================================================

--- USER PROMPT (assembled from 3 blocks) ---

┌─ FROM: user/directive/directive_enacting.txt ──────────────────────────────┐
│ [DIRECTIVE: ENACTING PROFILE]                                              │
│ CONTEXT: You are the Actor (Hands on Keyboard).                            │
│ 1. VERBALIZE INTENT: State your micro-goal before acting                   │
│ 2. REACTIVE EXECUTION: Just do it, don't command others                    │
│ 3. TRIAL-AND-ERROR: Quick iterations, not deep analysis                    │
└────────────────────────────────────────────────────────────────────────────┘

┌─ FROM: user/task/executor_debugging.txt ───────────────────────────────────┐
│ [TASK: EXECUTOR - DEBUGGING]                                               │
│ *** CRITICAL: Reference the specific error type you see ***                │
└────────────────────────────────────────────────────────────────────────────┘

┌─ FROM: user/context/executor.txt (with {SCREEN} replaced) ─────────────────┐
│ [CONTEXT]                                                                  │
│ Code: x = 10                                                               │
│ Output: NameError: 'y' is not defined                                      │
└────────────────────────────────────────────────────────────────────────────┘

================================================================================
"""

import os
from pathlib import Path
from typing import Optional, Tuple


class PromptAssembler:
    """
    Assembles SYSTEM and USER prompts from composable text blocks.
    
    Reads blocks from the filesystem and caches them for efficiency.
    
    NOTE: This class handles ρ_persona (language/style), NOT ρ_behavior.
    - ρ_persona: Affects prompts (vocabulary, tone, confusion markers)
    - ρ_behavior: Affects Markov model (transition probabilities) - handled elsewhere
    """
    
    def __init__(
        self, 
        persona_type: str = "low_performer",
        blocks_dir: Optional[Path] = None
    ):
        """
        Args:
            persona_type: ρ_persona - 'low_performer' or 'high_performer'
                          Controls language style, NOT behavioral transitions.
            blocks_dir: Path to blocks directory (default: same dir as this file)
        """
        if blocks_dir is None:
            blocks_dir = Path(__file__).parent / "blocks"
        
        self.blocks_dir = Path(blocks_dir)
        self.persona_type = persona_type  # ρ_persona (language/style)
        self._cache: dict[str, str] = {}
    
    def _read(self, *path_parts: str) -> str:
        """
        Reads and caches a text block.
        
        Args:
            *path_parts: Path components relative to blocks_dir
                         e.g., ("system", "static", "base") -> blocks/system/static/base.txt
            
        Returns:
            Block content as string
        """
        # Safety: lowercase all path parts to prevent casing bugs
        clean_parts = [p.lower().strip() for p in path_parts]
        key = "/".join(clean_parts)
        
        if key not in self._cache:
            path = self.blocks_dir.joinpath(*clean_parts).with_suffix(".txt")
            if not path.exists():
                return f"\n[MISSING BLOCK: {key}.txt]\n"
            with open(path, "r", encoding="utf-8") as f:
                self._cache[key] = f.read().strip()
        return self._cache[key]
    
    def _get_system_static(self) -> str:
        """
        Assembles static system components: Base Persona + Performer Modifier + Rules.
        
        Returns:
            Combined static system prompt text
        """
        base = self._read("system", "static", "base")
        modifier = self._read("system", "static", self.persona_type)
        rules = self._read("system", "static", "rules")
        return f"{base}\n\n{modifier}\n\n{rules}"
    
    def _get_mandate(self, meta_state: str, cog_state: str) -> str:
        """
        Gets the dynamic mandate for this (metacog, cognitive) pair.
        """
        mandate_file = f"{meta_state.lower()}_{cog_state.lower()}"
        return self._read("system", "dynamic", "mandate", mandate_file)
    
    def assemble_strategist_prompts(
        self,
        meta_state: str,
        cog_state: str,
        context_str: str
    ) -> Tuple[str, str]:
        """
        Assemble SYSTEM and USER prompts for Strategist.
        
        Args:
            meta_state: Metacognitive state (Planning, Enacting, Monitoring, Reflecting)
            cog_state: Cognitive state (Constructing, Debugging, Assessing)
            context_str: Dynamic context (history, buffers, etc.)
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        # SYSTEM = Static + Strategist Rules + Mandate
        static = self._get_system_static()
        strategist_rules = self._read("system", "static", "strategist_rules")
        mandate = self._get_mandate(meta_state, cog_state)
        system_prompt = f"{static}\n\n{strategist_rules}\n\n{mandate}"
        
        # USER = Task + Context
        task = self._read("user", "task", "strategist")
        ctx_template = self._read("user", "context", "strategist")
        context = ctx_template.replace("{HISTORY}", context_str)
        user_prompt = f"{task}\n\n{context}"
        
        return system_prompt, user_prompt
    
    def assemble_executor_prompts(
        self,
        meta_state: str,
        cog_state: str,
        strategy_output: dict,
        context_str: str
    ) -> Tuple[str, str]:
        """
        Assemble SYSTEM and USER prompts for Executor.
        
        Args:
            meta_state: Metacognitive state (Planning, Enacting, Monitoring, Reflecting)
            cog_state: Cognitive state (Constructing, Debugging, Assessing)
            strategy_output: Dict with keys 'goal', 'mindset', 'directive' from Strategist
                             These are (g_t, m_t, d_t) - the architectural directive
            context_str: Dynamic context (code, output, etc.)
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        # SYSTEM = Static + Executor Rules + Mandate
        static = self._get_system_static()
        executor_rules = self._read("system", "static", "executor_rules")
        mandate = self._get_mandate(meta_state, cog_state)
        system_prompt = f"{static}\n\n{executor_rules}\n\n{mandate}"
        
        # USER = Empirical Profile + Strategy Output + Task + Context
        
        # 1. Empirical Profile (static stage directions from LAK24)
        #    Controls linguistic style (e.g., "don't sound bossy")
        empirical_profile = self._read("user", "directive", f"directive_{meta_state.lower()}")
        
        # 2. Strategy Output (dynamic from Strategist)
        #    This is the architectural d_t that constrains the Executor
        goal = strategy_output.get('goal', '[No goal provided]')
        mindset = strategy_output.get('mindset', '[No mindset provided]')
        directive = strategy_output.get('directive', '[No directive provided]')
        
        current_strategy = (
            "=== CURRENT STRATEGY (from your planning) ===\n"
            f"GOAL (Long-term): {goal}\n"
            f"MINDSET (Your emotional state): {mindset}\n"
            f"DIRECTIVE (What to do NOW): {directive}\n"
            "=============================================="
        )
        
        # 3. Task (output format instructions)
        task = self._read("user", "task", f"executor_{cog_state.lower()}")
        
        # 4. Context (code, output, etc.)
        ctx_template = self._read("user", "context", "executor")
        context = ctx_template.replace("{SCREEN}", context_str)
        
        user_prompt = f"{empirical_profile}\n\n{current_strategy}\n\n{task}\n\n{context}"
        
        return system_prompt, user_prompt
    
    def clear_cache(self):
        """Clear the block cache (useful for hot-reloading during development)."""
        self._cache.clear()


# =============================================================================
# Standalone Test
# =============================================================================
if __name__ == "__main__":
    print("Testing PromptAssembler (System/User structure)...")
    
    assembler = PromptAssembler(persona_type="low_performer")
    
    # Test Strategist prompts
    print("\n" + "=" * 60)
    print("STRATEGIST: Planning + Debugging")
    print("=" * 60)
    sys_prompt, user_prompt = assembler.assemble_strategist_prompts(
        meta_state="Planning",
        cog_state="Debugging",
        context_str="Previous code: ...\nOutput: NameError on line 5"
    )
    print("\n--- SYSTEM PROMPT ---")
    print(sys_prompt[:600] + "..." if len(sys_prompt) > 600 else sys_prompt)
    print("\n--- USER PROMPT ---")
    print(user_prompt[:400] + "..." if len(user_prompt) > 400 else user_prompt)
    
    # Test Executor prompts (with strategy output from Strategist)
    print("\n" + "=" * 60)
    print("EXECUTOR: Enacting + Debugging (with strategy_output)")
    print("=" * 60)
    
    # Simulate what the Strategist would have produced
    strategy_output = {
        'goal': "Fix the variable errors",
        'mindset': "Frustrated but trying",
        'directive': "Just try changing the variable name"
    }
    
    sys_prompt, user_prompt = assembler.assemble_executor_prompts(
        meta_state="Enacting",
        cog_state="Debugging",
        strategy_output=strategy_output,
        context_str="Code: x = 10\nOutput: NameError: 'y' is not defined"
    )
    print("\n--- SYSTEM PROMPT ---")
    print(sys_prompt[:600] + "..." if len(sys_prompt) > 600 else sys_prompt)
    print("\n--- USER PROMPT ---")
    print(user_prompt[:800] + "..." if len(user_prompt) > 800 else user_prompt)
    
    print("\n✅ Assembler test complete!")
