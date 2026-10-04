# BEAGLE: Behavior-Enforced Agent for Grounded Learner Emulation

Code for the NeurIPS 2026 paper *BEAGLE: Behavior-Enforced Agent for Grounded Learner Emulation* (Hanchen David Wang, Clayton Cohn, Zifan Xu, Siyuan Guo, Gautam Biswas, Meiyi Ma).

Paper: [arXiv:2602.13280](https://arxiv.org/abs/2602.13280)

BEAGLE is a neuro-symbolic simulator of novice students in computational STEM (C-STEM) tasks, where students learn disciplinary content by writing and testing Python programs. It counters the *competency bias* of LLMs (their tendency to solve problems efficiently and correctly even when asked to act as a novice) with three architectural constraints:

1. **Semi-Markov control.** A model fit on real student logs decides which metacognitive and cognitive behavior the agent is in and for how long.
2. **Bayesian Knowledge Tracing with Explicit Flaw Injection (EFI).** BKT limits which knowledge components (KCs) the agent may use; EFI blocks selected KCs to produce stable knowledge gaps.
3. **Strategist/Executor split.** One LLM call plans, a second implements the plan as given, so flawed plans are not silently corrected.

## Repository layout

```
beagle/
  data_generation/
    studentv2/        BEAGLE agent: semi-Markov controller, BKT/EFI, Strategist/Executor graph, prompts
    baselines/        Vanilla, CoT, Few-Shot (and +M variants), LLM-SS, SimStudent, CoderAgent
    ide_oracle/       Runs student code against unit tests and returns (filtered) feedback
    classroom/        Per-KC assessment rules used for BKT credit assignment
    problems/         The four tasks: particle_simulator, bouncing_ball, inclined_plane, gradient_descent
  tutor/              Optional tutor strategies used by the assistance interrupt
  utils/              LLM client (pydantic-ai) and display helpers
evaluation/           Batch runners, LLM judge, and metric/table scripts
scripts/              Cross-task and cross-backbone experiment drivers
tests/                Unit tests (fully mocked, no API calls)
main.py               Run a single simulated student
```

## Installation

Python 3.12 is recommended.

```bash
pip install -r requirements.txt
cp .env.example .env   # then add the API key(s) for the provider you use
```

Models are addressed with pydantic-ai identifiers such as `google-gla:gemini-2.0-flash`, `openai:gpt-4o-mini`, or `anthropic:claude-haiku-4-5`.

## Quick start

Run one simulated student:

```bash
python main.py --problem particle_simulator --performance low --model google-gla:gemini-2.0-flash --max-steps 30
```

Run the tests:

```bash
python -m pytest tests -q
```

## Reproducing the experiments

All commands are run from the repository root. Each run writes one JSON file per trajectory under `--output-dir`.

**BEAGLE (main configuration: N=50, T=30).**

```bash
python evaluation/run_batch_simulations.py \
  --problem particle_simulator --n-per-level 25 --max-steps 30 \
  --model google-gla:gemini-2.0-flash --output-dir results/beagle
```

**Baselines.** `--baseline` is one of `vanilla`, `cot`, `fewshot`, `llmss`, `simstudent`, `coderagent`; add `--enable-metacog` for the +M variants.

```bash
python evaluation/run_baseline_batch.py --baseline cot --enable-metacog \
  --problem particle_simulator --n-per-level 25 --max-steps 30 \
  --model google-gla:gemini-2.0-flash --output-dir results/cot_metacog
```

**Ablations** (flags of `run_batch_simulations.py`; `evaluation/run_all_ablations.sh` runs them all):

| Ablation | Flag(s) |
|---|---|
| No BKT | `--no-bkt` |
| No semi-Markov | `--disable-markov` |
| No Interrupts | `--disable-assistance --disable-offtopic` |
| No Executor memory | `--disable-memory-executor` |
| No Strategist memory | `--disable-memory-strategist` |
| Combined Agent | `--enable-merged-pipeline` |

**LLM-as-judge.**

```bash
python evaluation/evaluator.py --folder results/beagle --llm-judge --llm-model google-gla:gemini-2.5-pro
```

**Metrics and tables.** `evaluation/compute_comprehensive_metrics.py` computes the behavioral and epistemic metrics (D_KL, D_debug, nonlinearity, error recurrence, reaction lag) and `evaluation/compute_ablation_table.py` builds the ablation table. Both scripts map table rows to result folder names at the top of the file; edit those mappings to point at your own output directories (or set `BEAGLE_RESULTS_DIR`).

**Other tasks and backbones.** `scripts/run_new_tasks.sh` runs BEAGLE and the baselines on additional tasks; `scripts/run_cross_model.sh` runs BEAGLE across LLM backbones.

## Adding a task

A task is a folder under `beagle/data_generation/problems/<task_id>/` with exactly three files:

- `description.json`: assignment text, entry-point name, and the list of `required_kcs`
- `reference_solution.py`
- `test_<task_id>.py`: pytest unit tests

No prompt or parameter changes are needed for a task that uses existing KCs. A new KC needs a registry entry in `studentv2/bkt/bkt.py` and a detection rule in `classroom/assessment_oracle.py`.

## Data

This repository does not include human-subject data. The real-student corpora used in the paper (the calibration corpus, the block-based and Python pilot corpora, and the Turing-test responses) were collected under IRB protocols and are not redistributed here.

- `studentv2/semi_markov_model.joblib` contains only the fitted aggregate parameters of the semi-Markov model (transition probabilities, Gamma duration parameters, and emission probabilities). The scripts in `studentv2/training/` show how it was fit; they need the calibration corpus, which is available from its authors on request.
- For D_KL and D_debug, the evaluation scripts fall back to the pinned aggregate reference distribution reported in the paper when the session-level reference files are absent.
- The Bielefeld Python programming dataset used for the Gradient Descent task is available from its authors (Paassen et al.).

## Telemetry

Logfire tracing is off by default. To enable it, set `BEAGLE_ENABLE_LOGFIRE=1` and `LOGFIRE_TOKEN`.

## Citation

```bibtex
@inproceedings{wang2026beagle,
  title     = {BEAGLE: Behavior-Enforced Agent for Grounded Learner Emulation},
  author    = {Wang, Hanchen David and Cohn, Clayton and Xu, Zifan and Guo, Siyuan and Biswas, Gautam and Ma, Meiyi},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  eprint    = {2602.13280},
  archivePrefix = {arXiv}
}
```

## Acknowledgments

This work was supported by the Institute of Education Sciences (IES), U.S. Department of Education, under Award Number R305C240010. The opinions expressed are those of the authors and do not represent the views of the Institute or the U.S. Department of Education.

## License

MIT. See [LICENSE](LICENSE).
