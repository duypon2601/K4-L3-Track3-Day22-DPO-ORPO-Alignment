# Day 22 DPO/ORPO Alignment lab — complete the CPU-only part (NB0)

TEST_CMD: .venv/bin/pytest -q scripts/

## Overview
Student lab repo (Python 3.12, CPU-only `.venv` already created with pytest, torch, pandas,
jupytext, nbconvert, ipykernel). This machine is an Apple Silicon Mac with no CUDA GPU, so only
NB0 can be done here. NB1-NB4 (SFT, DPO training, judging) and `submission/REFLECTION.md` need
real numbers from a Colab GPU run and are OUT OF SCOPE: do not touch `submission/`, `adapters/`,
`data/`, `models/`, and never invent metrics, screenshots or result files.

Layout:
- `notebooks/*.py` — Jupytext `py:percent` sources (source of truth). Cells start with `# %%`
  (code) or `# %% [markdown]` (every markdown line is a `# ` comment). Text is Vietnamese.
- `lab22/` — shared library. `lab22/dpo_math.py` holds the reference losses
  (`dpo_loss(pc, pr, rc, rr, beta)` returns `(loss, chosen_reward, rejected_reward)`). Do not edit it.
- `scripts/test_*.py` — pytest suite (CPU). `scripts/build_colab.py` regenerates
  `colab/*.ipynb` from `notebooks/` + `lab22/`.
- `colab/*.ipynb` are GENERATED. `scripts/test_smoke.py::test_colab_bundles_are_valid_and_current`
  fails if they are stale, so after ANY edit to `notebooks/*.py` run
  `.venv/bin/python scripts/build_colab.py` and never edit `colab/` by hand.

Conventions: match the surrounding style (ruff line length 120, short Vietnamese comments in
notebooks, English in `scripts/`). Keep the existing 54 tests passing. Do not add dependencies.

Command restriction: you may only run the commands listed in `.autowf.env`
(`git` read-only, `.venv/bin/python`, `.venv/bin/pytest`, `.venv/bin/jupytext`,
`.venv/bin/jupyter`, `ls`, `mkdir`), one per call, with no `;`, `&&`, `|` or `$(...)`.
Do not use MCP tools, use CLI commands only. Do not commit; autowf commits.

## Task 1: Implement `my_dpo_loss` in NB0
Files: `notebooks/00_dpo_loss_from_scratch.py` (modify), `scripts/test_nb0.py` (create),
`colab/Lab22_DPO_T4.ipynb` and `colab/Lab22_DPO_BigGPU.ipynb` (regenerate only).

Steps:
- In `notebooks/00_dpo_loss_from_scratch.py` replace the `# TODO` body of `my_dpo_loss` with the
  DPO loss written with `torch.nn.functional.logsigmoid`:
  `-logsigmoid(beta * ((pc - rc) - (pr - rr))).mean()`. Keep the signature and docstring.
- Create `scripts/test_nb0.py`. It must not execute the whole notebook: read the notebook source,
  use `ast` to extract the `my_dpo_loss` function definition, `exec` only that function in a
  namespace that has `torch`, and test it. Use `pytest.importorskip("torch")`.
- Run `.venv/bin/python scripts/build_colab.py` to refresh the Colab bundles.

**Acceptance criteria:** `.venv/bin/pytest -q scripts/` exits 0, including new tests in
`scripts/test_nb0.py`:
- `test_my_dpo_loss_matches_reference`: equals `lab22.dpo_math.dpo_loss(...)[0]` (atol 1e-6) on the
  notebook's toy tensors and for `beta` in (0.05, 0.1, 0.5);
- `test_my_dpo_loss_is_log2_at_init`: policy == reference gives `log 2`;
- `test_my_dpo_loss_has_no_todo`: the function source no longer contains `TODO` or `return None`;
- `test_colab_bundles_are_valid_and_current` still passes.

## Task 2: Answer the likelihood-displacement question in NB0
Files: `notebooks/00_dpo_loss_from_scratch.py` (modify), `scripts/test_nb0.py` (modify),
`colab/*.ipynb` (regenerate only).

Steps:
- Right after the code cell of section "## 5. Likelihood displacement bằng số" (before the RPO
  markdown cell), add one `# %% [markdown]` cell that starts with `# **Trả lời (NB0):**` and
  answers in Vietnamese, in 80-150 words: why the margin can rise while the chosen log-prob falls.
  Required content: DPO's loss depends only on the difference
  `(log π(y_w) − log π_ref(y_w)) − (log π(y_l) − log π_ref(y_l))`, so it is satisfied when the
  rejected log-prob drops faster than the chosen one; cite scenario B from the cell above
  (chosen reward −3, rejected reward −5, margin +2, same loss as scenario A); say that only the
  separate `rewards/chosen` curve in NB3 reveals it and that RPO adds NLL(chosen) to penalise it.
- Use only numbers that come from the notebook's own toy scenarios. No claims about real training
  results.
- Add a test, then run `.venv/bin/python scripts/build_colab.py`.

**Acceptance criteria:** `.venv/bin/pytest -q scripts/` exits 0, including
`test_nb0_answers_displacement_question` in `scripts/test_nb0.py`: the notebook source has a
markdown cell containing `**Trả lời (NB0):**`, that cell has at least 80 words, mentions
`rejected` and `chosen`, and is located between the section 5 heading and the RPO markdown cell;
`test_colab_bundles_are_valid_and_current` still passes.

## Task 3: Commit the executed NB0 notebook with outputs
Files: `notebooks/00_dpo_loss_from_scratch.ipynb` (create, generated), `scripts/test_nb0.py`
(modify).

Steps:
- Generate: `.venv/bin/jupytext --to notebook --update notebooks/00_dpo_loss_from_scratch.py`
  (drop `--update` if the `.ipynb` does not exist yet and the command complains).
- Execute in place: `.venv/bin/jupyter nbconvert --to notebook --execute --inplace
  --ExecutePreprocessor.timeout=600 notebooks/00_dpo_loss_from_scratch.ipynb`.
- Do not hand-edit the `.ipynb` or paste outputs; outputs must come from the real execution. If
  execution fails, fix the notebook source (and rebuild the Colab bundles), not the outputs.
- Add a test that reads the `.ipynb` as JSON.

**Acceptance criteria:** `.venv/bin/pytest -q scripts/` exits 0, including
`test_nb0_ipynb_is_executed` in `scripts/test_nb0.py`: `notebooks/00_dpo_loss_from_scratch.ipynb`
exists, every code cell has a non-null `execution_count`, no output has `output_type == "error"`,
some stream output contains `✓ Khớp tham chiếu`, some contains `loss at init = 0.6931`, and the
`my_dpo_loss` cell source in the `.ipynb` equals the one in the `.py` source.
