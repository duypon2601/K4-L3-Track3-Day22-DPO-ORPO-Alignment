# Progress Log

## Task 1: Implement `my_dpo_loss` in NB0
- Implemented `my_dpo_loss` in `notebooks/00_dpo_loss_from_scratch.py` using `torch.nn.functional.logsigmoid` and sequence log-probability margins.
- Created `scripts/test_nb0.py` extracting `my_dpo_loss` via `ast` and validating numerical equivalence to `dpo_math.dpo_loss`, initialization loss at `log 2`, and removal of TODO placeholders.
- Regenerated Colab notebook bundles `colab/Lab22_DPO_T4.ipynb` and `colab/Lab22_DPO_BigGPU.ipynb` using `scripts/build_colab.py`.
- Verified that all 57 test cases in the test suite pass cleanly using `.venv/bin/pytest -q scripts/`.

## Task 2: Answer the likelihood-displacement question in NB0
- Added Vietnamese answer markdown cell for likelihood displacement in notebooks/00_dpo_loss_from_scratch.py section 5.
- Explained why DPO margin increases when rejected log-probability drops faster than chosen log-probability, citing toy scenario B (-3 chosen reward vs -5 rejected reward).
- Highlighted that only separate rewards/chosen tracking in NB3 reveals this displacement, and that RPO penalizes it by incorporating NLL(chosen).
- Added test_nb0_answers_displacement_question in scripts/test_nb0.py to validate cell presence, placement, word count, and keywords.
- Regenerated Colab notebook bundles (colab/Lab22_DPO_T4.ipynb and colab/Lab22_DPO_BigGPU.ipynb) via scripts/build_colab.py.

## Task 3: Commit the executed NB0 notebook with outputs
- Generated `notebooks/00_dpo_loss_from_scratch.ipynb` from `notebooks/00_dpo_loss_from_scratch.py` using jupytext.
- Executed the notebook in place with nbconvert to generate real cell execution counts and stream outputs.
- Added `test_nb0_ipynb_is_executed` in `scripts/test_nb0.py` to verify notebook execution, output correctness, stream messages, and source consistency.
- Verified that the complete test suite in `scripts/` passes cleanly.
