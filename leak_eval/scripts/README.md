# SALT Scripts (Utilities)

Small helper scripts used during experiments.

## Scripts

- `run_gpt_eval_on_results.py` — Run LLM-as-a-judge on an existing results JSON and append summary (`gpt_utility_score`, `gpt_pii_leakage`, cost).
- `val_sweep.py` — Sweep over layers/strengths for multi-layer steering validation.
- `precompute_last_input_sidecars.py` — Precompute 1D last-input vectors as sidecar `.npy` for each activation `.npz`.
- `build_last_input_vector.py` — Build steering vector from last-layer last-input-token activations.
- `count_gpt_leaks.py` — Count GPT-evaluated PII leakages in a results JSON.
- `merge_results_by_example_id.py` — Merge multiple results files by example id.
- `split_dataset.py` — Deterministic split of a dataset JSON into train/val/test.

Notes

- These scripts assume outputs from `eval_cp.py`/`steered_eval_cp_resume.py`.
- Some scripts are optional convenience tools; adjust paths to your setup.
