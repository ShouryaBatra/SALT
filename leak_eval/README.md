# SALT Leak Evaluation

This folder contains the main evaluation and analysis code for SALT.

## Key entrypoints

- `eval_cp.py` — Baseline evaluation. Generates outputs, collects (optional) multi-layer activations, and computes leakage/utility metrics. Supports Hugging Face and OpenRouter providers.
- `steered_eval_cp_resume.py` — Evaluation with steering vectors (single or multi-layer). Resume-safe JSON writing.
- `find_leak_layers.py` — Contrastive per-neuron analysis to rank layers by leakage association; can export normalized steering vectors.
- `prompts/cp_open_ended_chat/` — Prompt templates (e.g., `vanilla`, `cot_explicit_unk`, `reasoning_explicit_unk`).

Notes

- For GPT-as-judge scoring (`--gpt_eval`), set `OPENAI_API_KEY`.
- Some models do not support `system` role; code handles this automatically, you should take a look.
