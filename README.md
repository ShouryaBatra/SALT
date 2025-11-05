# SALT: Steering Activations towards Leakage-free Thinking in Chain of Thought

### Overview

This is our research codebase for SALT. Includes evaluating and steering large language models to reduce privacy leakage in chain-of-thought (CoT) reasoning. It contains scripts to:

- run baseline and steered generations while capturing layer activations,
- compute privacy/utility metrics (with optional LLM-as-a-judge), and
- analyze which layers are most associated with leakage and save steering vectors.

If you use this repository in academic work, please cite the accompanying paper:

*Paper & citation not yet publicly available*
<!-- ```bibtex
@misc{tillman2025ppsv,
      title={{PPSV}: Using Steering Vectors to Mitigate Internal Reasoning Privacy Leakage in Large Language Models},
      author={Pierce Tillman and Samarth Gaggar and Shourya Batra and Shashank Kesineni and Kevin Zhu and Sunishchal Dev and Maheep Chaudhary},
      year={2025},
      eprint={TBD},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      note={Project AlgoVerse AI Research},
      url={https://github.com/AlgoVerseAI/PPSV}, 
}
``` -->

### Repository structure

- `leak_eval/`
  - `eval_cp.py`: Baseline evaluation with optional multi-layer activation capture and optional GPT eval.
  - `steered_eval_cp_resume.py`: Run evaluation with steering vectors applied (single- or multi-layer), resume-safe.
  - `find_leak_layers.py`: Contrastive per-neuron analysis to rank layers by leakage association and optionally export steering vectors.
  - `cp_eval_utils.py`: Metric helpers (utility/leakage), GPT evaluation, cost estimation.
  - `generate_utils.py`: Provider/model helpers and generation utilities.
  - `prompts/cp_open_ended_chat/`: Prompt templates (`vanilla.txt`, `cot_explicit_unk.txt`, `reasoning_explicit_unk.txt`, `situation_template.txt`).
  - `scripts/`: Small utilities (precompute vectors, merge/split datasets, run GPT eval on results, count leaks, sweeps).
- `notebooks/`: Prototyping and blueprint pipeline (`instruction.ipynb`).
- `results/`: Example outputs, sweeps and paper figures (reference only; you can regenerate locally).

### Requirements

- Python 3.10+
- A GPU is recommended for HF models; CPU works for small tests.
- Optional APIs:
  - OpenAI (for LLM-as-a-judge): `OPENAI_API_KEY`
  - OpenRouter (if using `--model_provider openrouter`): `OPENROUTER_API_KEY`
  - Hugging Face token as needed for gated models (`HF_TOKEN` or `HUGGINGFACEHUB_API_TOKEN`).

### Quickstart

See `notebooks/instruction.ipynb` for a quick start.

### Other

All of the other code in here is code we used for the paper. Data from our paper is also provided in `/results/`.

- Reference results are under `results/final_results/` and layer-analysis CSVs under `results/leak_layer_csvs/`.
- Plot scripts: `results/final_results/results_graph/graph.py` and `results/leak_layer_csvs/graphs/graph.py`.

Notebooks

- `notebooks/instruction.ipynb` is a high-level blueprint for the end-to-end pipeline.

Environment notes

- Some models do not support a `system` role; the code handles this automatically (e.g., Gemma) by stripping the system role in chat templates.
- Batch size is auto-tuned from GPU VRAM if not provided.

Results schema (abridged)

- `eval_cp.py` writes a JSON with at least `data` (list of examples with outputs/metrics) and `summary` (aggregate metrics, averages). When GPT eval is run, the summary also includes `gpt_utility_score`, `gpt_pii_leakage`, and `total_gpt_api_cost`.

### Acknowledgements

This work builds upon *Leaky Thoughts: Large Reasoning Models Are Not Private Thinkers* by Tommaso Green, Martin Gubri, Haritz Puerto, Sangdoo Yun, and Seong Joon Oh 
([arXiv:2506.15674](https://arxiv.org/abs/2506.15674)), and the accompanying [AirGapAgent-R Dataset](https://huggingface.co/datasets/parameterlab/leaky_thoughts).
<!-- Please cite SALT (add BibTeX/DOI/arXiv once available), and possibly the Leaky Thoughts paper we build off of. -->

### License

Contact & contributions
Issues and PRs are welcome. For substantive contributions, please open an issue first to discuss scope. If you build on SALT, let us know—happy to link community extensions here.

### Developers
