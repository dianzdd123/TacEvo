# TacEvo

**Start here:** [Setup, data, run order and limitations](docs/REUSE.md) · [Contributing](CONTRIBUTING.md)

**Repository:** [dianzdd123/TacEvo](https://github.com/dianzdd123/TacEvo)

**MSEI Lab:** [Website](https://zhangrobotics.com/msei_website/) · [Website on GitHub](https://github.com/dianzdd123/msei_website)

**Upstream attribution:** Forked from [LannWei/TacEvo](https://github.com/LannWei/TacEvo). Original authorship, citations and notices are retained.
TacEvo studies LLM-guided neural architecture evolution for tactile perception. This repository contains the search pipeline, mutation prompts, training utilities and supplementary material.

**Release status: incomplete.** The local dataset adapters, pre-generated seed architectures and archive initialisation module are absent. The full training pipeline cannot currently run from this checkout alone. The [reuse guide](docs/REUSE.md) names each blocker and explains the environment and configuration.

## Get started

```bash
git clone https://github.com/dianzdd123/TacEvo.git
cd TacEvo
python scripts/check_release.py
python scripts/run_pipeline.py --help
```

The release check currently reports missing components and exits with status 1. It does not install dependencies, call providers or train models. After the authors supply the missing files and you prepare the experiment environment, `bash run.sh` uses the same check before launching the pipeline.

## Repository map

| Directory | Purpose |
|---|---|
| `conf/` | Task, model, provider and search configuration |
| `pipeline/` | Candidate generation, validation, training and archives |
| `llm_mutation/`, `prompts/` | LLM integration and prompts |
| `training/`, `map_elites/` | Learning and diversity-search components |
| `plotter/` | Experiment visualisation |
| `scripts/`, `tests/` | Release checks and focused offline tests |

## Paper and supplementary material

See the [TacEvo preprint](https://arxiv.org/abs/2606.30109) and [supplementary page](index.html). Preserve the original publication authorship when citing this work, and identify this fork's commit when reporting changes.

## Validation

```bash
python scripts/check_repository.py
python -m unittest discover -s tests -v
```

These checks use Python 3.10+ and do not require the historical GPU/provider environment. Passing them establishes source structure and release-check behaviour, not complete reproducibility. No repository-wide licence is included; clarify the authors' reuse terms before redistribution.
