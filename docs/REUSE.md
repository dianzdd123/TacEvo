# Reuse guide — TacEvo

[README](../README.md) · [Dandan Zhang’s fork](https://github.com/dianzdd123/TacEvo) · [MSEI Lab](https://zhangrobotics.com/msei_website/)

## Get this copy

```bash
git clone https://github.com/dianzdd123/TacEvo.git
cd TacEvo
```

## Purpose and release status

TacEvo explores LLM-guided neural architecture evolution for tactile perception. The released code includes configuration, mutation prompts, MAP-Elites search, training utilities and plotting. **This checkout is an incomplete research release. It cannot currently execute the full pipeline.**

| Path | Role |
|---|---|
| `conf/config.py` | Task, network, population, training and provider settings |
| `pipeline/` | Initialise an experiment, generate, validate, train and archive candidates |
| `llm_mutation/` and `prompts/` | Provider integrations and architecture-generation prompts |
| `training/` | Task-specific training utilities |
| `map_elites/` | Archive and diversity/search utilities |
| `index.html` | Supplementary material |
| `scripts/check_release.py` | Dependency-free inventory of required local components |

## Start with the release check

```bash
python scripts/check_release.py --task-type force --network-type image_regression
python scripts/run_pipeline.py --help
```

The first command currently exits with status **1**, listing missing files. This is an expected report of the release state, not a successful experiment. The default path needs `pregens/init_archive_pregen.py`, seed architectures in `pregens/image_regression_pregens/`, and local dataset adapters for force and CIFAR-10. CIFAR-10 is imported by the ResNet training utility even when another task is selected. Other tasks need their matching adapters: shape, grating or `datasets/pose/pose_loader.py`. These are project modules, not a substitute supplied by installing the similarly named Hugging Face package.

Obtain the missing implementation and seed architectures from the authors, together with the task data, labels, split definitions and their usage terms. A file-presence check alone cannot validate their interfaces or the scientific protocol.

## Environment and eventual execution

The historical environment is recorded in [environment.yaml](../environment.yaml): Python 3.9.16, PyTorch 2.1.0 and CUDA 12.1. It includes platform-specific and older pins. [requirements.txt](../requirements.txt) does not install PyTorch. These manifests are references, not a freshly tested lockfile; resolve the target operating system and GPU stack before installing. The structural checker uses a separate Python 3.10+ interpreter.

After restoring the missing components and preparing an isolated environment, a deliberately small trial is:

```bash
bash run.sh --task-type force --network-type image_regression --generations 1 --net-training-epochs 1 --n-niches 5
```

The launcher checks local files first and does **not** install packages. Other exposed arguments are `--mutation`, `--generations`, `--net-training-epochs` and `--n-niches`. Configuration defaults remain in `conf/config.py`; verify image dimensions, output dimensions, batch size, device and candidate count before increasing the experiment size. The small trial is not a reproduction of the paper.

Provider keys are read from environment variables (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`); never put credentials in tracked configuration. Provider model names and access must be checked against the selected service. Candidates are generated Python programs and are executed for evaluation: use an isolated experiment environment without unrelated credentials or data.

## Outputs and reproducibility record

Experiments create `logs/experiments/<experiment-name>/` with archive, training, prompt and plotting outputs. Record the commit, resolved package versions, dataset/split identifiers, seed settings, provider/model and generation budgets with each run. External model responses can vary between runs.

## Validation scope

The automated suite tests the release inventory with temporary fixtures. It does not call providers, generate models, load datasets or train networks. A green structural workflow does not mean the missing release components have been supplied.

## Local quality checks

Run `python scripts/check_repository.py` from a Git checkout with Python 3.10+ and Git installed. It parses Python, JSON/notebook containers and links in the navigation documents without importing research modules. Notebook cells and remote links are not executed or tested.

## Attribution and reuse terms

This is a fork of [LannWei/TacEvo](https://github.com/LannWei/TacEvo). Original paper authorship, upstream citations and notices remain authoritative. Cite the associated work and identify this fork’s commit when reporting modifications.

No repository-wide licence file is present in this snapshot. Obtain the authors’ reuse terms before assuming permission to redistribute code, datasets or checkpoints. This documentation update does not relicense the work.
