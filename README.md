# Calibrated Chunk Execution

Code for identifying command-interface dynamics and compensating them when executing chunks from a frozen robot policy.

## Layout

- `cce/libero_exec_v4r1.py`: calibrated LIBERO executor.
- `cce/libero_identify_v4.py`: interface identification.
- `cce/libero_probe_v3.py`: probe collection.
- `cce/libero_plant*.py`: interface plant models.
- `cce/`: evaluation, training, analysis and diagnostic scripts.
- `cce/xvla_patch*/`: policy training and dataset integration code.
- `config/libero_config/config.yaml`: LIBERO asset path configuration.

## Setup

The scripts target Linux and depend on external simulator, policy and project modules. Install the required LIBERO / X-VLA or ManiSkill environment for the selected entry point. Model weights, datasets and generated results are not included.

Set `IDEA_V2_ROOT` to the repository directory (the import bootstrap defaults to this checkout) and `CTP_ROOT` to your external CalibrationToPrompt checkout. Configure the environment shell scripts and `config/libero_config/config.yaml` for your machine. `/opt/cce`, `/opt/CalibrationToPrompt` and `/opt/runtime` are generic example paths; evaluation and training entry points using these paths must be configured before use. Run scripts from the repository root after reviewing their arguments and imports.

The external CalibrationToPrompt source and tools, simulator installations, dataset assets and model checkpoints are not bundled. Dependencies vary by entry point; the source imports identify the required packages. This release does not claim a tested, pinned installation environment. Keep upstream attribution and comply with the applicable dependency licenses; no new blanket software license is granted by this repository.

Only source code and configuration are distributed here. Scripts may create local evaluation outputs when run; generated outputs are not tracked. Version suffixes distinguish implementations and are kept to preserve internal imports.

This distribution is source code, not a self-contained reproduction environment. No robot operation should be attempted without reviewing the hardware configuration and safety requirements.
