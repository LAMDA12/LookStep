# Reproducing the LookStep main experiments

[简体中文](REPRODUCIBILITY_ZH.md)

## 1. Recommended directory layout

`LookStep/` can be placed under any parent directory. By default, data is
stored in a sibling `data/` directory. You can also set `LOOKSTEP_DATA_ROOT` or
`DATA_ROOT`.

```text
parent/
├── LookStep/
└── data/
    ├── datasets/
    │   ├── r2r/{train,val_unseen}/...
    │   └── rxr/{train,val_unseen}/...
    ├── scene_datasets/
    │   └── mp3d/<scene>/<scene>.glb
    └── trajectory_data/                 # only required to rebuild training data
        ├── R2R-CE-640x480/train/<episode>/*.png
        └── RxR-CE-640x480/train/<episode>/*.png
```

The commands below assume that you run them from the parent directory that
contains `LookStep/`. If `LookStep/` is itself the repository root, remove the
`LookStep/` prefix from the commands.

## 2. Obtain the licensed data

- Obtain the MP3D scenes through the license process and place them under
  `data/scene_datasets/mp3d/`.
- Download the R2R-CE episodes and extract them to `data/datasets/r2r/`.
- Download the RxR-CE episodes and extract them to `data/datasets/rxr/`.
- To rebuild the training supervision, also download the
  [pre-collected expert RGB trajectories](https://www.modelscope.cn/datasets/misstl/JanusVLN_Trajectory_Data)
  and extract them to `data/trajectory_data/`. This part is not required when
  only evaluating a checkpoint.

First run the tests that do not depend on external data:

```bash
bash LookStep/reproduce_paper.sh test
```

## 3. Reproduce the main results with a downloaded model

Simulation and training use separate environments to avoid conflicts between
Habitat 0.2.4 and the training stack.

```bash
conda env create -f LookStep/simulation/environment.yml
conda activate lookstep-simulation
```

Check the checkpoint, processor, Habitat installation, and data:

```bash
MODEL_PATH=/path/to/downloaded/lookstep-checkpoint \
PROCESSOR_PATH=/path/to/Qwen3-VL-8B-Instruct \
DATA_ROOT=/path/to/data \
bash LookStep/reproduce_paper.sh check-sim
```

`MODEL_PATH` must directly contain `config.json`,
`model.safetensors.index.json`, and every weight shard referenced by the index.
It must not point to the parent training-output directory.

First run a two-episode connectivity test:

```bash
MODEL_PATH=/path/to/downloaded/lookstep-checkpoint \
PROCESSOR_PATH=/path/to/Qwen3-VL-8B-Instruct \
DATA_ROOT=/path/to/data \
bash LookStep/reproduce_paper.sh smoke-r2r
```

Then run the complete Val-Unseen evaluation:

```bash
MODEL_PATH=/path/to/downloaded/lookstep-checkpoint \
PROCESSOR_PATH=/path/to/Qwen3-VL-8B-Instruct \
DATA_ROOT=/path/to/data \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash LookStep/reproduce_paper.sh eval-all

bash LookStep/reproduce_paper.sh verify
```

Results are written to `LookStep/simulation/outputs/r2r_val_unseen/` and
`LookStep/simulation/outputs/rxr_val_unseen/`. Each rank writes an
`episodes_rank<N>.jsonl` file, and rank 0 writes `metrics.json`. A run can be
resumed with the same world size. `run_config.json` prevents results produced
with different checkpoints or configurations from being mixed in one output
directory.

## 4. Rebuild the training data from expert trajectories

```bash
conda env create -f LookStep/data_construction/environment.yml
conda activate lookstep-data

DATA_ROOT=/path/to/data bash LookStep/reproduce_paper.sh check-data
DATA_ROOT=/path/to/data bash LookStep/reproduce_paper.sh build-data
DATA_ROOT=/path/to/data bash LookStep/reproduce_paper.sh prepare-data
```

`build-data` explicitly runs with `--dataset r2r` and `--dataset rxr`. The
labels are deterministic functions of the expert action sequence. Future
actions are used only to construct teacher targets and are never included in
the model input. `prepare-data` performs an episode-level split with seed 7:
the R2R validation ratio is 0.02 and the RxR validation ratio is 0. Do not start
a long training run if the resulting statistics differ from the table on this
page.

## 5. Train from scratch

```bash
conda env create -f LookStep/model_training/environment.yml
conda activate lookstep-training

bash LookStep/reproduce_paper.sh check-train
MODEL_PATH=/local/path/to/Qwen3-VL-8B-Instruct \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash LookStep/reproduce_paper.sh train
```
