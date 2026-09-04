# LookStep

[简体中文](README_ZH.md) [MODEL](https://huggingface.co/Kunyang-YU/LookStep)

LookStep is a self-contained package for reproducing the main experiments in
**LookStep: Efficient Vision-Language Navigation with Linguistic Foresight and
Event Driven Memory**. After downloading the paper checkpoint and preparing
the licensed Matterport3D and VLN-CE data, this directory can be used on its
own to evaluate the R2R-CE and RxR-CE Val-Unseen main results. It also supports
reconstructing the supervision data from expert trajectories and training the
model.

## Three modules

```text
LookStep/
├── data_construction/   # 1. R2R/RxR expert trajectories -> LFS/EDRM supervision JSONL
├── model_training/      # 2. ms-swift conversion and Qwen3-VL full fine-tuning
└── simulation/          # 3. Online Habitat main-experiment evaluation for R2R/RxR
```

The complete pipeline is:

```text
R2R-CE + RxR-CE expert trajectories
  -> deterministic LookStep labels
  -> ms-swift multimodal JSONL
  -> Qwen3-VL-8B full fine-tuning
  -> Habitat R2R-CE / RxR-CE Val-Unseen evaluation
```

The training target uses the following fixed order:

```text
progress -> event -> memory_write -> memory_role
         -> outcomes for four candidate actions -> action
```

## Reproduce the main results with a downloaded model

First prepare the pinned environment and licensed data by following the
[reproduction guide](REPRODUCIBILITY.md), then run:

```bash
bash LookStep/reproduce_paper.sh test
conda activate lookstep-simulation

MODEL_PATH=/path/to/downloaded/lookstep-checkpoint \
PROCESSOR_PATH=/path/to/Qwen3-VL-8B-Instruct \
DATA_ROOT=/path/to/data \
bash LookStep/reproduce_paper.sh check-sim

MODEL_PATH=/path/to/downloaded/lookstep-checkpoint \
PROCESSOR_PATH=/path/to/Qwen3-VL-8B-Instruct \
DATA_ROOT=/path/to/data \
bash LookStep/reproduce_paper.sh smoke-r2r

MODEL_PATH=/path/to/downloaded/lookstep-checkpoint \
PROCESSOR_PATH=/path/to/Qwen3-VL-8B-Instruct \
DATA_ROOT=/path/to/data \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash LookStep/reproduce_paper.sh eval-all

bash LookStep/reproduce_paper.sh verify
```

## Main entry points

| Module | Entry point | Purpose |
| --- | --- | --- |
| Data construction | `data_construction/build_short_label_dataset.py` | Derive main-experiment labels from R2R/RxR expert actions |
| Model training | `model_training/prepare_ms_swift_data.py` | Split by episode and convert to ms-swift JSONL |
| Model training | `model_training/scripts/train_qwen3vl_full.sh` | Full-fine-tuning recipe for the paper checkpoint |
| Simulation | `simulation/evaluate_short_label_sim.py` | Online EventFIFO navigation in Habitat |
| Orchestrator | `reproduce_paper.sh` | Staged check/build/prepare/train/eval/verify workflow |

## Citation

```bibtex
@inproceedings{lookstep,
  title     = {LookStep: Efficient Vision-Language Navigation with Linguistic Foresight and Event Driven Memory},
  author    = {Kun-Yang Yu and Yingzhe Li and Hongyu Xu and Shi-Yu Tian and Zhi Zhou and Yang Chen and Ming Yang and Sheng Wang and Qing Yu and Lan-Zhe Guo and Yu-Feng Li},
  booktitle = {The 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026}
}
```

If you have any questions, please contact yuky@lamda.nju.edu.cn.
