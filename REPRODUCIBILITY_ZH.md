# LookStep 主实验复现指南

[English](REPRODUCIBILITY.md)


## 1. 推荐目录

`LookStep/` 可以放在任意父目录。默认数据放在其同级 `data/`，也可以设置
`LOOKSTEP_DATA_ROOT` 或 `DATA_ROOT`。

```text
parent/
├── LookStep/
└── data/
    ├── datasets/
    │   ├── r2r/{train,val_unseen}/...
    │   └── rxr/{train,val_unseen}/...
    ├── scene_datasets/
    │   └── mp3d/<scene>/<scene>.glb
    └── trajectory_data/                 # 仅重建训练数据时需要
        ├── R2R-CE-640x480/train/<episode>/*.png
        └── RxR-CE-640x480/train/<episode>/*.png
```

下文命令在包含 `LookStep/` 的父目录执行。若 `LookStep/` 本身是仓库根目录，请去掉命令
中的 `LookStep/` 前缀。

## 2. 获取受许可数据

- 按许可流程获取 MP3D
  scenes，放到 `data/scene_datasets/mp3d/`。
- 下载 R2R-CE episodes，
  解压到 `data/datasets/r2r/`。
- 下载 RxR-CE episodes，
  解压到 `data/datasets/rxr/`。
- 若要重建训练监督，再下载
  [预采集 expert RGB trajectories](https://www.modelscope.cn/datasets/misstl/JanusVLN_Trajectory_Data)，
  解压到 `data/trajectory_data/`。只评测 checkpoint 时不需要这部分。

先运行不依赖外部数据的测试：

```bash
bash LookStep/reproduce_paper.sh test
```

## 3. 用已下载模型复现主结果

仿真和训练使用独立环境，以避免 Habitat 0.2.4 与训练栈发生版本冲突。

```bash
conda env create -f LookStep/simulation/environment.yml
conda activate lookstep-simulation
```

检查 checkpoint、processor、Habitat 和数据：

```bash
MODEL_PATH=/path/to/downloaded/lookstep-checkpoint \
PROCESSOR_PATH=/path/to/Qwen3-VL-8B-Instruct \
DATA_ROOT=/path/to/data \
bash LookStep/reproduce_paper.sh check-sim
```

`MODEL_PATH` 必须直接包含 `config.json`、`model.safetensors.index.json` 和 index 引用的
全部权重分片，不能指向上一级训练目录。


先跑两个 episode 的连通性测试：

```bash
MODEL_PATH=/path/to/downloaded/lookstep-checkpoint \
PROCESSOR_PATH=/path/to/Qwen3-VL-8B-Instruct \
DATA_ROOT=/path/to/data \
bash LookStep/reproduce_paper.sh smoke-r2r
```

再运行完整 Val-Unseen：

```bash
MODEL_PATH=/path/to/downloaded/lookstep-checkpoint \
PROCESSOR_PATH=/path/to/Qwen3-VL-8B-Instruct \
DATA_ROOT=/path/to/data \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash LookStep/reproduce_paper.sh eval-all

bash LookStep/reproduce_paper.sh verify
```

结果写入 `LookStep/simulation/outputs/r2r_val_unseen/` 和
`LookStep/simulation/outputs/rxr_val_unseen/`。每个 rank 写
`episodes_rank<N>.jsonl`，rank 0 写 `metrics.json`。相同 world size 可续跑；
`run_config.json` 会阻止不同 checkpoint 或配置混入同一输出目录。


## 4. 从 expert trajectories 重建训练数据

```bash
conda env create -f LookStep/data_construction/environment.yml
conda activate lookstep-data

DATA_ROOT=/path/to/data bash LookStep/reproduce_paper.sh check-data
DATA_ROOT=/path/to/data bash LookStep/reproduce_paper.sh build-data
DATA_ROOT=/path/to/data bash LookStep/reproduce_paper.sh prepare-data
```

`build-data` 明确分别以 `--dataset r2r` 和 `--dataset rxr` 运行。标签是 expert action
序列的确定性函数；future actions 仅用于构造 teacher target，绝不会放入模型输入。
`prepare-data` 以 seed 7 做 episode-level split：R2R validation ratio 为 0.02，RxR 为 0。
如果统计与本页表格不一致，不要开始长时间训练。

## 5. 从头训练

```bash
conda env create -f LookStep/model_training/environment.yml
conda activate lookstep-training

bash LookStep/reproduce_paper.sh check-train
MODEL_PATH=/local/path/to/Qwen3-VL-8B-Instruct \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash LookStep/reproduce_paper.sh train
```
