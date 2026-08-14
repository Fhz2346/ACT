# ACT++

Bimanual ViperX (ALOHA-style) simulation stack for **scripted demonstration**, **behavior cloning**, and **policy evaluation**.

This repo covers the full imitation-learning loop on a dual-arm ViperX setup in dm_control / MuJoCo. The three stages are:

1. **Scripted demonstration** — A hand-designed scripted policy rolls out in an end-effector (mocap) control environment, producing expert trajectories for tasks such as cube handoff (`transfer_cube`) or peg insertion (`insertion`). Joint targets are then replayed in the joint-space simulator and saved as HDF5 episodes (images, `qpos`, actions) for training.

2. **Behavior cloning** — Policies are trained to map visual observations and proprioception to robot actions by imitating those demos. Two options are supported:
   - **CNNMLP**: single-step deterministic BC (`chunk_size=1`)
   - **ACT**: action-chunking transformer with CVAE (`chunk_size>1`), predicting a short future action sequence to reduce compounding error

3. **Policy evaluation** — The trained policy is deployed in the joint-space `sim_env`: at each step it reads camera images and `qpos`, outputs actions, and steps the MuJoCo physics. Rollouts report success rate / return and can optionally save onscreen playback or GIF recordings.

Tasks: `transfer_cube`, `insertion`.

**Note:** This repository is a personal **refactor / re-implementation** of [MarkFzp/act-plus-plus](https://github.com/MarkFzp/act-plus-plus) (ACT++). The goal is clearer project structure and config-driven workflows.

---
## Demo

### CNNMLP

![CNNMLP transfer_cube rollout](output/gifs/transfer_cube_CNNMLP_rollout000.gif)

### ACT

![ACT transfer_cube rollout](output/gifs/transfer_cube_ACT_rollout000.gif)

## Pipeline

```text
scripted EE rollout  →  joint-space replay  →  HDF5 demos
        ↓
   train CNNMLP / ACT
        ↓
   eval in sim_env
```

1. Collect demos in end-effector (mocap) space with a scripted policy  
2. Convert to joint commands and replay in joint-space env; save HDF5  
3. Train a policy on demos  
4. Roll out the policy in `sim_env`

---

## Project layout

```text
ACT++/
├── assets/                 # MuJoCo XML (bimanual_viperx_*.xml)
├── config/
│   ├── demonstration.yaml
│   ├── BimanualViperX.yaml
│   ├── Network.yaml        # ACT backbone / transformer defaults
│   ├── Task/               # transfer_cube, insertion
│   └── imitate_episodes/   # CNNMLP.yaml, ACT.yaml
├── scripts/
│   ├── Task/
│   │   ├── 0 demonstration/   # collect HDF5
│   │   └── 1 imitate_episodes/
│   │       ├── train.py       # CNNMLP
│   │       ├── eval.py
│   │       ├── train1.py      # ACT
│   │       └── eval1.py
│   └── Utils/visualize episodes/
├── src/
│   ├── env.py              # make_ee_sim_env / make_sim_env
│   ├── Task/               # dm_control tasks
│   ├── Policy/             # ACTPolicy, scripted policies
│   ├── Network/            
│   ├── load/               # HDF5 datasets, norm stats
│   ├── detr/               # ACT DETR-VAE implementation
│   └── Pathes.py           # workspace / data / ckpt paths
├── Install/
│   ├── aloha.yaml
│   └── requirements.txt
└── ckpt/                   # saved weights + dataset_stats.pkl
```

---

## Setup

### Environment

```bash
conda env create -f Install/aloha.yaml
conda activate aloha
pip install -r Install/requirements.txt
```

### Paths

Edit `src/Pathes.py` for your machine:

| Field | Meaning |
|-------|---------|
| `wsf` | Project root |
| `data_dir` | HDF5 dataset root (default `E:\Dataset\Aloha`) |
| `ckpt_dir` | Checkpoints (`ckpt/`) |
| `config_dir` | `config/` |
| `xml_dir` | `assets/` |

### PYTHONPATH

Scripts expect `src` on `PYTHONPATH`:

```bash
# Linux / macOS
export PYTHONPATH=/path/to/ACT++/src:$PYTHONPATH

# Windows PowerShell
$env:PYTHONPATH = "E:\ProjectLocal\Robotics\ACT++\src"
```
---

## 1. Collect demonstrations

Config: `config/demonstration.yaml`  
Set `task.name` to `transfer_cube` or `insertion`.

```bash
python "scripts/Task/0 demonstration/main.py"
```

Flow:

1. `make_ee_sim_env` + scripted policy rollout  
2. Extract joint targets (gripper from commanded `gctrl`)  
3. Replay in `make_sim_env`  
4. Save episodes under `{data_dir}/{task_name}_scripted/` (and optionally human data dirs)

Optional visualization:

```bash
python "scripts/Utils/visualize episodes/main.py"
```

---

## 2. Train

### CNNMLP (`train.py`)

Config: `config/imitate_episodes/CNNMLP.yaml`

```bash
python "scripts/Task/1 imitate_episodes/train.py"
```

- Loads `{task}_scripted` and `{task}_human` under `data_dir`  
- Saves `ckpt/dataset_stats.pkl` and `{task}_CNNMLP_{step}.pth`

### ACT (`train1.py`)

Config: `config/imitate_episodes/ACT.yaml`  
Model defaults (lr, transformer, etc.): `config/Network.yaml`  
`chunk_size` overrides `num_queries`.

```bash
python "scripts/Task/1 imitate_episodes/train1.py"
```

- Uses chunked actions + pad mask  
- Loss: L1 reconstruction + KL (`kl_weight`)  
- Saves `{task}_ACT_{step}.pth` (and `*_last.pth` / best val)

---

## 3. Evaluate in simulation

### CNNMLP (`eval.py`)

```bash
python "scripts/Task/1 imitate_episodes/eval.py"
```

Uses `CNNMLP.yaml` → `eval.*` (rollouts, onscreen render, ckpt name).

### ACT (`eval1.py`)

```bash
python "scripts/Task/1 imitate_episodes/eval1.py"
```

- Queries the policy every `chunk_size` steps  
- Open-loop executes the action chunk (no temporal aggregation / VQ)  
- Drops the last 2 dims of action (dummy base) before `env.step`

If `eval_ckpt_name` is unset, the latest matching `{task}_{policy}_*.pth` in `ckpt/` is used.

---

## Config cheat sheet

| File | Role |
|------|------|
| `demonstration.yaml` | Demo length, FPS/DT, task |
| `BimanualViperX.yaml` | Cameras, `dS`/`dA`, gripper limits, init pose |
| `Task/*.yaml` | Object pose randomization ranges |
| `imitate_episodes/CNNMLP.yaml` | BC training + eval for CNNMLP |
| `imitate_episodes/ACT.yaml` | ACT training + eval |
| `Network.yaml` | ACT architecture / optimizer defaults |

Robot XML name must match assets, e.g. `bimanual_viperx_transfer_cube.xml` (`config/BimanualViperX.yaml` → `name: bimanual_viperx`).

---

## Data format (HDF5)

Per episode file roughly contains:

```text
/observations/qpos
/observations/qvel
/observations/images/{cam_name}
/action
```

Loaders append a **2-D dummy base action** so `dA=16` while the joint env uses 14-D (`action[:-2]` at eval).

Normalization: per-dimension mean/std for `qpos` and `action` (`dataset_stats.pkl`).

---

## Notes

- **CNNMLP**: `(image, qpos) → action` each step; simple and train/test consistent.  
- **ACT**: predicts an action chunk; CVAE encodes action style into `z` at train time and uses prior mean (`z=0`) at inference. Chunking is the main practical benefit.  
- First dataset directory in `load_data` is split by `train_ratio` for val; extra dirs go fully into training.  
- Sim control timestep: `sim.DT` (default `0.02` → 50 Hz).

---

## Acknowledgments

This project is based on:

- [ACT++](https://github.com/MarkFzp/act-plus-plus) — primary upstream this repo refactors
- [ACT](https://github.com/tonyzhaozh/act) — Action Chunking with Transformers
- [ALOHA](https://github.com/tonyzhaozh/aloha) — bimanual ViperX teleop / sim setup
- [DETR](https://github.com/facebookresearch/detr) — transformer detection backbone used by ACT
