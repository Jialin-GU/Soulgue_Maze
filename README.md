# Soulgue-Maze

Dual-agent **tabular Q-learning** project for CDS524 Assignment 1.

## Features
- Random connected maze generation (default 8x8) with 2 entries and fixed blackholes per episode
- Training map distribution control:
  - `fixed_layout_seed`: single fixed maze
  - `maze_pool_size > 0`: sample from a small fixed maze set (default)
  - `maze_pool_size = 0` and no fixed seed: fully random every episode
- Blackhole placement is now constrained:
  - not too close to entries (`blackhole_min_entry_distance`)
  - must preserve major reachability from each entry (`blackhole_min_reachable_ratio`)
  - relaxed fallback threshold (`blackhole_relaxed_reachable_ratio`)
- Partial observability for Walker (local 4-neighbor walls only)
- Mapper reconstructs wall layout (`unknown/free/wall`) from Walker trajectory + local observations
- Mapper reset action to re-plan when stuck
- Multi-phase training for non-stationary dual-agent setup:
  - Phase A: train Walker, Mapper fixed NOOP
  - Phase B: freeze Walker, train Mapper reset policy
  - Phase C: alternating fine-tuning
- Pygame dual-view UI:
  - Left: Walker view (movement + fog only)
  - Right: Mapper view (reconstruction + blackhole visibility for mapper)
  - Right panel: metrics and controls
- Anti-oscillation safeguards in inference (small epsilon + stuck/oscillation guard reset)
- Joint objective shaping for both agents: finish full map reconstruction with fewer Walker steps
- Evaluation + ablation + plots for report

## Project Structure
- `/config.py`: centralized configs
- `/env.py`: environment, maze generation, observations, rewards
- `/agents.py`: Walker/Mapper tabular Q-learning
- `/train.py`: phase training pipeline
- `/evaluate.py`: evaluation, ablation, plotting
- `/ui.py`: local pygame demo
- `/scripts`: CLI entrypoints

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train
```bash
python scripts/run_train.py
```

Optional:
```bash
python scripts/run_train.py --height 6 --width 6 --phase-a 1200 --phase-b 900 --phase-c 1200
python scripts/run_train.py --maze-pool-size 8 --maze-pool-seed 2026 --accuracy-target 1.0
python scripts/run_train.py --fixed-layout-seed 1234   # single fixed maze
```

Outputs:
- `checkpoints/walker_q.npy`
- `checkpoints/mapper_q.npy`
- `artifacts/metrics.csv`

## Evaluate + Ablation + Plots
```bash
python scripts/run_eval.py --episodes 50
```

Outputs:
- `artifacts/ablation_summary.csv`
- `artifacts/plots/return_curve.png`
- `artifacts/plots/success_rate.png`
- `artifacts/plots/steps_vs_episode.png`
- `artifacts/plots/coverage_curve.png`

## Run UI (Mac local)
```bash
python scripts/run_ui.py
```

Optional:
```bash
python scripts/run_ui.py --maze-pool-size 8 --maze-pool-seed 2026
python scripts/run_ui.py --fixed-layout-seed 1234
```

### Keys
- `M`: manual/AI mode
- `SPACE`: pause/resume
- `R`: reset episode
- `B`: toggle blackhole debug
- `W`: toggle true wall debug
- `Arrow keys`: manual Walker move
- `E`: manual Mapper reset trigger (to Entry 0)

## Notes for Assignment
- Uses **tabular Q-learning only** (no neural network / DQN)
- Designed for offline training (Colab or local CPU) and local UI demo on Mac
- Report target: 1000-1500 words
- Demo target: 3-5 minutes
