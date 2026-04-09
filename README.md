# ZXYVLN: CoT-assisted Q-SFT for FantasyVLN

## Overview

CoT-assisted Q-SFT integrates Q-value weighted SFT with FantasyVLN's Visual CoT mechanism.

## Key Innovation

V-CoT generates `<var>...</var>` tokens representing imagined future observations.
These are used as implicit value estimators for Bellman targets:

```
w_t = r_t + gamma * V(s_{t+1})
where V(s_{t+1}) = max_a p_MM(a | imagined_obs_from_VCoT)
```

## Files

- `trainer_qsft_cot.py`: CoTQSFTTrainer with V-CoT integration
- `data/processor_qsft_cot.py`: Data processor with Q-weights and var mask
- `train_qsft_cot.py`: Training entry point
- `train_qsft_cot.sh`: Training script

## Quick Start

```bash
cp trainer_qsft_cot.py train_qsft_cot.py /path/to/fantasy-vln/
cp -r data/ /path/to/fantasy-vln/
bash train_qsft_cot.sh
```

## Hyperparameters

- `q_gamma=0.95`: Discount factor
- `lambda_cot=0.5`: CoT loss weight  
- `lambda_align=0.1`: Cross-mode alignment weight
- `clip_weight=5.0`: Q-weight clipping
- `ema_decay=0.995`: EMA target update rate
