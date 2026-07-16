# SAC training

`train_sac_minimal.py` trains a Stable-Baselines3 SAC policy against a 2D LBM
environment. The eel CPG mode maps four normalized policy outputs to amplitude,
frequency, wave number, and head bias before generating the segment commands.

## Eel CPG training

```powershell
python train_sac_minimal.py `
  --animal eel `
  --control-mode cpg `
  --task forward `
  --per-frame-steps 8 `
  --cpg-ramp-steps 10 `
  --cpg-hold-steps 30 `
  --episode-steps 100 `
  --warmup-exploration rand `
  --learning-starts 250 `
  --warmup-steps 15 `
  --checkpoint-every 1000 `
  --total-steps 10000
```

This is a straight-swimming task with no sampled target point. Its reward
encourages forward progress while penalizing lateral drift and heading error.
`--warmup-exploration rand` samples CPG parameters independently from their
configured physical ranges.

The training artifacts are written under `outputs/sac_minimal/` as
`sac_eel2d_forward_cpg.zip` and `sac_eel2d_forward_cpg_config.json`. The saved
JSON records the physical CPG parameter ranges used alongside the policy archive.

![Pretrained SAC forward policy evaluated in the projected eel environment](../assets/demos/sac-forward.jpg)

*The checked-in policy runs the forward task; the overlay reports reward,
accumulated return, simulation steps, and playback speed.*

## Export from an existing trained policy

Use `--load-model` with any compatible Stable-Baselines3 ZIP, then combine
`--eval-only` and `--record-video` to run inference and export an annotated LBM
MP4 without updating the policy. The command below uses the checked-in forward
policy as a runnable example.

The task, observation layout, and CPG ramp/hold timing must match those used to
train the selected model.

```powershell
python train_sac_minimal.py `
  --animal eel `
  --control-mode cpg `
  --task forward `
  --per-frame-steps 8 `
  --cpg-ramp-steps 10 `
  --cpg-hold-steps 60 `
  --load-model outputs/sac_minimal/sac_eel2d_forward_cpg.zip `
  --eval-only `
  --eval-episodes 1 `
  --record-video outputs/sac_minimal/videos/eel2d_forward_policy.mp4 `
  --playback-speed 5 `
  --render-substep-every 2
```

The exported video contains the LBM view with episode, reward, simulation-step,
and playback-speed annotations. Add `--render` only if a realtime preview is
also wanted during export.

## Continue training from an existing policy

To use an existing ZIP as a starting point for more optimization, omit
`--eval-only` and `--record-video`, then set a new training budget:

```powershell
python train_sac_minimal.py `
  --animal eel `
  --control-mode cpg `
  --task forward `
  --per-frame-steps 8 `
  --cpg-ramp-steps 10 `
  --cpg-hold-steps 60 `
  --load-model outputs/sac_minimal/sac_eel2d_forward_cpg.zip `
  --total-steps 10000 `
  --checkpoint-every 1000
```

Loading a Stable-Baselines3 ZIP restores the policy and optimizer state, but
not the previous replay buffer.

## Rendering

Add `--render` for a short visual check. Rendering adds CUDA readback and UI
overhead, so keep long training runs headless. The live CPG panel compares
executed and policy-target parameters and plots the generated yaw wave.
