# Results Manifest

This manifest records the exact values used in the report and generated artifacts.

## Baseline (accuracy_target=0.98, episodes=100)
- success_rate: 0.56
- avg_steps: 321.77
- avg_accuracy: 0.8970833333333335
- avg_coverage: 0.8970833333333335
- walker_return: -10.895290476190214
- mapper_return: 86.73944285714195

## Ablation
- no_blackhole success_rate: 0.4
- no_reset_action success_rate: 0.46
- no_action_mask success_rate: 0.06

## Strict mode (accuracy_target=1.0, episodes=100)
- success_rate: 0.09
- avg_steps: 406.66
- avg_accuracy: 0.8986805555555556

## Artifact files
- artifacts/metrics.csv
- artifacts/ablation_summary.csv
- artifacts/strict_eval_summary.json
- artifacts/plots/return_curve.png
- artifacts/plots/success_rate.png
- artifacts/plots/steps_vs_episode.png
- artifacts/plots/coverage_curve.png
