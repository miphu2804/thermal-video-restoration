# thermal-video-restoration

Restoring degraded thermal infrared video before object detection (IMP302m). Proposal (Vietnamese): [docs/proposal.md](docs/proposal.md).

## Setup

```bash
uv sync                  # Python 3.11, creates .venv
uv run pre-commit install  # isort + black on commit
uv run pytest
```

## Layout

```
src/thermal_restore/
  data/         # FLIR ADAS 14-bit loading, split by video sequence
  degradation/  # degradation simulator (proposal 3.3)
  methods/      # restoration methods M0–M7
  metrics/      # MSE, PSNR, SSIM, mAP
  detection/    # frozen object detector wrapper
configs/        # experiment YAML configs
notebooks/      # exploration
data/           # local data, not committed
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the git workflow.
