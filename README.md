# thermal-video-restoration

Restoring degraded thermal infrared video before object detection (IMP302m). Proposal (Vietnamese): [docs/proposal.md](docs/proposal.md).

## Setup

```bash
uv sync
uv run pre-commit install
scripts/download_dataset.sh   # FLIR ADAS v2 thermal frames (~4.7 GB) into data/
```

Checks before a PR:

```bash
uv run pre-commit run --all-files && uv run ruff check . && uv run pytest
```

## Layout

```
src/thermal_restore/
  data/         # FLIR ADAS 14-bit loading (loader), split by sequence, display helpers
  degradation/  # degradation simulator (proposal 3.3)
  methods/      # restoration methods M0–M7
  metrics/      # MSE, PSNR, SSIM, mAP
  detection/    # frozen object detector wrapper
configs/        # experiment YAML configs
notebooks/      # exploration and dataset checks
scripts/        # dataset download
tests/
data/           # local data, not committed (FLIR license forbids redistribution)
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the git workflow.
