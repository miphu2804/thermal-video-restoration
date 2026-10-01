# thermal-video-restoration

Khôi phục video hồng ngoại bị suy giảm trước khi phát hiện đối tượng (IMP302m). Proposal: [docs/proposal.md](docs/proposal.md).

## Cài đặt

```bash
uv sync          # Python 3.11, tạo .venv
uv run pytest
```

## Cấu trúc

```
src/thermal_restore/
  data/         # đọc FLIR ADAS 14-bit, chia tập theo chuỗi video
  degradation/  # bộ mô phỏng suy giảm (mục 3.3)
  methods/      # các phương pháp khôi phục M0–M7
  metrics/      # MSE, PSNR, SSIM, mAP
configs/        # cấu hình thí nghiệm
notebooks/      # thử nghiệm
data/           # dữ liệu cục bộ, không commit
```
