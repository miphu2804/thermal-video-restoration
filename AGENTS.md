# AGENTS.md

Hướng dẫn cho AI coding agent làm việc trong repo này.

## Dự án

Đồ án IMP302m: khôi phục video hồng ngoại bị suy giảm trước khi phát hiện đối tượng. Đặc tả đầy đủ ở `docs/proposal.md` — đọc mục liên quan trước khi sửa code (3.3 bộ mô phỏng, 3.4 phương pháp M0–M7, 3.6 thí nghiệm và thước đo).

## Môi trường

- Python 3.11, quản lý bằng `uv`. Thêm dependency bằng `uv add`, không sửa tay `uv.lock`.
- Lệnh kiểm tra: `uv run ruff check . && uv run ruff format --check . && uv run pytest`.

## Cấu trúc

```
src/thermal_restore/
  data/         # đọc FLIR ADAS 14-bit, chia tập theo chuỗi video
  degradation/  # bộ mô phỏng suy giảm (mục 3.3)
  methods/      # phương pháp khôi phục M0–M7
  metrics/      # MSE, PSNR, SSIM, mAP
tests/  notebooks/  configs/  data/ (gitignore)
```

## Quy ước

- Ưu tiên hàm thuần (numpy array vào, array ra); I/O file để ở `data/` hoặc script.
- Tham số suy giảm tính theo thang tương đương 8-bit (0–255) như proposal.
- Mọi thứ ngẫu nhiên nhận `seed` hoặc `np.random.Generator` để tái lập.
- Chia tập theo chuỗi video, không theo khung (tránh rò rỉ dữ liệu).
- Code, comment, docstring (Google style), commit message, tên branch và tiêu đề PR viết **tiếng Anh**. Chỉ tài liệu (`docs/`, README, CONTRIBUTING, mô tả issue/PR) viết tiếng Việt.
- Không commit dữ liệu, checkpoint hay output.

## Git

Theo `CONTRIBUTING.md`: branch `<type>/<issue-number>-<short-description>`, Conventional Commits, PR vào `main` cần 1 review, squash merge.
