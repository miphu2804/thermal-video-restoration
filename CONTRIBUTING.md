# Đóng góp

## Quy trình

1. Nhận một issue (tự assign).
2. Tạo branch từ `main` mới nhất.
3. Commit, push, mở Pull Request ghi `Closes #<số issue>`.
4. Cần ít nhất 1 approve; merge bằng **Squash and merge** (branch tự xoá sau merge).

Không push thẳng lên `main`.

## Tên branch

`<loại>/<số-issue>-<mô-tả-ngắn>`, chữ thường, nối bằng `-`.

```
feat/002-degradation-simulator
fix/003-psnr-range
docs/001-flir-stats
```

## Commit message

Theo [Conventional Commits](https://www.conventionalcommits.org/):

```
<loại>(<phạm vi>): <mô tả ngắn>
```

| Loại | Dùng khi |
|---|---|
| `feat` | Thêm tính năng (phương pháp, thước đo, loader…) |
| `fix` | Sửa lỗi |
| `exp` | Thí nghiệm, notebook, config chạy |
| `docs` | Tài liệu |
| `test` | Thêm/sửa test |
| `refactor` | Đổi cấu trúc code, không đổi hành vi |
| `chore` | Dependency, cấu hình tool |

Phạm vi gợi ý: `data`, `degradation`, `methods`, `metrics`, `detector`.

Mô tả viết tiếng Việt hoặc tiếng Anh, ngắn gọn, không chấm câu cuối. Ví dụ:

```
feat(degradation): thêm sọc cố định và sọc theo thời gian
fix(metrics): tính PSNR trên thang 0–255
```

Vì merge bằng squash, **tiêu đề PR** sẽ thành commit trên `main` — đặt tiêu đề PR theo đúng format trên.

## Trước khi mở PR

```bash
uv sync
uv run ruff check .
uv run ruff format .
uv run pytest
```

## Dữ liệu

Không commit dữ liệu, checkpoint (`*.pt`) hay output. FLIR ADAS có giấy phép không cho phân phối lại; để trong `data/` (đã gitignore).
