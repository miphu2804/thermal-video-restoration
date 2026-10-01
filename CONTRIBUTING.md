# Đóng góp

## Quy trình

1. Nhận một issue (tự assign).
2. Tạo branch từ `main` mới nhất.
3. Commit, push, mở Pull Request ghi `Closes #<số issue>`.
4. Cần ít nhất 1 approve; merge bằng **Squash and merge** (branch tự xoá sau merge).

Không push thẳng lên `main`.

## Ngôn ngữ

- **Tiếng Anh:** tên branch, commit message, tiêu đề PR, tên biến/hàm, comment và docstring trong code.
- **Tiếng Việt:** tài liệu (`docs/`, README, CONTRIBUTING, mô tả issue/PR).

## Tên branch

`<type>/<issue-number>-<short-description>`, tiếng Anh, chữ thường, nối bằng `-`.

```
feat/002-degradation-simulator
fix/003-psnr-range
docs/001-flir-stats
```

## Commit message

Theo [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short description>
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

Mô tả viết tiếng Anh, thể mệnh lệnh, chữ thường đầu câu, không chấm câu cuối. Ví dụ:

```
feat(degradation): add fixed and temporal stripe noise
fix(metrics): compute PSNR on 0-255 scale
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
