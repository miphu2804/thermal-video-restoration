# Khôi phục video hồng ngoại bị suy giảm trước khi phát hiện đối tượng

> IMP302m | Proposal đồ án. Trích từ `docs/proposal.pdf`.

## 1. Động lực

Camera hồng ngoại (camera nhiệt) ghi lại nhiệt độ thay vì ánh sáng, nên vẫn nhìn được trong bóng tối, khói và sương mù. Vì vậy nó được dùng trong giám sát, lái xe và robot. Nhưng video nhiệt thô thường bị nhiễu, và nhiễu đó làm hỏng các tác vụ phía sau như phát hiện đối tượng. Có năm lý do để làm đề tài này.

- **Nhiễu đặc thù của cảm biến.** Mỗi điểm ảnh của cảm biến đáp ứng hơi khác nhau, tạo ra nhiễu mẫu cố định. Camera không làm lạnh phải cập nhật hiệu chỉnh định kỳ [2], và sọc dọc có thể xuất hiện trở lại khi nhiệt độ môi trường thay đổi nhanh [3].
- **Nhiễu làm hỏng tác vụ.** Trong thí nghiệm của TIDY, hệ đo quán tính bị nhiễu cố định đánh lừa, vì nhiễu này đứng yên trong ảnh trong khi cảnh di chuyển [5].
- **Tăng tương phản có thể làm tệ hơn.** Nhiễu còn sót lại sau hiệu chỉnh có thể bị tăng tương phản cục bộ khuếch đại thành các vết dễ nhầm với cấu trúc nhiệt thật [6]. Vì vậy thứ tự xử lý quan trọng.
- **Video chưa được khai thác.** Các kỹ thuật hiệu chỉnh sọc phổ biến thường chỉ dùng một khung [4], và TIDY xử lý từng khung riêng lẻ, coi việc dùng chuỗi khung là hướng tương lai [5].
- **Khoảng cách giữa nhiễu mô phỏng và nhiễu thật.** Bộ lọc trung vị và song phương đạt khoảng 28 dB PSNR trên nhiễu mô phỏng nhưng chỉ khoảng 11 dB trên nhiễu thật trong thí nghiệm của TIDY. Đó là hai bộ dữ liệu khác nhau nên chỉ nên đọc như một dấu hiệu [5]. TIR-Diffusion cũng nêu rằng mô hình huấn luyện trên nhiễu Gaussian mô phỏng không loại được nhiễu cố định thật [8].

**Đồ án làm gì.** Đồ án so sánh có hệ thống các phương pháp khôi phục trong syllabus, cộng thêm một mạng nhỏ, trên video nhiệt bị suy giảm có kiểm soát. Kết quả được đo ở hai tầng: chất lượng ảnh và hiệu năng phát hiện đối tượng. Toàn bộ thiết kế ưu tiên chi phí huấn luyện thấp để chạy được trên một GPU phổ thông.

## 2. Mô tả bài toán

### 2.1 Phát biểu

Cho video nhiệt sạch x(t). Bộ mô phỏng tạo video suy giảm y(t) theo chuỗi bốn bước, theo thứ tự của mô hình suy giảm trong PPFN [7]:

```
x(t) → nén tương phản → làm mờ → thêm sọc → thêm nhiễu ngẫu nhiên → y(t)
```

Sọc gồm hai loại: sọc cố định giữ nguyên trong cả chuỗi video, và sọc theo thời gian đổi ở mỗi khung do nhiễu nhiệt của bộ chuyển đổi tương tự–số [9].

Mục tiêu là một bộ khôi phục R, nhận y(t) cùng các khung lân cận và cho ra ảnh x̂(t) gần x(t). Tiêu chí thành công có hai tầng: x̂(t) gần x(t) về chỉ số chất lượng ảnh, và một detector cố định F cho kết quả trên x̂(t) gần kết quả trên x(t).

### 2.2 Câu hỏi nghiên cứu

| Mã | Câu hỏi | Giả thuyết ban đầu (cần kiểm chứng) |
|---|---|---|
| RQ1 | Phương pháp nào khôi phục tốt nhất cho từng loại suy giảm? | Không có phương pháp thắng ở mọi loại. |
| RQ2 | Chỉ số chất lượng ảnh (PSNR, SSIM) có đồng thuận với mAP không? | Không phải lúc nào cũng đồng thuận; làm mịn mạnh có thể tăng PSNR nhưng mất chi tiết cần cho detector. |
| RQ3 | Dùng nhiều khung hình giúp thêm bao nhiêu so với một khung? | Giúp rõ với nhiễu ngẫu nhiên và sọc theo thời gian. Với sọc cố định, chỉ giúp khi có chuyển động và đã căn chỉnh khung (suy luận của tác giả). |
| RQ4 | Khử nhiễu trước rồi tăng tương phản có tốt hơn thứ tự ngược lại không? | Có, vì tăng tương phản khuếch đại nhiễu còn sót [6][7]. |

## 3. Phương pháp kỹ thuật

### 3.1 Tổng quan chuỗi xử lý

Video nhiệt gốc được làm suy giảm bằng bộ mô phỏng, sau đó khôi phục theo thứ tự ngược với lúc làm suy giảm, rồi tăng tương phản. Detector cố định cho kết quả để tính mAP. Chất lượng ảnh được đo riêng bằng cách so với ảnh gốc.

*Hình 1 (trong PDF): Chuỗi xử lý và hai tầng đo (chất lượng ảnh và hiệu năng phát hiện).*

### 3.2 Dữ liệu

| Bộ dữ liệu | Nội dung | Định dạng | Vai trò |
|---|---|---|---|
| FLIR ADAS [11] | 26.442 khung có nhãn, hơn 520.000 khung bao quanh, 15 loại đối tượng. Phần video: 7.498 khung ghi ở 24 Hz, khung nhiệt và khung thường khớp 1:1. Camera 640×512. | Nhiệt 14-bit TIFF (chưa qua AGC); nhiệt 8-bit JPEG; nhãn dạng MSCOCO | Bộ chính. Khung 14-bit làm ảnh gốc. Giấy phép phi thương mại, không phân phối lại. |
| SCaN-TIR [5] | Khoảng 32,5 nghìn cặp ảnh 640×512 sạch và nhiễu thật, 9 chuỗi liên tục, ghi bằng hai camera đặt cạnh nhau (một camera tắt NUC). | Ảnh nhiệt 8-bit, không có nhãn phát hiện | Tuỳ chọn. Kiểm chứng trên nhiễu thật, chỉ đo PSNR và SSIM. |
| LLVIP [13] | 15.488 cặp ảnh hồng ngoại và ảnh thường, 26 địa điểm, cảnh đường phố buổi tối, có nhãn người đi bộ. | Ảnh hồng ngoại, nhãn khung bao quanh | Tuỳ chọn. Kiểm tra detector trên bộ dữ liệu thứ hai. |

- **Chia theo chuỗi video, không chia theo khung.** Khung liền kề gần như giống nhau, nên chia theo khung gây rò rỉ dữ liệu. Tỉ lệ 70% / 15% / 15% cho huấn luyện / chọn tham số / kiểm tra.
- Chọn tham số trên tập chọn tham số, báo cáo trên tập kiểm tra.

> **Cần xác minh khi tải về:** số khung video thực tế và tốc độ khung (trang FLIR ghi 24 Hz, bản mô tả cũ của DSIAC [12] nhắc đến 30 khung/giây và lấy mẫu thưa 1–2 khung/giây ở phần ảnh có nhãn); hai con số 26.442 và 7.498 có giao nhau không; số cặp của LLVIP (bản arXiv đầu ghi 16.836 cặp, bản mới nhất ghi 15.488); giấy phép và việc phát hành của SCaN-TIR.

### 3.3 Bộ mô phỏng suy giảm

Suy giảm được sinh ngay khi huấn luyện (on-the-fly) từ khung 14-bit gốc, nên không phải lưu dữ liệu suy giảm và mỗi lần lặp thấy một bản nhiễu khác. Tham số tính theo thang tương đương 8-bit (0–255), là giá trị khởi điểm đề xuất, sẽ hiệu chỉnh để ảnh mô phỏng trông gần nhiễu thật.

| Bước | Loại | Mô hình | Nhẹ / Vừa / Nặng | Mục syllabus |
|---|---|---|---|---|
| 1 | Tương phản thấp | Nén dải động quanh giá trị trung bình về một tỉ lệ của dải gốc. | 60% / 40% / 25% | Chương 1 |
| 2 | Mờ | Làm mờ Gaussian (độ lệch chuẩn σb, đơn vị điểm ảnh). | σb = 1 / 2 / 3 | 3.4 |
| 3a | Sọc cố định | Mỗi cột nhận một độ lệch ngẫu nhiên (phân phối chuẩn, σs), giữ nguyên cả chuỗi. | σs = 2 / 4 / 8 | 3.2 |
| 3b | Sọc theo thời gian | Mỗi cột nhận độ lệch ngẫu nhiên mới ở mỗi khung (σl). | σl = 1 / 2 / 4 | 3.2 |
| 4 | Nhiễu ngẫu nhiên | Nhiễu chuẩn độc lập ở mỗi điểm ảnh và mỗi khung (σt). | σt = 3 / 6 / 12 | 3.2 |

Giả định nhiễu chuẩn là một đơn giản hoá; mục 3.2 của syllabus liệt kê nhiều mô hình nhiễu khác.

### 3.4 Các phương pháp khôi phục

Các phương pháp chia làm hai nhóm. Nhóm thứ nhất không cần huấn luyện. Nhóm thứ hai chỉ gồm một mạng nhỏ, có hai biến thể dùng chung kiến trúc. Sau khôi phục luôn có bước tăng tương phản (M1), trừ thí nghiệm E4.

| Mã | Phương pháp | Mục syllabus | Nhắm tới |
|---|---|---|---|
| M0 | Không xử lý (đường cơ sở) | | Mốc so sánh |
| M1 | Cân bằng histogram; CLAHE | Chương 1 | Tương phản thấp |
| M2 | Lọc không gian: Gaussian, trung vị, lọc theo cột | 2.1, 2.2, 2.4 | Nhiễu ngẫu nhiên; sọc |
| M3 | Lọc miền tần số: chặn dải tại thành phần sọc. Sọc dọc đổi theo phương ngang, nên năng lượng tập trung dọc trục tần số ngang. | 2.5, 3.3 | Sọc dọc |
| M4 | Khử nhiễu wavelet | 3.5, 4.5 | Nhiễu ngẫu nhiên; tách sọc theo hướng |
| M5 | Wiener và CLS (dùng lại notebook đã có) | 3.6–3.8 | Mờ kèm nhiễu |
| M6 | Lọc theo thời gian có bù chuyển động: ước lượng chuyển động, căn chỉnh khung, rồi lấy trung bình hoặc trung vị | 3.11, 3.12 | Nhiễu ngẫu nhiên; sọc theo thời gian |
| M7a | Mạng nhỏ, một khung vào. Mạng tích chập dư dự đoán phần nhiễu; ảnh khôi phục bằng đầu vào trừ nhiễu dự đoán. | 4.7 | Tất cả loại suy giảm |
| M7b | Cùng mạng đó, ba khung vào: khung hiện tại và hai khung lân cận đã căn chỉnh bằng ước lượng chuyển động cổ điển. Chỉ khác số kênh đầu vào. | 3.11, 4.7 | Như M7a, thêm thông tin thời gian |

### 3.5 Cấu hình huấn luyện nhẹ

Chỉ có hai thứ cần huấn luyện, và cả hai đều nhỏ.

- **Detector.** Một mô hình phát hiện đối tượng nhỏ có sẵn, tinh chỉnh một lần trên khung nhiệt sạch của tập huấn luyện rồi đóng băng. Mọi điều kiện so sánh dùng đúng mô hình đó. Không huấn luyện lại trên ảnh suy giảm.
- **Mạng khôi phục (M7a, M7b).** Thiết kế khởi điểm: khoảng 8–10 lớp tích chập, 32–64 kênh, vài trăm nghìn tham số. Đầu vào là mảnh ảnh 128×128 cắt ngẫu nhiên, mức suy giảm chọn ngẫu nhiên trong ba mức, hàm mất mát L1 giữa ảnh khôi phục và ảnh gốc, bộ tối ưu Adam, tính toán độ chính xác hỗn hợp.
- **Tuỳ chọn giảm chi phí.** Chạy mạng trong miền wavelet Haar một mức: độ phân giải mỗi chiều giảm một nửa, và theo TIDY chi phí tính toán giảm xấp xỉ 4 lần [5].
- **Mục tiêu chi phí.** Mỗi lần huấn luyện xong trong vài giờ trên một GPU 8–12 GB. Đây là mục tiêu thiết kế, sẽ đo thực tế.
- **Không huấn luyện mô hình lớn.** PPFN, mô hình hiện đại cho bài toán này, huấn luyện trên 4 GPU với 300 epoch [7]; đồ án chỉ dùng nó làm tài liệu tham khảo.

### 3.6 Thí nghiệm và thước đo

| Điều kiện | Đầu vào của detector | Ý nghĩa |
|---|---|---|
| A | Khung nhiệt gốc, chưa suy giảm | Cận trên: mức tốt nhất có thể đạt |
| B | Khung đã suy giảm, không khôi phục | Cận dưới: mức khi không làm gì |
| C | Khung đã suy giảm và đã khôi phục (M1–M7) | Kết quả cần đo; xem khôi phục lấp được bao nhiêu khoảng cách giữa B và A |

| Mã | Trả lời | Cách làm |
|---|---|---|
| E1 | RQ1 | Chạy M0–M7 trên từng loại suy giảm và ba mức; đo MSE, PSNR, SSIM; ghi thời gian xử lý mỗi khung. |
| E2 | RQ2 | Với cùng các kết quả, đo thêm mAP; so thứ hạng phương pháp theo hai thước đo. |
| E3 | RQ3 | So sánh bản một khung và bản nhiều khung của cùng phương pháp (M6 và M7a so với M7b), với cửa sổ 1, 3, 5 khung. |
| E4 | RQ4 | So sánh khử nhiễu rồi tăng tương phản với thứ tự ngược lại. |
| E5 | Tuỳ chọn | Kiểm chứng trên SCaN-TIR: chỉ PSNR và SSIM, vì không có nhãn phát hiện. Hai camera được căn bằng hiệu chỉnh lập thể nên ảnh tham chiếu không khớp tuyệt đối từng điểm ảnh (suy luận của tác giả). |

- **Chất lượng ảnh (chương 8):** MSE, PSNR, SSIM; cần ảnh tham chiếu nên chỉ áp dụng cho dữ liệu mô phỏng.
- **Hiệu năng phát hiện (chương 11):** mAP tại ngưỡng IoU 0,5 là thước đo chính, thêm mAP trung bình qua nhiều ngưỡng làm thước đo phụ.
- **Độ tin cậy:** lặp lại với nhiều hạt giống ngẫu nhiên cho nhiễu mô phỏng và báo cáo mức dao động.

### 3.7 Hạn chế đã biết

- Khung "gốc" vẫn mang nhiễu của chính cảm biến, nên PSNR đo mức khôi phục so với tham chiếu đó, không phải so với sự thật tuyệt đối.
- Nhiễu mô phỏng có thể không giống nhiễu thật (mục 1); E5 là cách kiểm tra một phần.
- Detector tinh chỉnh trên ảnh sạch có thể nhạy với suy giảm; bài WiSE-OD ghi nhận tinh chỉnh trên ảnh hồng ngoại tăng độ chính xác nhưng giảm độ bền khi phân phối dữ liệu đổi [10]. Đồ án báo cáo đúng kết quả này thay vì huấn luyện lại detector, để giữ chi phí thấp.

## 4. Thuật ngữ

| Thuật ngữ | Giải thích |
|---|---|
| Hồng ngoại / ảnh nhiệt | Ảnh ghi bức xạ nhiệt của vật thể thay vì ánh sáng nhìn thấy. |
| FPN (Fixed Pattern Noise) | Nhiễu mẫu cố định: mỗi điểm ảnh lệch một lượng gần như không đổi so với điểm khác. |
| NUC (Non-Uniformity Correction) | Hiệu chỉnh không đồng đều: bù phần lệch giữa các điểm ảnh. |
| AGC (Automatic Gain Control) | Điều chỉnh độ lợi tự động, thường dùng để ép ảnh nhiệt 14-bit về 8-bit để hiển thị; bước này làm mất thông tin. |
| 14-bit / 8-bit | Số mức xám mỗi điểm ảnh có thể có: 14-bit là 16.384 mức, 8-bit là 256 mức. |
| MSE, PSNR, SSIM | Ba thước đo chất lượng ảnh so với ảnh tham chiếu: sai số bình phương trung bình; tỉ lệ tín hiệu trên nhiễu đỉnh; độ tương tự cấu trúc. |
| CLAHE | Cân bằng histogram thích nghi có giới hạn tương phản: cân bằng theo từng vùng nhỏ và hạn chế khuếch đại nhiễu. |
| Wiener / CLS | Hai bộ lọc khôi phục: Wiener tối thiểu hoá sai số bình phương trung bình; CLS (bình phương tối thiểu có ràng buộc) thêm ràng buộc độ mượt. |
| Wavelet | Phép biến đổi tách ảnh thành các thành phần theo tỉ lệ và hướng, tiện cho việc tách nhiễu. |
| Bù chuyển động | Căn chỉnh các khung liền kề theo chuyển động của cảnh trước khi kết hợp chúng. |
| Detector | Mô hình phát hiện đối tượng: cho ra khung bao quanh và nhãn của từng đối tượng. |
| IoU, mAP | IoU: mức chồng lấp giữa khung dự đoán và khung đúng. mAP: độ chính xác trung bình trên các lớp đối tượng. |
| On-the-fly | Sinh dữ liệu ngay trong lúc huấn luyện thay vì lưu sẵn ra đĩa. |

## 5. Tài liệu tham khảo

1. Tài liệu bài giảng IMP302m, chương 1–12.
2. ADMIRE: a locally adaptive single-image, non-uniformity correction and denoising algorithm: application to uncooled IR camera. arXiv:2411.03615. https://arxiv.org/abs/2411.03615
3. ASCNet: Asymmetric Sampling Correction Network for Infrared Image Destriping. arXiv:2401.15578. https://arxiv.org/abs/2401.15578
4. Temporal Denoising of Infrared Images via Total Variation and Low-Rank Bidirectional Twisted Tensor Decomposition. Remote Sensing 17(8), 1343 (2025). https://doi.org/10.3390/rs17081343
5. Rhee, T. H., Lee, D.-G., Kim, A. TIDY: Thermal Infrared Image Denoising via Wavelet Domain Entropy and Directional Stripe Index. arXiv:2606.19813 (06/2026). https://arxiv.org/abs/2606.19813 ; mã và dữ liệu: https://github.com/williamrheeth/TIDY
6. SA-RNE: structure-aware residual noise estimation for pattern-preserving infrared image denoising. Springer (2026), DOI 10.1007/s10044-026-01715-x. (Mới đọc phần tóm tắt.)
7. Liu, J. et al. Enhancing Infrared Vision: Progressive Prompt Fusion Network and Benchmark (PPFN). NeurIPS 2025. arXiv:2510.09343. https://arxiv.org/abs/2510.09343
8. Rhee, T. H. et al. TIR-Diffusion: Diffusion-based Thermal Infrared Image Denoising via Latent and Wavelet Domain Optimization. ICRA 2025 TIRO workshop. arXiv:2508.03727. https://arxiv.org/abs/2508.03727
9. Cai, L., Dong, X., Zhou, K., Cao, X. Exploring Video Denoising in Thermal Infrared Imaging: Physics-Inspired Noise Generator, Dataset, and Model. IEEE Trans. Image Processing 33, 3839–3854 (2024). DOI 10.1109/TIP.2024.3390404
10. Medeiros, H. R. et al. WiSE-OD: Benchmarking Robustness in Infrared Object Detection. WACV 2026. arXiv:2507.18925. https://arxiv.org/abs/2507.18925
11. Teledyne FLIR Thermal Dataset for Algorithm Training (FLIR ADAS). https://www.flir.com/adasdataset ; điều khoản: https://flir.ca/oem/adas/adas-dataset-agree
12. DSIAC. Infrared Imagery Datasets. https://dsiac.dtic.mil/technical-inquiries/notable/infrared-imagery-datasets
13. Jia, X. et al. LLVIP: A Visible-infrared Paired Dataset for Low-light Vision. arXiv:2108.10831. https://arxiv.org/abs/2108.10831

*Ghi chú: các tài liệu được tra cứu qua toàn văn (TIDY, PPFN) hoặc tóm tắt và đoạn trích (các tài liệu còn lại); cần xác nhận trước khi trích dẫn chính thức.*
