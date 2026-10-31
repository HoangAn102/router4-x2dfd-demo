# PREREG — R6 (đăng ký trước phân tích số liệu test)

Viết: 31/10/2026, khoảng 10:15 UTC (giờ commit + sha256: xem `PROGRESS.md` và commit git chứa bản sao `docs/PREREG_R6.md`). Sau khi commit: **không sửa luật, không đổi ngưỡng hay chỉ số sau khi thấy kết quả test.** Mọi phân tích ngoài danh sách dưới đây phải mang nhãn **"thăm dò (post-hoc)"**.

## 0. Công khai trước (trung thực)
- Lúc ~09:35 UTC 31/10, mình đã chạy thử `tools/v3/r6_report.py --n-boot 20` trên dữ liệu **chưa đủ** để kiểm code: (a) mới có 83.968/85.176 frame, chưa có (b-old) và V3_gray. Kết quả ghi vào thư mục nháp, không vào `r6_full`. Lần chạy đó đã cho mình thấy số sơ bộ trên test của V3, (b), (interim) và (a): AUC, FPR, và GATE R6 sơ bộ không đạt vì điều kiện FPR trên CDF và DFDCP.
- Vì vậy PREREG **không** đặt luật mới cho các phần đó. Định nghĩa GATE R6 và các chỉ số lấy nguyên văn từ hai nguồn có trước lần chạy thử:
  - `ROOT_FIX_PLAN.md`: R6 và R6 "Yêu cầu bổ sung" (người dùng, 30/10).
  - Code đã commit `e7377b8` (30/10 14:59 UTC, `tools/v3/r6_report.py`) và `013754e` (30/10 14:49 UTC, `tools/eval/stats.py`).
- Luật mới duy nhất trong PREREG là tiêu chí "LoRA V3 dùng ảnh" (mục 5). Mình chưa thấy dữ liệu V3_gray nào: stage này chưa chạy.

## 1. Dữ liệu
- **TEST** (khoá tới R6): toàn bộ FFPP test (700 video / 22.388 frame), Celeb-DF-v2 test list (518 / 16.420), DFDCP test (654 video có frame / 17.222; 123/777 video không có frame trên đĩa) và DFF-test (3.332 ảnh, mỗi ảnh là 1 "video"). Bảng frame: `r6_full/frames_full.csv`.
- **DEV**: FFPP val (22.354 frame / 700 video) + DFF-val (3.460).
- Kiểm rò rỉ (mục 6) chạy **trước** khi tính. Nếu có trùng thì loại khỏi TEST, ghi số lượng; khi đó chỉ số tính trên TEST đã loại.

## 2. Hệ thống
(a) X2DFD gốc; (b-old) live trước audit, chỉ có trên TEST; (b) parity v2; (interim) Router v3 + D1; (V3) lineage V3; (V3_gray) V3 với ảnh xám 128 cùng prompt. Điểm = `fake_prob` từng frame trong `results_<hệ>.csv`; điểm của (b) và (interim) lấy từ `frames_full.csv` (`b_fake`, `interim_fake`).

## 3. Chỉ số
- **Chính:** frame AUC trên từng tập test (FFPP, CDF, DFDCP, DFF_test).
- **Phụ:** video AUC (video = trung bình `fake_prob` các frame); acc, FPR(thật), recall(giả) @0.5 và @t_dev.
- **CI:** mọi AUC có CI 95% bootstrap theo **video**: 1.000 lần, `numpy.random.default_rng(seed=0)`, lấy mẫu có hoàn lại các video trong từng tập (`stats.bootstrap_auc`), lấy phân vị 2,5 và 97,5.
- **Chênh lệch ghép cặp** (cùng mẫu bootstrap): V3−(b), V3−(a), V3−(interim), (b)−(b-old), V3−V3_gray, cho cả frame AUC và video AUC. Mỗi chênh lệch báo điểm ước lượng, CI 95%, và CI có loại trừ 0 hay không.

## 4. Ngưỡng
- t_dev = argmax(TPR − FPR) (Youden, `sklearn.metrics.roc_curve`) trên **DEV gộp** (FFPP val + DFF-val, mức frame), chọn riêng cho từng hệ có điểm DEV đầy đủ: (a), (b), (interim), V3. **Không dùng test.** (b-old) không có DEV nên chỉ báo @0.5.
- **Đổi ngưỡng web** (khỏi 0.5) chỉ khi trên DEV, ở t_dev, FPR giảm VÀ recall giảm ≤ 2 điểm so với 0.5.

## 5. GATE R6 (nguyên văn ROOT_FIX_PLAN, áp dụng máy móc)
> "GATE R6: V3 ≥ interim trên FFPP (không giảm > 0.5 điểm AUC) VÀ tốt hơn trên ít nhất 2/3 tập CDF/DFDCP/DFF; FPR thật không tăng > 1 điểm. Đạt → V3; không đạt → giữ interim, ghi số liệu."

Cách đọc đã có trong code `e7377b8` (trước mọi số test):
- AUC = frame AUC.
- (i) AUC(V3) − AUC(interim) trên FFPP ≥ −0.5 điểm.
- (ii) AUC(V3) > AUC(interim) (điểm ước lượng) trên ≥ 2 trong 3 tập CDF, DFDCP, DFF_test.
- (iii) FPR thật @0.5 của V3 − interim ≤ +1 điểm trên **mỗi** tập trong 4 tập test.
- Đạt khi cả (i), (ii), (iii) đều đúng. Quyết định theo **điểm ước lượng**; bên cạnh đó báo CI ghép cặp của từng thành phần và CI có loại trừ 0 không (chỉ để thông tin, không đổi quyết định).

**Tiêu chí "LoRA V3 dùng ảnh"** (luật mới, chưa thấy dữ liệu): frame AUC(V3) − AUC(V3_gray) > 0 với CI ghép cặp loại trừ 0 trên **cả 4** tập test. Ghi chú FE "không phân tích trực tiếp ảnh" chỉ được gỡ khi GATE R6 đạt **VÀ** tiêu chí này đạt.

## 6. Kiểm rò rỉ (trước khi tính)
So mỗi tập TEST với: SFT V3 (30.000), dữ liệu train teacher/Router V3, dữ liệu fit calibrator V3. Có thể thì so thêm lineage (b)/(interim): SFT FINAL_CLEAN, Router v2/v3 train, calibrator v2/D1.
- Theo **video_id** (FFPP: số thứ tự video nguồn/đích; CDF: id + số video; DFDCP: tên video; DFF: tên ảnh) và theo **danh tính** nếu tập có (CDF `idX`, FFPP id nguồn).
- Theo **sha256 nội dung ảnh**.
- Báo số trùng từng cặp.

## 7. Báo cáo
- Mọi hệ × mọi tập, kể cả chỗ V3 thua.
- (b) vs (b-old): số giá trị khác nhau, tỉ lệ hoà điểm, chênh lệch AUC ghép cặp.
- Histogram Router (mô tả).
- test_imgs: bảng từng ảnh (chỉ báo cáo).
- demo_imgs: có cả ca SAI.
- sha256 mọi CSV kết quả và script phân tích ghi vào PROGRESS và REPORT.
- **Kiểm chéo độc lập:** một subagent chỉ nhận CSV thô + PREREG này, tự viết code sklearn tính lại bảng chỉ số chính. Lệch > 1e-4 thì tìm nguyên nhân trước khi báo cáo.
