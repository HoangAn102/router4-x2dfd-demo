# PREREG_E2: V3 so với X2DFD gốc theo đúng giao thức báo cáo cũ (Bảng 2–4)

Viết: 01/11/2026, khoảng 08:20 UTC, **trước khi chạy** bất kỳ phần nào của E2 (giờ commit và sha256: xem PROGRESS và bản sao `docs/PREREG_E2.md`). Sau khi commit, **không đổi** luật, chỉ số, ngưỡng hay định nghĩa hệ. Mọi phân tích ngoài danh sách này được ghi nhãn **"thăm dò (post-hoc)"**.

## 0. Những gì đã thấy trước khi viết (công khai)
- **Kết quả cũ đã thấy:** số của báo cáo cũ (Original, Router4 cũ, CI) ở `ROUTER4_CURRENT_FROZEN_EXTERNAL_20261010/outputs/comparison.md`.
- **Kết quả R6 đã thấy** của (a) và V3-P0 trên CDF, DFDCP và FFPP val. Các frame này **trùng 100%** với manifest E2 CDF, DFDCP val và FFPP development; nhãn trùng 100%.
- **Chưa tính gì** cho V3-P1 và ORACLE-blending trên bất kỳ tập nào. Chưa tính gì trên DFDC (132.116 frame không có trong R6).
- **Router confidence:** chưa tính phân bố trên các tập E2 test.
- **Ngưỡng 0.30 của P1:** đã có sẵn trong `config.py` (`ROUTER_LOW_CONFIDENCE`, đặt ngày 26/10 từ FFPP val: khoảng 18% đáy). P1 do người dùng chốt trước, **không** chọn trên test.

## 1. Dữ liệu (dùng NGUYÊN manifest, không lấy mẫu lại)

| Tập | Frame / video | Manifest | sha256 |
|---|---|---|---|
| Celeb-DF-v2 | 16.420 / 518 | `…/runs/ROUTER4_CURRENT_FROZEN_EXTERNAL_20261010/manifest.csv` | `33c676d0…35fc` |
| DFDC | 132.116 / 4.704 (đã loại 296 metadata key) | như trên | như trên |
| DFDCP val (author-arrangement) | 17.222 / 652 | như trên | như trên |
| FFPP development | 22.354 / 700 | `…/runs/ROUTER4_IMPROVEMENT_RESEARCH_V1/internal_ffpp_val_manifest.csv` | `b2cc19b0…a529` |

- Thư mục gốc: `/home/aiotlab/router4_x2dfd_FULL_SOURCE_20260924_054600/replication_author_audit/runs`.
- Cả 4 tập dùng khoá video `official_video_key`.
- **FFPP development** chính là FFPP val (DEV của R6). Vì vậy đây là tập **trong miền, không độc lập**, và được báo cáo như báo cáo cũ.
- Dùng thêm từ R6 (cho luật P1 và báo cáo):
  - **DFF_test:** 3.331 frame, sau khi loại 1 frame rò rỉ ở R6.
  - **FFPP test R6:** chỉ để thông tin.
- **DEV (chọn t_dev):** FFPP val + DFF-val, như R6.
- **Predictions cũ dùng lại**, không chạy lại:
  - Router4 cũ: `router4_predictions.csv` (sha `757d586e…`). FFPP dev: `router4_cal_internal_predictions.csv` (sha `613338f6…`).
  - Original cũ: `original_predictions.csv` (sha `4458182c…`), chỉ dùng cho kiểm tái lập.

## 2. Hệ thống

| Hệ | Định nghĩa |
|---|---|
| **(a) X2DFD gốc** | Câu hỏi do runner tác giả tạo (`_build_conversations_multi`, expert của tác giả, cwd X2DFD) + LoRA tác giả `llava-v1.5-7b-lora-[ble-diff]`. Read-out như R6 (`r6_batched.py --system a`). |
| **V3-P0** | Lineage V3 đang deploy: Router V3 top-1 → calibrator V3 → LoRA V3, prompt `<image>\n…`, read-out teacher-forced. |
| **V3-P1** | Như P0, nhưng khi `max softmax(Router V3) < 0.30` thì expert := **blending** (calibrator V3 blending). Chốt sẵn, không chọn trên test. |
| **ORACLE-blending** | Như P0, nhưng expert := blending cho **mọi** frame. Chỉ để chẩn đoán, **không deploy**. |
| **Router4 cũ** | Predictions của báo cáo cũ (CDF/DFDC/DFDCP + FFPP dev). |
| **Expert đứng riêng** | AUC của điểm raw từng expert (4 expert, wrapper live) trên từng tập. Chỉ mô tả. |

- Frame trùng R6 tái dùng kết quả R6: (a) `results_a.csv`, V3-P0 `results_V3.csv`.
- P1 và ORACLE chỉ chấm lại những frame có expert khác P0; frame khác giữ điểm P0 (cùng prompt ⇒ cùng điểm).
- DFDC chạy mới: raw 4 expert, Router V3 (xác suất), calibrator V3, LLaVA V3, câu hỏi tác giả + LoRA tác giả.

## 3. Chỉ số và giao thức (đúng `ROUTER4_CURRENT_FROZEN_EXTERNAL_20261010/finalize.py`, sha `0752f21a…`)
- **AUC:** frame và video AUC theo từng tập. Video = trung bình fake_prob các frame cùng `official_video_key`. Fake là lớp dương. AUC dạng weighted Mann–Whitney (`weighted_auc` của finalize.py).
- **Delta:** hệ − Original, đơn vị điểm phần trăm.
- **CI ghép cặp:** 2.000 lần bootstrap. Với mỗi (tập, mức), tạo mới `np.random.default_rng(20261001)`; mỗi lần lấy `multiplicity = bincount(rng.integers(0, nv, nv))` trên các video đã sắp xếp. Frame có trọng số theo video của nó; video có trọng số multiplicity. Lấy phân vị 2,5 và 97,5; bỏ các lần bootstrap chỉ có một lớp.
- **Kiểm code bootstrap:** trước khi dùng, code của mình phải tái tạo đúng CI cũ của Router4 cũ từ predictions cũ (sai khác ≤ 0.001 pp).
- **Các cặp:**
  - Chính: V3-P0 − Original, V3-P1 − Original.
  - Phụ: V3-P1 − V3-P0, ORACLE − V3-P0, V3-P0 − Router4 cũ.
- **Ngưỡng:** FPR thật và recall giả @0.5 và @t_dev.
  - t_dev = Youden trên DEV (FFPP val + DFF-val) của từng hệ.
  - (a) và V3-P0 lấy t_dev từ R6: 0.906 và 0.4610168.
  - P1 và ORACLE tính t_dev trên DEV với điểm của chính hệ đó.

## 4. Kiểm tái lập (trước khi so)
- (a) chạy lại phải khớp Original cũ, frame / video AUC:
  - CDF 0.869587 / 0.916755
  - DFDC 0.817831 / 0.834027
  - DFDCP val 0.876608 / 0.895869
- Lệch > 0.005 ở bất kỳ ô nào ⇒ tìm nguyên nhân (manifest, crop, cách đọc điểm) và ghi rõ **trước** khi so.
- Bảng chính dùng **(a) chạy lại**. Thăm dò: bảng song song dùng Original cũ.

## 5. Luật deploy P1 (máy móc, theo điểm ước lượng)
**Deploy P1** khi cả hai điều kiện sau đều đúng:
1. Frame AUC(P1) ≥ frame AUC(P0) trên ít nhất 2 trong 3 tập ngoài miền: CDF, DFDC, DFDCP val.
2. Frame AUC(P1) − frame AUC(P0) ≥ −0.5 điểm trên **FFPP development** VÀ trên **DFF_test**.

- Đạt ⇒ deploy P1:
  - `ROUTER_FALLBACK_POLICY=low_conf_blending`, ngưỡng conf 0.30.
  - `DECISION_THRESHOLD` = t_dev của P1 trên DEV (cùng cách chủ dự án đã chọn cho V3).
  - Smoke, tag `final-v3-p1`.
- Không đạt ⇒ giữ V3-P0.
- FFPP test (R6) chỉ báo, không thuộc luật.

## 6. Kiểm rò rỉ (trước khi tính)
- Như R6 (`tools/v3/r6_leak_check.py`): path, video_id, danh tính, sha256.
- Áp cho DFDC + CDF + DFDCP val + FFPP development, so với mọi nguồn train/fit của V3, interim và (b).
- Có trùng ⇒ loại khỏi tập đó và ghi rõ.
- FFPP development vốn là DEV nên không phải test; vẫn kiểm.

## 7. Báo cáo
- **Bảng theo định dạng Bảng 4 cũ** (Dataset | Level | Original | Router4 | Delta pp | Paired 95% CI pp): một bảng cho V3-P0, một bảng cho V3-P1.
- **Bảng "trước/sau sửa":** Original | Router4 CŨ | V3-P0 | V3-P1, kèm Δ so Original và CI ghép cặp (giao thức cũ). CI của Router4 cũ tính lại từ predictions cũ.
- **Mô tả:** phân bố expert theo tập (P0/P1), AUC từng expert đứng riêng, ORACLE (chẩn đoán).
- **Kiểm chéo độc lập:** subagent chỉ nhận CSV thô + PREREG này, tự tính lại bảng chính. Lệch > 1e-4 ⇒ tìm nguyên nhân.
- sha256 mọi CSV và script ghi vào PROGRESS/REPORT.
