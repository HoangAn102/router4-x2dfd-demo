# MA TRẬN KIỂM CHỨNG TÍNH SẴN SÀNG (VERIFICATION MATRIX)
## DỰ ÁN ROUTER4 × X²-DFD WEBSITE DEMO

> **Phiên bản:** 1.0 (Sau Sprint 0 Source Audit)  
> **Ký hiệu trạng thái:**  
> - 🟢 **VERIFIED**: Đã có mã nguồn, cấu hình, logic kiểm chứng rõ ràng trong repository.  
> - 🟡 **UNVERIFIED**: Đã có mã nguồn nhưng cần chạy probe test trên server GPU để kiểm tra độ ổn định thực tế.  
> - 🔴 **BLOCKED**: Chưa có mã nguồn hoặc chưa có thuật toán xác minh từ nghiên cứu; kích hoạt cơ chế fail-closed.

---

## 1. Bảng Ma trận Kiểm chứng Từng Thành phần

| Hạng mục / Module | Thành phần kiểm tra | Trạng thái | Bằng chứng từ Mã nguồn / Server | Hành động Kỹ thuật tương ứng |
|:---|:---|:---:|:---|:---|
| **Expert 1: Blending** | Mã nguồn suy luận | 🟢 VERIFIED | `source_snapshot/projects/X2DFD/src/blending/detector.py` | Sử dụng hàm `infer()`, lấy `predictions[:, 1]` làm fake score. |
| **Expert 2: Diffusion** | Mã nguồn suy luận | 🟢 VERIFIED | `source_snapshot/projects/X2DFD/src/diffusion/detector.py` | Bọc `Aligner` core, lấy score từ sub-model config. |
| **Expert 3: Frequency** | Mã nguồn & Wrapper | 🟢 VERIFIED | `source_snapshot/server_wrappers/router8_x2crop/run_frequency.py` | `resnet50(num_classes=1)`, lấy `torch.sigmoid()`. |
| **Expert 4: Texture** | Mã nguồn & Wrapper | 🟢 VERIFIED | `source_snapshot/server_wrappers/router8_x2crop/run_texture.py` | Gram-Net ResNet18, **class 0 = fake** (`F.softmax[:, 0]`). |
| **Router4** | Classifier 4 nhánh | 🟢 VERIFIED | `source_snapshot/projects/x2dfd_router_integration/build_router_moe_cache_full.py` | EfficientNet-B0, trích xuất `selected_expert`, `confidence`, `margin`. |
| **Calibrators** | Bundle 4 calibrator | 🟢 VERIFIED | `outputs/.../01_teacher/calibrators_FINAL_CLEAN.joblib` | Áp dụng `calibrator[expert].predict([raw_score])`. |
| **WFS Prompt** | Định dạng câu hỏi | 🟢 VERIFIED | `source_snapshot/final_clean_integration/03_wfs/build_router4_train_wfs_FINAL.py` | Khóa định dạng: `"<image>\nIs this image real or fake? And the {alias} score is {calibrated_score:.3f}."` |
| **Continuous Score** | Pairwise Softmax Logits | 🟡 UNVERIFIED | `source_snapshot/projects/X2DFD/utils/lora_inference.py` (`single_image_infer_with_scores`) | Logic code đã có; cần chạy probe test trên server GPU để kiểm chứng. Áp dụng fail-closed toàn diện. |
| **AI Explanation** | Khả năng sinh giải thích | 🟢 VERIFIED | `lora_inference.py` giải mã sequences | Nếu có text tự nhiên -> `explanation: str`. Nếu chỉ có nhãn -> `explanation: null`. UI ẩn khi null. |
| **Video Extraction** | Giao thức trích xuất frame | 🟢 VERIFIED | `source_snapshot/server_wrappers/router8_x2crop/crop_ffpp_x2dfd.py` | Trích xuất cách đều 32 frame tuần tự theo chuẩn DeepfakeBench. |
| **Video Aggregation** | Thuật toán tổng hợp điểm | 🔴 BLOCKED | Không có script tính điểm tổng hợp duy nhất trong snapshot | **KHÔNG TỰ BỊA ĐIỂM/VERDICT**. Trả `status: "blocked"`, `final: null`, hiển thị frame-level timeline. |
| **Môi trường Conda** | Runtime wrappers | 🟢 VERIFIED | `source_snapshot/runtime_wrappers/runtime_clean_v2/` (`x2python`, `freqpython`, `texpython`) | Gọi qua `SubprocessRunner` cô lập, `shell=False`, timeout đa tầng. |
| **Điều phối GPU** | Chống xung đột OOM | 🟢 VERIFIED | `RUNBOOK.md` & `web/backend/app/utils/gpu_lock.py` | Cấu hình 1 worker Uvicorn + Semaphore(1) + GpuLock (filelock). |
| **Dọn dẹp File Tạm** | Quản lý ổ cứng | 🟢 VERIFIED | `web/backend/app/services/file_manager.py` | Cơ chế Hai Tầng: `finally` dọn dẹp theo request + Startup sweep dọn rác mồ côi $> 1$h. |

---

## 2. Kết luận Điều kiện Chuyển Giai đoạn (Sprint Gate 0 -> 1)
- [x] Đã khảo sát và xác minh đầy đủ 4 expert, Router4, Calibrator và quy ước prompt WFS.
- [x] Đã xác định rõ điểm chặn (Blocker) của thuật toán Video Aggregation và thiết kế cơ chế fail-closed `status: "blocked"`.
- [x] Đã thiết lập quy tắc nghiêm ngặt cho `final.explanation`.
- [x] **Đủ điều kiện chuyển sang Sprint 1: Lập trình Backend Foundation, Async Video Job Engine và Schemas.**
