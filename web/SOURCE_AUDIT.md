# BÁO CÁO KHẢO SÁT MÃ NGUỒN VÀ TIỀN THỰC THI (SOURCE AUDIT REPORT)
## SPRINT 0 — DỰ ÁN ROUTER4 × X²-DFD WEBSITE DEMO

> **Ngày thực hiện:** 24/09/2026  
> **Căn cứ tài liệu:** [IMPLEMENTATION_PLAN_ROUTER4_X2DFD_WEBSITE_V3.md](file:///d:/demomain/router4-x2dfd-demo-main/IMPLEMENTATION_PLAN_ROUTER4_X2DFD_WEBSITE_V3.md), `HANDOFF_ROUTER4_X2DFD_WEBSITE_V3.md`  
> **Mục đích:** Xác minh tính sẵn sàng, đường dẫn, hàm suy luận, quy ước điểm số và các điểm chặn kỹ thuật từ mã nguồn thực tế tại `source_snapshot/` và `runtime_context/`.

---

## 1. Kết quả Audit 4 Expert (B, D, F, T)

### 1.1. Blending Expert
- **Đường dẫn mã nguồn:** `source_snapshot/projects/X2DFD/src/blending/detector.py`
- **Class/Hàm chính:** `BlendingDetector.infer(image_paths, batch_size=64)`
- **Kiến trúc mô hình:** Tạo qua `timm.create_model(model_name, pretrained=False, num_classes=num_class)`
- **Tiền xử lý ảnh:** `Resize((img_size, img_size))` -> `ToTensor()` -> `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`
- **Quy ước điểm số (Score Convention):**
  ```python
  outputs = self.model(batch)
  predictions = torch.nn.functional.softmax(outputs, dim=-1)
  fake_scores = predictions[:, 1].detach().cpu().tolist()  # Index 1 là xác suất FAKE
  ```
  -> **Xác nhận:** `score = predictions[:, 1]` đại diện trực tiếp cho `P(fake)`.
- **Trạng thái:** **VERIFIED (Có mã nguồn đầy đủ trong snapshot)**.

### 1.2. Diffusion Expert
- **Đường dẫn mã nguồn:** `source_snapshot/projects/X2DFD/src/diffusion/detector.py` (bọc `src/diffusion/core.py` - Aligner)
- **Class/Hàm chính:** `DiffusionDetector.infer(image_paths, batch_size=256)`
- **Kiến trúc:** Bọc Aligner sub-model (đọc config từ `{weights_dir}/{model}/config.yaml`)
- **Quy ước điểm số (Score Convention):** Trả về `{path: {"score": float}}`. Điểm trích xuất từ Aligner score.
- **Trạng thái:** **VERIFIED (Có mã nguồn wrapper và core trong snapshot)**.

### 1.3. Frequency Expert (DFFreq)
- **Đường dẫn mã nguồn:** `source_snapshot/projects/DFFreq-main/` và wrapper tại `source_snapshot/server_wrappers/router8_x2crop/run_frequency.py`
- **Kiến trúc mô hình:** `resnet50(num_classes=1)`
- **Checkpoint dự kiến:** `"checkpoints/model_epoch_last.pth"`
- **Tiền xử lý ảnh:** `CenterCrop(256)` -> `ToTensor()` -> `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`
- **Quy ước điểm số (Score Convention):**
  ```python
  scores = torch.sigmoid(model(images)).flatten().cpu().tolist()
  ```
  -> **Xác nhận:** Điểm số qua hàm `torch.sigmoid()`, thuộc khoảng $[0.0, 1.0]$, đại diện cho xác suất fake.
- **Trạng thái:** **VERIFIED (Mã nguồn và wrapper đầy đủ trong snapshot)**.

### 1.4. Texture Expert (Global Texture Enhancement / Gram-Net)
- **Đường dẫn mã nguồn:** `source_snapshot/projects/Global_Texture_Enhancement_for_Fake_Face_Detection_in_the-Wild/` và wrapper tại `source_snapshot/server_wrappers/router8_x2crop/run_texture.py`
- **Kiến trúc mô hình:** Gram-Net ResNet18
- **Tiền xử lý ảnh:** `cv2.imread` -> Transpose `(2, 0, 1)` -> `torch.from_numpy`
- **Quy ước điểm số (Score Convention - ĐẶC BIỆT LƯU Ý):**
  ```python
  # Gram-Net original: class 0 = fake!
  if output.ndim == 2 and output.shape[1] >= 2:
      score = F.softmax(output, dim=1)[:, 0]  # Index 0 là FAKE!
  else:
      score = torch.sigmoid(output.reshape(-1))
  ```
  -> **Xác nhận:** Đối với Texture Expert, **class 0 mới là FAKE**! Cần giữ nguyên quy ước này khi lấy raw score để không bị đảo chiều nhãn.
- **Trạng thái:** **VERIFIED (Mã nguồn và wrapper đầy đủ trong snapshot)**.

---

## 2. Kết quả Audit Router4 và Calibrators

### 2.1. Router4 Classifier
- **Mã nguồn tích hợp:** `source_snapshot/projects/x2dfd_router_integration/build_router_moe_cache_full.py`
- **Kiến trúc:** `torchvision.models.efficientnet_b0` với `classifier[1] = nn.Linear(nf, 4)`.
- **Thứ tự 4 Expert:**
  `EXPERTS = ['blending', 'diffusion', 'frequency', 'texture']`  
  `ALIASES = {'blending': 'Blending', 'diffusion': 'Diffusion', 'frequency': 'Frequency', 'texture': 'Texture'}`
- **Tiền xử lý ảnh:** `Resize(256)` -> `CenterCrop(224)` -> `ToTensor()` -> `Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])`.
- **Đầu ra:**
  ```python
  prob = torch.softmax(model(x), dim=1)
  top2 = prob.topk(2, dim=1)
  idx = top2.indices[:, 0]        # selected_expert index
  conf = top2.values[:, 0]        # router_confidence
  margin = top2.values[:, 0] - top2.values[:, 1]
  ```
- **Checkpoint đường dẫn server:** `/home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919/02_router/final_checkpoint/best.pt`
- **Trạng thái:** **VERIFIED (Logic và cấu trúc rõ ràng trong mã nguồn)**.

### 2.2. Calibrators Bundle
- **Đường dẫn artifact server:** `/home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919/01_teacher/calibrators_FINAL_CLEAN.joblib`
- **Cấu trúc Bundle:** Dictionary chứa `bundle['experts']` với 4 khóa `blending`, `diffusion`, `frequency`, `texture`.
- **Cơ chế suy luận:**
  ```python
  calibrated = float(calibrators[expert].predict([float(raw)])[0])
  ```
  Calibrator nhận đầu vào là mảng chứa `raw_score` của chính expert đó và trả về `calibrated_score` (xác suất fake đã hiệu chuẩn).
- **Trạng thái:** **VERIFIED (Logic phân giải và gọi calibrator đã được chuẩn hóa)**.

---

## 3. Khảo sát Cơ chế Lấy Final Continuous Score & Explanation

### 3.1. Trích xuất Continuous Score từ LLaVA LoRA
- **Đường dẫn mã nguồn:** `source_snapshot/projects/X2DFD/utils/lora_inference.py` (hàm `single_image_infer_with_scores`)
- **Cơ chế hoạt động:**
  1. Mô hình LLaVA sinh token với `output_scores=True`, `return_dict_in_generate=True`, `num_beams=1`.
  2. Mã nguồn lấy logits tại đúng bước sinh nhãn (`label_step` nơi token đầu ra khớp với token ID của từ "real" hoặc "fake"):
     ```python
     real_tokens = tokenizer.encode("real", add_special_tokens=False)
     fake_tokens = tokenizer.encode("fake", add_special_tokens=False)
     real_id = int(real_tokens[0])
     fake_id = int(fake_tokens[0])
     ...
     pair_logits = torch.stack([logits[real_id], logits[fake_id]])
     pair_probs = torch.softmax(pair_logits, dim=0)
     real_prob = float(pair_probs[0].detach().cpu().item())
     fake_prob = float(pair_probs[1].detach().cpu().item())
     ```
  3. Hàm trả về dictionary:
     ```python
     out = {
         "answer": answer,
         "real_score": round(real_prob, 8),
         "fake_score": round(fake_prob, 8)
     }
     ```
- **Đánh giá kiểm tra:**
  - Logic trích xuất logits và pairwise softmax đã tồn tại trong mã nguồn.
  - Tuy nhiên, tính ổn định trên môi trường GPU live và hành vi khi model sinh token viết hoa (Real/Fake) hoặc từ chối trả lời cần được kiểm chứng probe trên server thực tế.
- **Trạng thái:** **VERIFIED VỀ MẶT CODE / CHỜ PROBE TEST TRÊN SERVER GPU**.

### 3.2. Khả năng sinh Giải thích (AI Explanation)
- **Mã nguồn:** Giá trị `answer` được giải mã từ `generation_output.sequences`.
- **Thực tế:** Trong cấu hình WFS chuẩn của X²-DFD, câu hỏi là:  
  `"<image>\nIs this image real or fake? And the {alias} score is {score}."`  
  Nếu mô hình được tinh chỉnh chỉ để trả lời nhãn trực tiếp ("real" / "fake"), `answer` sẽ chỉ chứa nhãn. Nếu mô hình được huấn luyện sinh văn bản giải thích, `answer` sẽ có đoạn văn tự nhiên sau nhãn.
- **Quy tắc triển khai:**
  - Nếu `answer` chứa nội dung giải thích có nghĩa (độ dài $> 2$ từ sau khi tách nhãn) $\to$ Backend trả về `final.explanation: str`.
  - Nếu `answer` chỉ là nhãn hoặc không có giải thích $\to$ Backend trả về `final.explanation: null`.
  - Frontend bắt buộc **hoàn toàn ẩn khối giải thích** nếu giá trị này là `null`.
- **Trạng thái:** **ĐÃ THIẾT LẬP QUY TẮC AN TOÀN FAIL-CLOSED**.

---

## 4. Khảo sát Quy ước Prompt WFS (Weighted Forensic Scoring)

- **Mã nguồn huấn luyện:** `source_snapshot/final_clean_integration/03_wfs/build_router4_train_wfs_FINAL.py`
- **Bằng chứng từ dòng 355–371:**
  ```python
  # Primary experiment:
  # heterogeneous experts use a unified calibrated P(fake).
  df["selected_score"] = df["selected_cal_score"]

  df["wfs_text"] = df.apply(
      lambda r: f"And the {r['selected_alias']} score is {float(r['selected_score']):.3f}.",
      axis=1
  )
  ```
- **Kết luận xác minh:**
  1. LoRA adapter chính thức (`router4_x2dfd_FINAL_CLEAN`) được huấn luyện bằng **`selected_cal_score`** (điểm đã qua Calibrator).
  2. Định dạng câu hỏi WFS chuẩn xác:
     ```text
     <image>
     Is this image real or fake? And the {alias} score is {calibrated_score:.3f}.
     ```
     Trong đó:
     - `{alias}` là một trong 4 từ viết hoa: `Blending`, `Diffusion`, `Frequency`, `Texture`.
     - `{calibrated_score:.3f}` là điểm sau khi chạy qua Calibrator, làm tròn 3 chữ số thập phân.
- **Trạng thái:** **VERIFIED (Đã có chứng cứ mã nguồn dứt khoát)**.

---

## 5. Khảo sát Quy trình Video (Trích xuất Frame & Tổng hợp Điểm)

### 5.1. Giao thức Trích xuất Frame Video
- **Mã nguồn tham chiếu:** `source_snapshot/server_wrappers/router8_x2crop/crop_ffpp_x2dfd.py`
- **Giao thức DeepfakeBench:**
  ```python
  dbp.video_manipulate(
      movie_path=video,
      mask_path=None,
      dataset_path=c23root,
      mode="fixed_num_frames",
      num_frames=32,
      stride=10,
  )
  ```
- **Phân tích:** Trong benchmark server, video được trích xuất cố định 32 frame cách đều theo thời gian từ DeepfakeBench.
- **Trạng thái:** **VERIFIED GIAO THỨC TRÍCH XUẤT (32 frames uniform sampling)**.

### 5.2. Thuật toán Tổng hợp Video (Video Aggregation)
- **Khảo sát:** Rà soát toàn bộ thư mục `source_snapshot/projects/X2DFD` và `source_snapshot/final_clean_integration/`.
- **Hiện trạng:**
  - Benchmark của bài báo đánh giá trên tập ảnh/frame trích xuất sẵn (`test/frames/<video>/*.png`).
  - Trong source snapshot **chưa có file script chính thức nào định nghĩa một công thức toán học duy nhất** (ví dụ: mean, trimmed mean, hay threshold voting) để tổng hợp ra 1 điểm duy nhất cho video ở chế độ real-time inference.
  - File `build_protocol_locked_eval.py` lưu ý: *"Frame-level scores and video-level aggregated scores must be reported separately."* nhưng không kèm hàm aggregate mã nguồn đóng gói.
- **Quyết định Kỹ thuật (Fail-Closed Gate):**
  - Đánh dấu **BLOCKED VIDEO FINAL VERDICT**.
  - Trong hệ thống website:
    - Vẫn thực hiện trích xuất frame và chạy toàn bộ pipeline AI cho từng frame độc lập.
    - Trả về danh sách chi tiết `frames[]` với đầy đủ timestamp, expert được chọn và điểm từng frame.
    - Trạng thái trả về: `status: "blocked"`, `error_code: "VIDEO_AGGREGATION_UNVERIFIED"`, `final: null`.
    - UI hiển thị rõ: `KẾT LUẬN VIDEO CHƯA XÁC MINH (THIẾU THUẬT TOÁN TỔNG HỢP GỐC)`, không tự ý tính mean để bịa ra verdict.
- **Trạng thái:** **BLOCKED (Ghi nhận ngoại lệ kỹ thuật, fail-closed an toàn)**.

---

## 6. Môi trường Thực thi & Python Wrappers trên Server

- `X2DFD_PYTHON`: `/home/aiotlab/miniconda3/envs/X2DFD/bin/python` (chạy X²-DFD LLaVA + LoRA)
- `runtime_clean_v2/x2python`: Script wrapper cô lập LD_LIBRARY_PATH và CUDA devices.
- `runtime_clean_v2/freqpython`: Script wrapper chạy DFFreq.
- `runtime_clean_v2/texpython`: Script wrapper chạy Gram-Net Texture.
- **Quy tắc tích hợp Backend:** Backend FastAPI chạy độc lập trong môi trường web riêng; gọi các AI wrappers thông qua `SubprocessRunner` an toàn (`shell=False`, timeout, kill process tree).

---

## 7. Kết luận Sprint 0

1. **Pipeline Ảnh (Image)**: Đầy đủ 100% mã nguồn và quy ước khoa học (Router4 EfficientNet-B0 -> Expert B/D/F/T -> Calibrator -> WFS prompt với calibrated score -> LLaVA LoRA -> Continuous Pairwise Softmax). Sẵn sàng lập trình Live Worker!
2. **Pipeline Video (Video)**: Đã xác minh giao thức trích xuất frame; chưa có thuật toán tổng hợp video từ mã nguồn nên kích hoạt cơ chế an toàn `status: "blocked"` và hiển thị per-frame timeline.
3. Toàn bộ phát hiện đã được ánh xạ vào file ma trận kiểm chứng [web/VERIFICATION_MATRIX.md](file:///d:/demomain/router4-x2dfd-demo-main/web/VERIFICATION_MATRIX.md).
