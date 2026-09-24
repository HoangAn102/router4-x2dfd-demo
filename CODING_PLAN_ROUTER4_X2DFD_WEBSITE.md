# KẾ HOẠCH LẬP TRÌNH CHI TIẾT (ACTIONABLE CODING PLAN - FINAL V5)
## DỰ ÁN: WEBSITE DEMO PHÁT HIỆN DEEPFAKE ROUTER4 × X²-DFD (IMAGE & VIDEO)

> **Tài liệu căn cứ:** [IMPLEMENTATION_PLAN_ROUTER4_X2DFD_WEBSITE_V3.md](file:///d:/demomain/router4-x2dfd-demo-main/IMPLEMENTATION_PLAN_ROUTER4_X2DFD_WEBSITE_V3.md) và tài liệu bàn giao `HANDOFF_ROUTER4_X2DFD_WEBSITE_V3.md`.  
> **Nguyên tắc chỉ đạo:** Không suy đoán thuật toán; fail-closed khi thiếu thông tin/score; audit trước khi code; tách bạch tuyệt đối giữa Mock Dev và Live AI; cô lập an toàn subprocess; chốt kiến trúc bất đồng bộ (Async Job + Polling) cho Video; chỉ hiển thị giải thích khi mô hình thực sự tạo ra; phân định rạch ròi giữa PASS toàn diện và BLOCKED được ghi nhận.

---

## 1. Sơ đồ Luồng Kỹ thuật Chuẩn & Các Bất biến Cốt lõi

### 1.1. Luồng suy luận Ảnh (Đồng bộ - Synchronous Pipeline)
Thời gian suy luận ảnh đơn lẻ là 1–3 giây, sử dụng kết nối HTTP đồng bộ:

```text
[Input Image]
      │
      ▼ (HTTP POST /api/v1/analyze/image)
1. Validation (MIME, Magic Bytes, Pixel Bounds <= 4096, Decode Test)
      │
      ▼
2. Router4 (EfficientNet-B0) ──► Chọn ĐÚNG 1 Expert trong {B, D, F, T} + router_confidence (q_i)
      │
      ▼
3. Selected Expert Engine ─────► Chạy duy nhất expert được chọn qua subprocess cô lập ──► Tạo raw_expert_score
      │
      ▼
4. Calibrator (Joblib) ────────► Áp dụng Calibrator riêng của expert đó ──► Tạo calibrated_expert_score
      │
      ▼
5. WFS Prompt Builder ─────────► Ghép prompt theo CẤU HÌNH ĐÃ AUDIT TẠI SPRINT 0
      │                          (Xác định dùng raw hay calibrated score dựa trên LoRA training config)
      ▼
6. X²-DFD (LLaVA-v1.5 + LoRA) ─► Suy luận sinh câu trả lời + Logits tại token nhãn thực tế
      │
      ▼
7. Continuous Score Extractor ─► Pairwise Softmax giữa token "real" và "fake" tại label step
      │
      ▼
8. Strict Fail-Closed Check ───► Kiểm tra toàn diện CẢ real_score VÀ fake_score:
      │                          Từ chối bool, kiểm tra float, math.isfinite(), 0 <= score <= 1, sum ≈ 1.0
      ▼
9. Explanation Gate ───────────► NẾU mô hình sinh văn bản giải thích thực sự -> explanation: str
                                 NẾU mô hình chỉ sinh nhãn/không có giải thích -> explanation: null (UI ẨN)
```

### 1.2. Luồng suy luận Video (Bất đồng bộ - Asynchronous Job & Polling Pipeline)
Do video cần trích xuất $N$ frame và chạy qua full pipeline từng frame nên thời gian xử lý có thể kéo dài (60s – 300s). **Chốt kiến trúc: Bất đồng bộ với Job ID và Polling** để loại bỏ 100% rủi ro đứt kết nối do HTTP timeout của browser/reverse proxy:

```text
[Browser Client]
      │
      ▼ 1. HTTP POST /api/v1/analyze/video (Upload file video)
[FastAPI Backend]
      │ - Validate media (codec, size <= 100MB, duration <= 60s)
      │ - Tạo `job_id` (UUIDv4), lưu video vào thư mục cách ly `uploads/{job_id}/`
      │ - Khởi chạy Background Worker Task
      ▼
      ◄── Trả về HTTP 202 Accepted:
          { "job_id": "...", "status": "processing", "poll_url": "/api/v1/analyze/video/jobs/{job_id}" }

[Client Polling Loop]
      │
      ▼ Gọi GET /api/v1/analyze/video/jobs/{job_id} mỗi 2.5 giây
[Job Manager Backend]
      │ - Trả về tiến trình thực tế:
      │   { "status": "processing", "progress": { "stage": "analyzing_frames", "current": 12, "total": 32 } }
      │
      ▼ (Khi Background Task hoàn tất:)
[Background Worker Task Pipeline]
      │
      ├─► 1. Frame Extraction (theo chuẩn audit Sprint 0) -> N frames
      │
      ├─► 2. Per-Frame Loop: Từng frame chạy Router4 -> Expert -> Calibrator -> WFS -> X²-DFD
      │      (Áp dụng FRAME_INFERENCE_TIMEOUT_SEC cho từng frame)
      │
      ├─► 3. Video Aggregation Gate:
      │      - Trường hợp A: ĐÃ XÁC MINH THUẬT TOÁN TỪ NGUỒN (Sprint 0 PASS)
      │        -> status: "ok", final: { verdict: "REAL"|"FAKE", score, explanation: str|null }, frames: [...]
      │        (Được tính là PASS TOÀN BỘ PIPELINE VIDEO)
      │
      │      - Trường hợp B: CHƯA XÁC MINH ĐƯỢC THUẬT TOÁN (Sprint 0 UNVERIFIED/BLOCKED)
      │        -> TUYỆT ĐỐI KHÔNG TỰ BỊA ĐIỂM HOẶC VERDICT CHO VIDEO!
      │        -> status: "blocked", error_code: "VIDEO_AGGREGATION_UNVERIFIED", final: null, frames: [...]
      │        (Ghi nhận là ngoại lệ kỹ thuật thiếu nguồn, KHÔNG coi là PASS)
      │
      └─► 4. Dọn dẹp thư mục tạm của job và lưu kết quả vào Memory Cache (TTL: 15 phút)

[Client Nhận Kết quả Cuối] ──► Dừng polling và render giao diện.
```

### 1.3. Các Bất biến Kỹ thuật Cốt lõi (Strict Invariants)
1. **Quy tắc Giải thích của AI (AI Explanation Invariant)**:
   - Chỉ hiển thị giải thích khi mô hình X²-DFD/LLaVA thực sự sinh ra văn bản giải thích tự nhiên.
   - Tuyệt đối **không tự bịa đặt, suy đoán hoặc dùng template cố định** (như dựa vào tên expert hay độ tự tin) để làm giải thích.
   - Nếu mô hình không sinh giải thích: Backend trả về `final.explanation: null`, Frontend **hoàn toàn ẩn khối giải thích** trên UI.
2. **Kiến trúc Video Bất đồng bộ (Async Job Invariant)**:
   - Endpoint upload: `POST /api/v1/analyze/video` trả về ngay HTTP 202 với `job_id`.
   - Endpoint kiểm tra trạng thái: `GET /api/v1/analyze/video/jobs/{job_id}` trả về stage thực tế và kết quả khi xong.
   - Loại bỏ hoàn toàn nguy cơ timeout của HTTP proxy khi xử lý video nhiều frame.
3. **Phân biệt Nghiệm thu Video**:
   - `PASS (Complete Video Pipeline)`: Video chạy thành công từ đầu đến cuối, mọi frame được phân tích và thuật toán tổng hợp được kiểm chứng tạo ra điểm cuối hợp lệ.
   - `BLOCKED (Documented Blocker - Missing Aggregation)`: Trích xuất và phân tích frame thành công nhưng tổng hợp video bị chặn do thiếu nguồn. Đây là **ngoại lệ kỹ thuật được ghi nhận**, không được coi là hoàn thành trọn vẹn pipeline video.
4. **Bộ kiểm tra Final Score Không Bỏ sót**:
   - Bắt buộc từ chối giá trị `bool` trước tiên:
     `if isinstance(s, bool) or not isinstance(s, (float, int)): raise ...`
   - Bắt buộc kiểm tra tính hữu hạn: `math.isfinite(real_f) and math.isfinite(fake_f)` (loại bỏ hoàn toàn `nan`, `inf`, `-inf`).
   - Bắt buộc kiểm tra miền xác suất: `0.0 <= real_f <= 1.0` và `0.0 <= fake_f <= 1.0`.
   - Bắt buộc kiểm tra chuẩn hóa: `abs((real_f + fake_f) - 1.0) <= 1e-4`.
   - Nếu bất kỳ điều kiện nào sai lệch -> Raise `FinalScoreUnavailableError` ngay lập tức, trả về `status: "error"`.
5. **Cơ chế Dọn dẹp File Tạm Hai Tầng (Dual-Layer Temp Cleanup)**:
   - Tầng 1: Dọn dẹp thư mục job ngay sau khi tác vụ hoàn tất.
   - Tầng 2: Quét dọn mồ côi (Orphan Stale Cleanup) tại FastAPI Lifespan startup và định kỳ mỗi giờ, quét và xóa các thư mục tạm có tuổi thọ $> 1$ giờ nhằm xử lý triệt để các trường hợp tiến trình bị `SIGKILL` hoặc OOM bất ngờ.
6. **Điều phối GPU Tránh Xung đột Đa Process**:
   - Trong môi trường triển khai server, cấu hình máy chủ web chạy **1 worker tiến trình duy nhất** (`uvicorn --workers 1`), kết hợp với `asyncio.Semaphore(1)` nội bộ.
   - Bổ sung `GpuLock` (filelock) bảo vệ VRAM nếu mở rộng đa tiến trình.
7. **Cơ chế Hiển thị Thumbnail Timeline An toàn**:
   - Frontend timeline sử dụng video player client tua đến `frame.timestamp_seconds` (`videoRef.currentTime = timestamp`), kết hợp nhúng micro-thumbnail dạng Base64 (`128x128 JPEG`) trực tiếp trong payload JSON, không lưu file ảnh tĩnh lâu dài trên server.

---

## 2. Cấu trúc Thư mục Toàn dự án (`web/`)

```text
d:\demomain\router4-x2dfd-demo-main\
├── web/
│   ├── backend/                      # FastAPI Backend
│   │   ├── app/
│   │   │   ├── __init__.py
│   │   │   ├── main.py               # FastAPI app, CORS, lifespan (quét rác stale temp)
│   │   │   ├── config.py             # Cấu hình Pydantic BaseSettings (RUN_MODE=mock|live)
│   │   │   ├── schemas/              # Pydantic Schemas đồng bộ với API Contract
│   │   │   │   ├── __init__.py
│   │   │   │   ├── common.py         # BaseResponse, ErrorResponse, HealthResponse
│   │   │   │   ├── image.py          # ImageAnalyzeResponse, RouterInfo, ExpertInfo
│   │   │   │   └── video.py          # VideoJobResponse, VideoJobStatusResponse, FrameResult
│   │   │   ├── routers/              # API Endpoints
│   │   │   │   ├── __init__.py
│   │   │   │   ├── health.py         # /health và /ready (kiểm tra worker thực tế)
│   │   │   │   ├── analyze_image.py  # POST /api/v1/analyze/image (Đồng bộ)
│   │   │   │   └── analyze_video.py  # POST /api/v1/analyze/video & GET /api/v1/analyze/video/jobs/{id}
│   │   │   ├── services/             # Media handling & validation
│   │   │   │   ├── __init__.py
│   │   │   │   ├── media_validator.py# Magic bytes, MIME, pixel/duration limits
│   │   │   │   ├── file_manager.py   # Temporary isolation directory, cleanup & sweep
│   │   │   │   ├── job_manager.py    # Quản lý hàng đợi job video, polling cache, TTL
│   │   │   │   ├── video_extractor.py# Frame extraction theo giao thức đã audit
│   │   │   │   └── video_aggregator.py# Video score aggregation theo giao thức đã audit
│   │   │   └── utils/
│   │   │       ├── __init__.py
│   │   │       ├── logger.py         # Logging có request_id
│   │   │       ├── gpu_lock.py       # Inter-process FileLock điều phối GPU đa process
│   │   │       └── subprocess_runner.py # Bộ thực thi subprocess an toàn (timeout, kill tree)
│   │   ├── requirements.txt
│   │   └── .env.example
│   │
│   ├── model_worker/                 # Tầng điều phối AI (Orchestration & Adapters)
│   │   ├── __init__.py
│   │   ├── base.py                   # BaseModelWorker Interface
│   │   ├── mock_worker.py            # Mock Worker (chỉ dùng cho RUN_MODE=mock)
│   │   ├── live_worker.py            # Live Worker điều phối pipeline thực tế
│   │   ├── router_client.py          # Router4 (EfficientNet-B0) Client
│   │   ├── expert_client.py          # Expert Runner (B, D, F, T) + Calibrator Client
│   │   ├── x2dfd_client.py           # X²-DFD (LLaVA + LoRA) Continuous Score & Explanation Client
│   │   └── worker_factory.py         # Factory cấp phát Worker theo RUN_MODE
│   │
│   ├── frontend/                     # React + Vite Application
│   │   ├── src/
│   │   │   ├── components/
│   │   │   │   ├── Header.jsx        # Navigation bar & Mock Mode Warning Banner
│   │   │   │   ├── Dropzone.jsx      # Upload ảnh/video (Drag & Drop, validate cùng limit BE)
│   │   │   │   ├── ProcessingStatus.jsx # Tiến trình thực tế (hiển thị frame X/N từ Polling)
│   │   │   │   ├── ImageResultCard.jsx  # Hiển thị kết quả ảnh, explanation (nếu có), 3 điểm số
│   │   │   │   ├── VideoResultCard.jsx  # Hiển thị kết quả video (tách rõ video vs frame, xử lý BLOCKED)
│   │   │   │   ├── FrameTimeline.jsx    # Timeline động (kết nối video preview seek hoặc micro-thumbnail)
│   │   │   │   └── ErrorModal.jsx       # Modal thông báo lỗi fail-closed kèm request_id
│   │   │   ├── services/
│   │   │   │   └── api.js            # API client (Axios/Fetch, hỗ trợ Polling loop cho Video)
│   │   │   ├── App.jsx               # Tab switcher (Image vs Video)
│   │   │   ├── main.jsx
│   │   │   └── index.css             # Dark modern theme, glassmorphism
│   │   ├── package.json
│   │   └── vite.config.js
│   │
│   ├── tests/                        # Toàn bộ kiểm thử tự động
│   │   ├── test_media_validator.py   # Test validation MIME, magic bytes, limits
│   │   ├── test_api_contracts.py     # Test API schema và fail-closed error
│   │   ├── test_mock_worker.py       # Test mock worker consistency
│   │   ├── test_score_semantics.py   # Test score normalization, bounds, từ chối bool
│   │   ├── test_explanation_rules.py # Test quy tắc explanation (null vs string)
│   │   ├── test_video_async_jobs.py  # Test quy trình tạo job và polling video
│   │   ├── test_subprocess_safety.py # Test timeout, process kill, error code handling
│   │   ├── test_cleanup_resilience.py# Test dọn dẹp file tạm thường quy và quét dọn mồ côi
│   │   └── test_browser_e2e.py       # Kịch bản kiểm thử E2E qua browser
│   │
│   ├── SOURCE_AUDIT.md               # Kết quả kiểm định mã nguồn AI tại Sprint 0
│   ├── VERIFICATION_MATRIX.md        # Ma trận kiểm chứng tính sẵn sàng của các module
│   ├── API_CONTRACT.md               # Đặc tả chi tiết Request/Response schemas
│   ├── RUNBOOK.md                    # Hướng dẫn vận hành, log, xử lý OOM, cấu hình 1 worker
│   └── README_WEB.md                 # Hướng dẫn cài đặt và sử dụng
```

---

## 3. Lộ trình Thực hiện theo Thứ tự Bắt buộc (Sprint 0 đến 7)

### SPRINT 0 — Source Audit & Runtime Preflight (Bắt buộc trước khi code AI)
> **Mục tiêu:** Xác minh 4 expert, checkpoint, môi trường Python, cơ chế trích xuất continuous score, khả năng sinh explanation, quy ước prompt WFS và quy trình video từ mã nguồn/server thực tế.

- [x] **Task 0.1: Audit 4 Expert (Blending, Diffusion, Frequency, Texture)**
  - Xác minh source code, entrypoint, checkpoint và hàm suy luận của từng expert trong `source_snapshot/` và trên server.
  - Xác định quy ước điểm số (hướng điểm, logit hay xác suất fake) và môi trường conda tương ứng (`freqpython`, `texpython`, v.v.).
  - Nếu thiếu bất kỳ expert nào: Đánh dấu **BLOCKED** trong ma trận kiểm chứng, không tự viết lại.
- [x] **Task 0.2: Audit Router4 và Calibrators**
  - Kiểm tra checkpoint `best.pt` của Router4: kiến trúc `efficientnet_b0`, số class đầu ra, thứ tự class.
  - Kiểm tra `calibrators_FINAL_CLEAN.joblib`: cấu trúc bundle, các hàm `.predict()`, đầu vào là raw score của expert tương ứng.
- [x] **Task 0.3: Audit Cơ chế Lấy Final Continuous Score & Explanation (Trạng thái: UNVERIFIED)**
  - Khảo sát mã nguồn `lora_inference.py` và `runner.py`: kiểm tra hàm `single_image_infer_with_scores`.
  - Xác minh vị trí token `real`/`fake`, cách lấy logits, áp dụng pairwise softmax.
  - **Kiểm tra trường Explanation**: Xác minh xem mô hình có thực sự sinh đoạn text giải thích sau nhãn không. Nếu chỉ sinh token nhãn, ghi nhận `explanation_supported = False`.
  - Viết probe script chạy thử nghiệm trên 1 ảnh kiểm thử trên server để xác nhận score và explanation thực tế.
- [x] **Task 0.4: Audit Quy ước Prompt WFS (Trạng thái: UNVERIFIED)**
  - Đối chiếu mã nguồn huấn luyện LoRA (`train/pipeline.py` hoặc các config LoRA) để xác định: LoRA được huấn luyện với **`raw_score`** hay **`calibrated_score`** của expert.
  - Khóa quy ước cấu trúc chuỗi prompt WFS: `"<image>\nIs this image real or fake? And the {alias} score is {score}."`
- [x] **Task 0.5: Audit Quy trình Video (Trích xuất & Tổng hợp - Trạng thái: UNVERIFIED)**
  - Tìm kiếm và xác minh quy trình trích xuất frame trong source (`source_snapshot/server_wrappers/router8_x2crop/` hoặc DeepfakeBench). Xác định cách chọn frame (số frame, stride, cách lấy landmark/face crop nếu có).
  - Tìm kiếm thuật toán tổng hợp video chính thức trong mã nguồn nghiên cứu.
  - **Quy tắc an toàn:** Nếu không tìm thấy thuật toán tổng hợp video chính thức từ source, đánh dấu **BLOCKED VIDEO FINAL VERDICT**; trong code xử lý theo Case B (§1.2).
- [x] **Đầu ra của Sprint 0:**
  - File `web/SOURCE_AUDIT.md`: Ghi lại chi tiết đường dẫn, hàm, quy ước điểm, prompt WFS, trạng thái explanation và các phát hiện.
  - File `web/VERIFICATION_MATRIX.md`: Bảng trạng thái `VERIFIED / UNVERIFIED / BLOCKED` cho từng thành phần.

---

### SPRINT 1 — Backend Foundation, Async Job Engine & API Schemas
> **Mục tiêu:** Xây dựng khung ứng dụng FastAPI hoàn chỉnh, cấu hình chế độ chạy duy nhất, phân tầng timeout, Job Manager cho video bất đồng bộ, Pydantic schemas và các endpoint kiểm tra sức khỏe.

- [x] **Task 1.1: Cấu hình Tập trung & Phân tầng Timeout (`web/backend/app/config.py`)**
  - Khai báo chế độ chạy duy nhất: `RUN_MODE: Literal["mock", "live"] = "mock"`.
  - Giới hạn file: `MAX_IMAGE_SIZE_MB = 20`, `MAX_VIDEO_SIZE_MB = 100`, `MAX_VIDEO_DURATION_SEC = 60`, `MAX_IMAGE_DIMENSION = 4096`.
  - Phân tầng Timeout:
    - `IMAGE_INFERENCE_TIMEOUT_SEC = 60`
    - `FRAME_INFERENCE_TIMEOUT_SEC = 45`
    - `VIDEO_TOTAL_TIMEOUT_SEC = 300`
  - Cấu hình Job Manager: `JOB_TTL_SECONDS = 900` (15 phút).
- [x] **Task 1.2: Pydantic Contract Schemas (`web/backend/app/schemas/`)**
  - `common.py`: `ErrorDetail`, `ErrorResponse` (bắt buộc có `stage`, `error_code`, `message`, `request_id`).
  - `image.py`: `ImageAnalyzeResponse` (bắt buộc trường `final.explanation: Optional[str] = None`).
  - `video.py`:
    - `VideoJobCreatedResponse`: `{ "job_id": str, "status": "processing", "poll_url": str }`.
    - `VideoJobStatusResponse`: Hỗ trợ cả 3 trạng thái: `processing` (kèm stage & progress count), `completed` (kèm `VideoAnalyzeResponse`), `blocked` (kèm `frames[]`, `final: null`, mã lỗi `VIDEO_AGGREGATION_UNVERIFIED`), và `failed` (kèm `ErrorDetail`).
- [x] **Task 1.3: Quản lý Job Bất đồng bộ (`web/backend/app/services/job_manager.py`)**
  - In-memory thread-safe dictionary lưu trạng thái và kết quả của các video job theo `job_id`.
  - Tự động xóa các job đã hết hạn TTL (> 15 phút).
- [x] **Task 1.4: Endpoints Kiểm tra Sức khỏe (`web/backend/app/routers/health.py`)**
  - `GET /api/v1/health`: Kiểm tra API process sống.
  - `GET /api/v1/ready`: **Kiểm tra khả năng hoạt động thực tế của Worker** (gọi `worker.check_readiness()`).
- [x] **Task 1.5: Main App & Quét Rác Lifespan (`web/backend/app/main.py`)**
  - Cấu hình CORS.
  - Lifespan: Quét dọn thư mục upload tạm mồ côi lúc khởi động (`cleanup_stale_temp_dirs()`).

---

### SPRINT 2 — Media Validation, Dọn dẹp Hai Tầng & Mock Worker (Dev Mode)
> **Mục tiêu:** Xây dựng tầng validation an toàn, quản lý dọn dẹp file tạm hai tầng và Mock Worker độc lập để phục vụ phát triển FE song song.

- [x] **Task 2.1: Bộ Xác thực Media Chặt chẽ (`web/backend/app/services/media_validator.py`)**
  - Kiểm tra Magic Bytes đầu file (chống đổi đuôi giả mạo).
  - Decode thử nghiệm ảnh bằng Pillow, kiểm tra resolution $\le 4096\times 4096$.
  - Kiểm tra video bằng OpenCV/ffprobe: thời lượng $\le 60\text{s}$, đọc được ít nhất frame đầu tiên.
- [x] **Task 2.2: Quản lý Thư mục Tạm Hai Tầng (`web/backend/app/services/file_manager.py`)**
  - Tạo thư mục tạm cách ly theo `request_id` / `job_id`.
  - Khối `finally` đảm bảo dọn sạch ngay sau khi xong; hàm `cleanup_stale_temp_dirs()` dọn sạch rác mồ côi lúc startup và định kỳ.
- [x] **Task 2.3: Mock Worker Chuyên dụng (`web/model_worker/mock_worker.py`)**
  - Chỉ kích hoạt khi `RUN_MODE=mock`.
  - Mô phỏng chính xác luồng dữ liệu, timing và hỗ trợ trả về `explanation: str` hoặc `null` để test FE.
  - Đính kèm cờ rõ ràng trong response: `provenance.mode = "mock_development"`.
- [x] **Task 2.4: Các Endpoint Phân tích Ban đầu (`web/backend/app/routers/`)**
  - `POST /api/v1/analyze/image`: Xử lý đồng bộ, trả về kết quả ảnh.
  - `POST /api/v1/analyze/video`: Nhận video, đẩy vào `JobManager` và trả về HTTP 202 kèm `job_id`.
  - `GET /api/v1/analyze/video/jobs/{job_id}`: Trả về trạng thái xử lý hiện tại hoặc kết quả hoàn tất.

---

### SPRINT 3 — Live Image Pipeline (Tích hợp AI Ảnh Thực tế)
> **Mục tiêu:** Kết nối Router4, 4 Expert, Calibrator và X²-DFD theo đúng thứ tự đã audit; thực thi subprocess an toàn; kiểm thử probe trên server GPU.

- [x] **Task 3.1: Router4 Client (`web/model_worker/router_client.py`)**
  - Nạp mô hình Router4 (`best.pt`, EfficientNet-B0).
  - Tiền xử lý ảnh chuẩn: `Resize(256) -> CenterCrop(224) -> ToTensor() -> Normalize`.
  - Trích xuất: Top-1 index (`selected_expert`), `router_confidence` ($q_i$), `margin`.
- [x] **Task 3.2: Expert Engine & Calibrator Client (`web/model_worker/expert_client.py`)**
  - Thực thi **duy nhất expert được chọn** qua wrapper môi trường thích hợp (`freqpython`, `texpython`, v.v.).
  - Sử dụng bộ thực thi an toàn `SubprocessRunner`: `shell=False`, timeout, cô lập env.
  - Nhận `raw_expert_score` từ expert.
  - **Gọi Calibrator tương ứng**: Nạp `calibrators_FINAL_CLEAN.joblib`, áp dụng `calibrator[selected_expert].predict([raw_score])` để tạo `calibrated_expert_score`.
- [x] **Task 3.3: WFS Prompt Builder**
  - Xây dựng prompt WFS theo cấu hình đã audit tại Task 0.4.
- [x] **Task 3.4: X²-DFD LLaVA + LoRA Client (`web/model_worker/x2dfd_client.py`)**
  - Gọi inference qua wrapper `x2python` thực thi `single_image_infer_with_scores`.
  - Trích xuất `real_score` và `fake_score` từ pairwise softmax logits tại label step.
  - **Kiểm định Fail-Closed Toàn diện (Chống Bug `bool` và `NaN`)**:
    ```python
    if isinstance(real_score, bool) or isinstance(fake_score, bool):
        raise FinalScoreUnavailableError("Score must not be a boolean value")
    if not isinstance(real_score, (float, int)) or not isinstance(fake_score, (float, int)):
        raise FinalScoreUnavailableError(f"Score type invalid: real={type(real_score)}, fake={type(fake_score)}")

    real_f = float(real_score)
    fake_f = float(fake_score)

    if not (math.isfinite(real_f) and math.isfinite(fake_f)):
        raise FinalScoreUnavailableError(f"Non-finite score encountered: real={real_f}, fake={fake_f}")

    if not (0.0 <= real_f <= 1.0 and 0.0 <= fake_f <= 1.0):
        raise FinalScoreUnavailableError(f"Score out of bounds [0, 1]: real={real_f}, fake={fake_f}")

    if abs((real_f + fake_f) - 1.0) > 1e-4:
        raise FinalScoreUnavailableError(f"Scores do not sum to 1.0: sum={real_f + fake_f}")
    ```
  - **Trích xuất Giải thích (Explanation Gate)**:
    - Nếu câu trả lời sinh ra từ LLaVA có phần giải thích tự nhiên hợp lệ -> gán `explanation = clean_explanation_text`.
    - Nếu không có hoặc chỉ sinh nhãn "real"/"fake" -> gán `explanation = None`. Tuyệt đối không sinh text giả.
- [x] **Task 3.5: Điều phối GPU & Tích hợp Live Worker (`web/model_worker/live_worker.py`)**
  - Cấu hình 1 worker uvicorn + `asyncio.Semaphore(1)` + `GpuLock` (filelock).
  - Không fallback sang mock khi lỗi.
- [x] **Task 3.6: Smoke Test Ảnh Thật trên Server**
  - Gửi 1 ảnh test thật qua HTTP endpoint `/api/v1/analyze/image`, kiểm tra log và response trả về đủ 3 điểm số và explanation hợp lệ (hoặc null).

---

### SPRINT 4 — Live Video Pipeline (Tích hợp AI Video Thực tế)
> **Mục tiêu:** Trích xuất frame và tổng hợp video theo đúng phương pháp đã xác minh từ Sprint 0; tích hợp Background Job và Polling.

- [x] **Task 4.1: Video Frame Extractor (`web/backend/app/services/video_extractor.py`)**
  - Trích xuất $N$ frame theo chuẩn audit Sprint 0.
  - Sinh micro-thumbnail (base64 JPEG $128\times 128$) cho từng frame nhúng vào payload JSON.
- [x] **Task 4.2: Background Execution & Tiến trình Polling**
  - Tác vụ nền chạy vòng lặp xử lý từng frame, cập nhật liên tục tiến trình vào `JobManager`:
    `progress = { "stage": "analyzing_frames", "current": frame_idx + 1, "total": total_frames }`.
  - Mỗi frame áp dụng `FRAME_INFERENCE_TIMEOUT_SEC`.
- [x] **Task 4.3: Video Aggregation Engine & Fail-Closed Gate (`web/backend/app/services/video_aggregator.py`)**
  - **Trường hợp A (ĐÃ XÁC MINH tại Sprint 0):**
    - Áp dụng công thức khoa học đã xác minh để tính `final_video_score` và gán nhãn `REAL`/`FAKE`.
    - Trả response: `status: "ok"`, `final: { verdict, fake_probability, continuous_score, explanation: null }`, `frames: [...]`.
  - **Trường hợp B (CHƯA XÁC MINH / BLOCKED):**
    - **TUYỆT ĐỐI KHÔNG TỰ BỊA ĐIỂM HOẶC VERDICT CHO VIDEO.**
    - Trả response: `status: "blocked"`, `error_code: "VIDEO_AGGREGATION_UNVERIFIED"`, `final: null`, `frames: [...]`.
    - Ghi nhận rõ ràng là trạng thái bị chặn do thiếu nguồn, không tính là PASS trọn vẹn.
- [x] **Task 4.4: Smoke Test Video Thật trên Server**
  - Chạy smoke test video, xác nhận client polling thành công và dọn sạch file frame tạm.

---

### SPRINT 5 — Frontend Hoàn thiện (React + Vite)
> **Mục tiêu:** Xây dựng giao diện web thẩm mỹ cao, hiện đại, hỗ trợ cả ảnh và video, thể hiện trung thực tiến trình polling và quy tắc explanation.

- [x] **Task 5.1: Cấu trúc & Mock Mode Banner (`web/frontend/src/components/Header.jsx`)**
  - Header hiện đại + Banner cảnh báo màu vàng cam khi chạy `RUN_MODE=mock`.
- [x] **Task 5.2: Upload Component (`web/frontend/src/components/Dropzone.jsx`)**
  - Hai tab rõ ràng: Image và Video. Drag & Drop và xem trước file.
- [x] **Task 5.3: Processing Indicator Trung thực (`web/frontend/src/components/ProcessingStatus.jsx`)**
  - Đối với Video: Hiển thị tiến trình thực tế nhận từ API Polling (ví dụ: `Đang phân tích frame 14/32...`).
  - Đối với Ảnh: Hiển thị spinner và thông báo trung thực (`Đang phân tích dữ liệu, vui lòng chờ...`).
  - **Tuyệt đối không dùng bộ đếm phần trăm giả.**
- [x] **Task 5.4: Card Kết quả Ảnh & Explanation Block (`web/frontend/src/components/ImageResultCard.jsx`)**
  - Badge Verdict: **REAL** hoặc **FAKE** + Gauge Continuous Score.
  - Bảng chi tiết minh bạch: Selected Expert, Router Confidence, Expert Score, Final Score.
  - **Quy tắc hiển thị Explanation**:
    - NẾU `result.final.explanation` tồn tại và không rỗng $\to$ Hiển thị khối "AI Reasoning & Explanation".
    - NẾU `result.final.explanation === null` $\to$ **Hoàn toàn ẩn khối giải thích**, không để lại ô trống hoặc text mẫu.
- [x] **Task 5.5: Card Kết quả Video & Interactive Timeline (`web/frontend/src/components/VideoResultCard.jsx`)**
  - **Xử lý trạng thái BLOCKED**: Nếu `status == "blocked"`, hiển thị cảnh báo `KẾT LUẬN VIDEO CHƯA XÁC MINH (THIẾU THUẬT TOÁN TỔNG HỢP GỐC)`, ẩn verdict REAL/FAKE.
  - **Frame Timeline & Video Seek**: Click frame tua video preview đến đúng `timestamp_seconds`.
- [x] **Task 5.6: Modal Xử lý Lỗi Fail-Closed (`web/frontend/src/components/ErrorModal.jsx`)**
  - Hiển thị lỗi rõ ràng kèm `request_id`.

---

### SPRINT 6 — Bộ Kiểm thử Toàn diện & Nghiệm thu Trình duyệt
> **Mục tiêu:** Kiểm thử tự động bảo vệ toàn bộ các ràng buộc nghiệp vụ, kiểm thử an toàn subprocess, bảo vệ production guard, test polling job và nghiệm thu E2E thật qua browser.

- [x] **Task 6.1: Unit Test Validation (`web/tests/test_media_validator.py`)**
  - Test file rỗng, đổi đuôi giả mạo, ảnh quá khổ, video quá 60s.
- [x] **Task 6.2: Test API Contracts & Fail-Closed (`web/tests/test_api_contracts.py`)**
  - Test response schema, test từ chối `bool` score, test score out-of-bounds.
- [x] **Task 6.3: Test Quy tắc Explanation (`web/tests/test_explanation_rules.py`)**
  - Kiểm tra trường `final.explanation` chỉ nhận `str` hợp lệ hoặc `None`, không nhận template bịa đặt.
- [x] **Task 6.4: Test Async Video Jobs & Polling (`web/tests/test_video_async_jobs.py`)**
  - Test luồng: Upload video $\to$ nhận 202 với `job_id` $\to$ Polling trả về `processing` $\to$ Polling trả về kết quả cuối.
- [x] **Task 6.5: Test An toàn Subprocess & Timeout (`web/tests/test_subprocess_safety.py`)**
  - Test kill toàn bộ process tree khi timeout.
- [x] **Task 6.6: Production Guard Test**
  - Khẳng định khi cấu hình `RUN_MODE=live`, server không bao giờ âm thầm nạp Mock Worker.
- [x] **Task 6.7: Nghiệm thu End-to-End qua Trình duyệt (Browser E2E Acceptance)**
  - Chạy **1 ảnh thật** từ trình duyệt -> FastAPI -> Live AI -> UI hiển thị đầy đủ 3 điểm số.
  - Chạy **1 video thật** từ trình duyệt -> FastAPI -> Async Polling -> Frame extraction -> Per-frame AI -> UI hiển thị đầy đủ timeline.
  - Đánh giá phân loại kết quả nghiệm thu video:
    - Nếu có thuật toán tổng hợp xác minh: Ghi nhận **PASS (Complete Video Pipeline)**.
    - Nếu thiếu thuật toán tổng hợp: Ghi nhận **BLOCKED (Documented Blocker - Missing Aggregation Source)**, không đánh đồng là PASS toàn bộ.

---

### SPRINT 7 — Triển khai, Vệ sinh Git & Bàn giao
> **Mục tiêu:** Hoàn thiện tài liệu vận hành, kiểm tra vệ sinh Git và bàn giao hệ thống.

- [x] **Task 7.1: Build Kiểm tra Frontend (`npm run build`)**
  - Đảm bảo build thành công không có lỗi.
- [x] **Task 7.2: File Cấu hình Mẫu (`web/.env.example`)**
  - Đầy đủ biến môi trường, phân tầng timeout, TTL job.
- [x] **Task 7.3: Tài liệu Vận hành (`web/RUNBOOK.md`)**
  - Hướng dẫn chạy Uvicorn 1 worker (`uvicorn --workers 1`), giám sát VRAM, dọn dẹp thư mục tạm.
- [x] **Task 7.4: Hướng dẫn Sử dụng (`web/README_WEB.md`)**
- [x] **Task 7.5: Kiểm tra Vệ sinh Git & Commit/Push**
  - Rà soát `.gitignore`: Tuyệt đối không commit checkpoint nặng (`*.pt`, `*.safetensors`, `*.bin`, `*.joblib`), raw data hay secrets.
  - Push lên nhánh tính năng được cấp phép.

---

## 4. Tiêu chuẩn Nghiệm thu (Definition of Done)

- [x] **Sprint 0 hoàn thành**: Có `web/SOURCE_AUDIT.md` và `web/VERIFICATION_MATRIX.md` xác nhận rõ ràng trạng thái của 4 expert, checkpoints, video protocol, continuous score, explanation capability và quy ước prompt WFS.
- [x] **Backend sẵn sàng**: FastAPI khởi động thành công với 1 worker; endpoint `/health` và `/ready` phản ánh đúng trạng thái thực tế; Video Async Job Engine hoạt động trơn tru.
- [x] **Frontend hoàn thiện**: Giao diện React Dark theme hiện đại, upload được cả ảnh và video, polling tiến độ trung thực, có Frame Timeline động kết nối video seek/thumbnail, phân biệt rõ Mock mode; ẩn hoàn toàn explanation khi null.
- [x] **Pipeline Ảnh chuẩn xác**: Chạy đúng thứ tự `Router4 -> 1 Expert -> Calibrator -> WFS -> X²-DFD -> Continuous Score`.
- [x] **Pipeline Video trung thực & Phân định nghiệm thu**:
  - Trích xuất frame theo chuẩn đã audit.
  - Nếu tổng hợp video đã xác minh: Nghiệm thu **PASS (Complete Video Pipeline)**.
  - Nếu tổng hợp video chưa xác minh: Trả `status: "blocked"`, `final: null`, ghi nhận rõ ngoại lệ kỹ thuật **BLOCKED (Missing Aggregation Source)**, không đánh đồng là PASS toàn bộ.
- [x] **Fail-Closed 100%**: Mọi trường hợp file hỏng, thiếu điểm, `bool` score, timeout, OOM đều trả về lỗi chi tiết, không bao giờ sinh verdict `REAL`/`FAKE` giả mạo.
- [x] **An toàn Subprocess & GPU**: `shell=False`, timeout đa cấp, process tree termination, điều phối GPU bằng Semaphore/GpuLock.
- [x] **Bộ Test Suite PASS**: Đầy đủ unit test, validation test, contract test, explanation rules test, async job test, subprocess safety test, stale cleanup test.
- [x] **Nghiệm thu E2E Trình duyệt**:
  - Test 1 ảnh thật từ trình duyệt -> FastAPI -> Live AI -> UI hiển thị đầy đủ 3 điểm số.
  - Test 1 video thật từ trình duyệt -> FastAPI -> Async Polling -> Live Extraction -> Live Per-frame -> UI hiển thị đầy đủ timeline và kết quả (hoặc cảnh báo blocked).
- [x] **Vệ sinh Git**: Rà soát `.gitignore`, không đưa checkpoint nặng, raw data hay secrets lên Git; commit/push mã nguồn `web/` lên nhánh được phép.

