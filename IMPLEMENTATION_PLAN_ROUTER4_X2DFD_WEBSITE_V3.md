# IMPLEMENTATION PLAN V3 — ROUTER4 × X²-DFD WEBSITE DEMO

**Ngày lập:** 24/09/2026  
**Đầu vào:** `HANDOFF_ROUTER4_X2DFD_WEBSITE_V3.md`; bản ZIP `router4-x2dfd-demo-main(1).zip` người dùng cung cấp.  
**Phạm vi:** Website thật, chạy end-to-end trên máy chủ nghiên cứu; **IMAGE + VIDEO đều là P0/MVP**.  
**Trạng thái:** Kế hoạch triển khai; **chưa phải bằng chứng website hoặc mô hình đã chạy thành công**.

> Tài liệu này thay thế kế hoạch website chỉ hỗ trợ ảnh. Phần nào chưa xác minh bằng mã nguồn/server được đánh dấu **UNVERIFIED/BLOCKED**, không tự suy đoán hoặc dùng mock để báo hoàn thành.

---

## 1. Mục tiêu, phạm vi và sản phẩm bàn giao

Người dùng mở website, tải **ảnh hoặc video**, nhấn **Analyze**, theo dõi trạng thái xử lý và nhận kết quả **REAL/FAKE thực tế từ Router4 × X²-DFD** khi hệ thống tạo được **điểm liên tục của bộ phát hiện cuối**. Nếu bất kỳ chặng cần thiết nào lỗi hoặc thiếu score hợp lệ, UI hiển thị **FAILED/UNKNOWN**, không hiện kết luận thành công.

**Các sản phẩm bắt buộc:**

- [ ] Frontend responsive: upload ảnh/video bằng chọn file hoặc kéo-thả; preview; validation; loading/progress trung thực; kết quả; lỗi.
- [ ] Backend API: nhận/kiểm tra media, quản lý yêu cầu, gọi pipeline thật, trả response có schema rõ ràng.
- [ ] Tích hợp inference cho ảnh: **Router4 → đúng 1 expert → expert score → WFS → X²-DFD/LLaVA + Router-aware LoRA → final continuous score**.
- [ ] Tích hợp inference cho video: trích xuất frame bằng đường xử lý đã xác minh; **từng frame** chạy pipeline ảnh; tổng hợp video theo quy tắc **được xác minh từ source**.
- [ ] `/health` và `/ready` phản ánh đúng tình trạng API và runtime/model.
- [ ] Kiểm thử backend, frontend, AI semantics, an toàn upload, video và end-to-end trên server.
- [ ] `README_WEB.md`, `API_CONTRACT.md`, `RUNBOOK.md`, `.env.example` và log/test evidence; commit/push mã nguồn được phép.

**Không nằm trong MVP nếu chưa có yêu cầu riêng:** đăng nhập, tài khoản, CSDL lịch sử, dashboard admin, huấn luyện lại expert, tinh chỉnh trên benchmark ngoài. Không bổ sung các phần này để che những blocker P0.

## 2. Những gì đã thấy trong tài liệu và ZIP; những gì chưa xác minh

| Hạng mục | Tình trạng tại thời điểm lập plan | Hành động |
|---|---|---|
| Handoff V3 yêu cầu ảnh **và** video | **CONFIRMED — tài liệu yêu cầu** | Cả hai phải có trong MVP. |
| ZIP có `runtime_context/`, `source_snapshot/`, X2DFD, DFFreq, Texture, tích hợp Router/cache, runtime/server wrappers | **CONFIRMED — từ ZIP** | Đọc mã thực tế, tận dụng wrapper phù hợp. |
| Thư mục ứng dụng `web/` hoàn chỉnh trong ZIP | **Chưa thấy `web/` trong bản ZIP** | Tạo mới, giữ nguyên research snapshot. |
| Bốn expert có đủ entrypoint + checkpoint + preprocessing trên server | **UNVERIFIED** | Audit riêng B/D/F/T trước khi gọi live. |
| Checkpoint Router, calibrators, base LLaVA, LoRA tại server | **Đường dẫn được handoff ghi nhận; chưa xác minh tồn tại/chạy được** | Preflight path, hash nếu có, quyền đọc, load thử. |
| Final score thực sự có trong output `lora_inference.py` → `runner.py` | **UNVERIFIED / blocker cần kiểm tra** | Kiểm tra return/serialization/schema; fail-closed. |
| Video frame extraction và quy tắc video aggregation | **UNVERIFIED** | Tìm nguồn X²-DFD/server, kiểm tra exact protocol. |
| `router_moe_cache` hoạt động cho benchmark replay | **Có blocker ghi nhận trong handoff; trạng thái source mới chưa audit** | Audit trước, không biến cache thành expert thứ 5. |

**Các file nên đọc đầu tiên từ ZIP:**

1. `README.md`, `runtime_context/artifact_manifest.tsv`, `runtime_context/runtime_paths.env.example`, `runtime_context/environment/`, `runtime_context/logs/`.
2. `source_snapshot/projects/X2DFD/eval/infer/runner.py` và `source_snapshot/projects/X2DFD/utils/lora_inference.py`.
3. `source_snapshot/projects/x2dfd_router_integration/`, `source_snapshot/final_clean_integration/`.
4. `source_snapshot/runtime_wrappers/`, `source_snapshot/server_wrappers/`.
5. Source chuyên biệt cho từng expert Blending, Diffusion, Frequency và Texture trong ZIP **và** trên server.

**Thứ tự ưu tiên khi nguồn mâu thuẫn:** source/checkpoint/log thực tế trên server → output/hashes/report → ZIP/Git hiện tại → source/paper X²-DFD → Handoff V3 → trao đổi cũ. Không dùng lịch sử chat thay cho chứng cứ runtime.

---

## 3. Sơ đồ chức năng bắt buộc

### 3.1. Ảnh

```text
Browser (upload ảnh)
  → API: kiểm tra loại file, magic bytes, decode, kích thước
  → Router4(image)                         [image-only]
  → chọn đúng 1 expert trong B / D / F / T
  → chạy expert đã chọn và áp dụng đúng score convention/calibration
  → build WFS từ selected expert alias + selected expert score
  → X²-DFD / LLaVA + Router-aware LoRA
  → lấy continuous FINAL detector score thật
  → validate → REAL/FAKE + P(fake)/P(real) + explanation nếu thật sự có
  → API → UI
```

### 3.2. Video

```text
Browser (upload video)
  → API: xác minh format/codec, size, duration, decode
  → frame extraction / sampling theo nguồn thực tế ĐÃ XÁC MINH
  → giữ nguyên thứ tự frame + timestamp + số frame
  → với từng frame: Router4 → 1 expert → WFS → X²-DFD/LoRA → final frame score
  → video aggregation theo quy tắc ĐÃ XÁC MINH từ source
  → final VIDEO score + frame-level outputs tách biệt
  → API → UI (video result và timeline/table)
```

**Bất biến:** Router confidence `q_i` ≠ expert fake probability `p_i` ≠ final detector probability. Không dùng benchmark cache cho upload mới; không dùng nhãn ground truth/oracle; không huấn luyện lại expert. Không tự chọn frame giữa, chọn frame ngẫu nhiên, nhân bản frame hoặc tự nghĩ mean/max aggregation.

---

## 4. Kế hoạch thực hiện theo thứ tự (gates trước khi mở rộng)

### Giai đoạn 0 — Chuẩn bị repository, nhánh làm việc và checklist

**Việc làm**

- [ ] Clone/pull repository và ghi commit hash; so với ZIP hiện có trước khi sửa.
- [ ] Tạo nhánh tính năng, ví dụ `feature/website-image-video`; không commit trực tiếp vào nhánh chính nếu nhóm dùng Pull Request.
- [ ] Đọc Handoff V3, README, manifest, runtime env và log lỗi gần nhất.
- [ ] Thiết lập thư mục `web/` mới; không sửa snapshot nghiên cứu để làm giả một demo dễ chạy.
- [ ] Lập bảng `VERIFIED / UNVERIFIED / BLOCKED` cho mỗi module, path, artifact, input/output và môi trường.
- [ ] Xác nhận người có quyền truy cập server GPU, checkpoint, conda environments và quyền push.

**Đầu ra:** `web/README_WEB.md` mô tả phạm vi + `web/VERIFICATION_MATRIX.md` (hoặc mục tương đương) + nhánh Git sạch.  
**Điều kiện qua giai đoạn:** xác định được những file, môi trường và điểm chặn phải audit; **không** đánh dấu artifact là tồn tại chỉ dựa trên đường dẫn từ handoff.

### Giai đoạn 1 — Source audit & runtime preflight (P0)

**1.1. Kiểm tra từng expert**

- [ ] **Blending:** đúng source/entrypoint, checkpoint, preprocessing, input/output, score chiều fake, Python env.
- [ ] **Diffusion:** tương tự; xác định mode/model từ cấu hình đang sử dụng.
- [ ] **Frequency:** kiểm tra DFFreq source, wrapper thực tế và quy tắc sigmoid/logit (không suy ra từ tên file).
- [ ] **Texture:** kiểm tra Texture source, wrapper thực tế và quy tắc softmax/index; không tự đổi chiều nhãn.
- [ ] Ghi test nhỏ cho từng expert trên input kiểm thử được phép; không dùng benchmark ground truth để chạy website.
- [ ] Nếu thiếu bất kỳ source/entrypoint/checkpoint thiết yếu: đánh dấu **BLOCKED** với file/path cần bổ sung; không tự viết lại expert từ trí nhớ.

**1.2. Kiểm tra Router và final model**

- [ ] Kiểm tra `best.pt`, calibrators `.joblib`, LoRA adapter, base LLaVA, version/env và quyền đọc.
- [ ] Xác minh Router chỉ nhận ảnh/frame; trả đúng 1 expert, optional router confidence nếu thật sự có.
- [ ] Đọc `lora_inference.py` và `runner.py`: hàm lấy token/logit/score, giá trị `real_score`/`fake_score`, JSON được lưu và gọi qua runner.
- [ ] Xác minh score cuối là **liên tục thật**, ý nghĩa `P(fake)`/`P(real)`, miền hợp lệ và ngưỡng verdict; không suy diễn từ text verdict.
- [ ] Kiểm tra lỗi `Unknown score provider: router_moe_cache` đã được sửa chưa; cache chỉ dùng cho benchmark replay, không cho live user upload.

**1.3. Kiểm tra video**

- [ ] Tìm entrypoint thực tế cho video preprocessing/frame extraction trong source **và** trên server.
- [ ] Ghi rõ cách chọn frame (deterministic/source protocol), timestamp, thứ tự, hành vi frame lỗi, frame count và decoder.
- [ ] Tìm phép tổng hợp kết quả video thực tế và cách xử lý frame thiếu điểm; ghi file/dòng mã/chứng cứ.
- [ ] Nếu chưa xác minh aggregation: **BLOCKED video final verdict**; không trả một video probability do tự đặt công thức.

**1.4. Kiểm tra môi trường và GPU**

- [ ] Xác minh executable Python riêng, requirements/version riêng của X2DFD/DFFreq/Texture/các expert khác.
- [ ] Preflight GPU/CUDA/VRAM, input/output test, quyền thư mục tạm và giới hạn concurrency.
- [ ] Không gộp tất cả AI dependencies vào môi trường FastAPI; dùng wrapper/subprocess khi cần.

**Đầu ra:** `web/SOURCE_AUDIT.md` gồm source path, env, checkpoint, score semantics, video protocol, log chứng cứ, blocker và quyết định đã xác minh.  
**Điều kiện qua giai đoạn:** biết đường đi thật của ảnh, khả năng thật của video và những blocker còn lại. Nếu bị chặn, vẫn có thể làm FE/BE theo schema nhưng **không được báo AI E2E PASS**.

### Giai đoạn 2 — Chốt contract API và tổ chức mã nguồn

**Cấu trúc đề xuất (theo Handoff V3, chỉnh sửa nếu audit cho thấy cần):**

```text
web/
├── frontend/                 # React/Vite hoặc stack tương đương
├── backend/                  # API + validation + orchestration
├── model_worker/             # wrapper/subprocess gọi các env AI
├── tests/                    # unit, API, integration, E2E
├── API_CONTRACT.md
├── SOURCE_AUDIT.md
├── README_WEB.md
├── RUNBOOK.md
└── .env.example
```

**API dự kiến — phải chốt sau source audit:**

| Endpoint | Vai trò | Điều kiện |
|---|---|---|
| `GET /api/v1/health` | API còn sống | Không giả định model sẵn sàng. |
| `GET /api/v1/ready` | Trạng thái thực của artifact/env/model | Trả thiếu thành phần với mã lỗi phù hợp. |
| `POST /api/v1/analyze/image` | Upload ảnh và gọi pipeline thật | Kết quả thành công chỉ khi final score hợp lệ. |
| `POST /api/v1/analyze/video` | Upload video và chạy pipeline theo frame | Thành công chỉ khi frame path **và** video aggregation được xác minh. |

Handoff cho phép **một** `POST /api/v1/analyze` tự nhận diện media thay vì hai endpoint. Nhóm chọn một phương án và khóa contract; bảng trên là **đề xuất triển khai**, không phải API đã tồn tại trong repo.

**Schema chung phải phân biệt rõ:**

- `request_id`, `media_type`, `status` (`ok`/`error`), timing và warnings thực tế.
- Ảnh: `router.selected_expert`, optional `router.router_confidence`; `expert.fake_probability`; `final.label`, `final.real_probability`, `final.fake_probability`, `final.continuous_score`, optional `final.explanation`.
- Video: `video.duration_seconds`, `video.frames_analyzed`; danh sách `frames[]` có timestamp, selected expert, expert score, **final frame score**; `final` là **video-level**, không đồng nhất với frame cuối.
- Nếu `final.continuous_score` thiếu, NaN, sai kiểu, out-of-contract hoặc upstream lỗi: response `status="error"` có `stage`, `error_code`, `message`, `request_id`; **không có successful REAL/FAKE**.
- Không tự suy ra explanation từ tên expert, confidence hoặc template. Trả `null`/ẩn nếu model không tạo explanation thật.

**Đầu ra:** `API_CONTRACT.md`, schema model và mẫu response **được ghi là ví dụ schema, không phải kết quả AI thực tế**.  
**Điều kiện qua giai đoạn:** FE và BE có một hợp đồng nhất quán, bao gồm cả lỗi.

### Giai đoạn 3 — Backend nền tảng & xử lý media

- [ ] Khởi tạo Python API (FastAPI + Pydantic nếu tương thích runtime); cấu hình bằng biến môi trường.
- [ ] Tạo request ID; giới hạn upload body/file size; chặn loại file không hỗ trợ trước khi xử lý nặng.
- [ ] Xác thực MIME **và magic bytes**, decode thật; kiểm tra ảnh quá lớn/pixel limit; video duration/format/codec/size theo giới hạn đã chốt.
- [ ] Lưu file tạm vào thư mục ngẫu nhiên riêng theo request; không tin tên file/path do người dùng nhập.
- [ ] Cleanup trong `finally` cả khi cancel, exception, timeout; không commit/log nội dung ảnh/video hoặc secrets.
- [ ] Tạo adapter gọi worker với Python executable cố định theo config, danh sách args an toàn (`shell=False` nếu dùng subprocess), timeout, stdout/stderr, return code và schema validation.
- [ ] Bổ sung semaphore/concurrency limit cho GPU, xử lý OOM, timeout, worker crash, file hỏng.
- [ ] Cài `/health`, `/ready` phân biệt API alive và readiness thật; không báo ready chỉ vì file path tồn tại.

**Đầu ra:** BE nhận và validate ảnh/video, có cấu trúc lỗi an toàn, worker interface và health/readiness.  
**Điều kiện qua giai đoạn:** unit/API tests cho upload và failure cases chạy được mà không cần giả vờ AI đã hoạt động.

### Giai đoạn 4 — Model worker: chạy IMAGE thật

- [ ] Dùng inference entrypoint/weights/env đã audit; không tái hiện thuật toán bằng mã mới không kiểm chứng.
- [ ] Normalize/decode ảnh đúng quy trình Router4; route **đúng 1** expert trong B/D/F/T.
- [ ] Chạy selected expert trong runtime thích hợp, lấy score thật, áp dụng calibration/convention đã xác minh.
- [ ] Build WFS prompt từ **alias + score của expert đã chọn**; không truyền router confidence vào chỗ expert score.
- [ ] Gọi X²-DFD/LLaVA + Router-aware LoRA; parse điểm cuối liên tục, nhãn và explanation **nếu có thật**.
- [ ] Validate score finite, schema, miền score và ngữ nghĩa; nếu không hợp lệ **fail closed** (`FINAL_SCORE_UNAVAILABLE` hoặc lỗi cụ thể).
- [ ] Ghi timings từng công đoạn và provenance không chứa secrets/personal media.
- [ ] Chạy một smoke test ảnh thật qua HTTP trên máy chủ AI, lưu response/log đã ẩn thông tin nhạy cảm.

**Đầu ra:** endpoint ảnh chạy thật và `web/tests/` bao phủ score/route/failure.  
**Điều kiện qua giai đoạn:** **ít nhất 1 ảnh** từ HTTP → Router4 → expert → WFS → X²-DFD/LoRA → kết quả có final score hợp lệ; không dùng benchmark cache hoặc mock.

### Giai đoạn 5 — Model worker: chạy VIDEO thật (P0)

- [ ] Chỉ sau khi có evidence extraction/aggregation: cắm đúng preprocessor và aggregator đã audit.
- [ ] Video validation/decode; chọn frame **theo source protocol**, ghi timestamp, thứ tự, frame count và warning thực tế.
- [ ] Từng frame gọi cùng pipeline ảnh thật, **Router4 được chạy riêng cho mỗi frame**.
- [ ] Giữ frame-level results: timestamp, selected expert, expert fake probability, final frame score.
- [ ] Tổng hợp video bằng thuật toán **đã được xác minh**, bảo toàn quy tắc xử lý frame lỗi/thiếu score.
- [ ] Tạo **video-level** verdict/probabilities riêng; không coi frame là video độc lập trong scientific metrics.
- [ ] Khi extraction/aggregation chưa xác minh hoặc score video không hợp lệ: trả `UNKNOWN/FAILED` kèm stage; không dựng số demo.
- [ ] Chạy 1 smoke test video thật trên server, đo latency/frame count/tình trạng GPU và log lỗi.

**Đầu ra:** video API, frame outputs, video final outputs (khi có bằng chứng hợp lệ), kiểm thử video.  
**Điều kiện qua giai đoạn:** **ít nhất 1 video thật** qua HTTP → frame extraction đã xác minh → per-frame AI → verified aggregation → response hợp lệ; hoặc báo chính xác blocker nếu chưa đủ nguồn.

### Giai đoạn 6 — Frontend hoàn chỉnh (có thể phát triển song song giai đoạn 3–5)

**Màn hình/khối giao diện cần làm:**

1. **Upload:** 2 lựa chọn Image / Video; drag/drop + file picker; loại/size/duration validation; ảnh/video preview; nút Analyze; chặn submit trùng.
2. **Processing:** progress theo trạng thái BE thực sự cung cấp; thời gian xử lý; timeout/cancel nếu backend đã hỗ trợ. Không hiện phần trăm tiến độ giả.
3. **Image result:** REAL/FAKE; `final P(fake)` / `final P(real)`; selected expert; expert fake score; router confidence **chỉ khi có ngữ nghĩa hợp lệ**; explanation thật nếu có; request ID, latency, warnings, runtime/provenance có thể mở rộng.
4. **Video result:** video final verdict + final probabilities; duration; frames analyzed; timeline/bảng frame theo timestamp; expert và final frame score theo frame; tách rõ **frame-level** / **video-level**; warning.
5. **Failure/UNKNOWN:** lỗi upload, runtime thiếu, router/expert/X²-DFD/frame extraction/aggregation lỗi, timeout, OOM, final score thiếu; nút thử lại khi hợp lý và request ID để tra log.

**Công việc kỹ thuật FE:**

- [ ] Chọn React/Vite (hoặc stack tương đương); responsive desktop/mobile, accessibility cơ bản.
- [ ] Tạo API client và TypeScript types (nếu dùng TS) từ contract; API base URL từ env.
- [ ] Dùng mock **chỉ để phát triển UI**, test rõ được đánh dấu mock; production path **không chứa giá trị AI giả**.
- [ ] Phân biệt trực quan/nhãn văn bản cho `router confidence`, `expert fake probability`, `final fake probability`.
- [ ] Không hiển thị verdict `FAKE/REAL` khi `status=error`, score invalid hoặc video aggregation chưa verified.
- [ ] Chạy FE lint/build và test trạng thái upload/processing/success/error cho cả 2 media.

**Đầu ra:** browser giao diện hoàn chỉnh; kết nối BE theo contract; không hardcode kết quả giả.  
**Điều kiện qua giai đoạn:** người dùng xử lý được cả ảnh và video trong UI; UI thất bại trung thực.

### Giai đoạn 7 — Kiểm thử toàn hệ thống và kiểm thử ngữ nghĩa AI

| Nhóm test | Các trường hợp bắt buộc | Tiêu chí |
|---|---|---|
| Static | Python compile/import; FE lint/build; schema | Không có lỗi cản trở chạy. |
| Upload ảnh | Hợp lệ, sai định dạng, file rỗng, đổi đuôi, ảnh decode lỗi, quá size/pixel | Validation thực, không crash. |
| Upload video | Hợp lệ, sai codec/format, file hỏng, quá size/duration, không có frame khả dụng | Error chính xác; cleanup. |
| Runtime | Thiếu checkpoint, Python env lỗi, expert/Router/X²-DFD lỗi, subprocess exit != 0, timeout, GPU OOM | Không có normal success sau lỗi. |
| Score | `null`, `NaN`, vô hạn, sai kiểu, ngoài miền, hard label không có score | `status=error`, không phỏng đoán P(fake). |
| AI semantics | image-only Router; 1 expert/frame; đúng B/D/F/T; WFS alias/score; phân biệt 3 score; không GT/cache | Test/assert có bằng chứng. |
| Video protocol | frame ordering/timestamp; không random/middle-only/duplicate ngoài source; verified aggregation | Không bịa video score. |
| Frontend | Desktop/mobile, preview, loading, duplicate submit, lỗi, frame/video tách biệt | Mọi trạng thái hiển thị đúng. |
| E2E | **1 ảnh thật** và **1 video thật** từ browser đến mô hình rồi quay lại UI | Ghi log/response/runbook; chỉ PASS khi chạy thật. |

**Quy tắc kiểm thử:** fixture/mock chỉ chứng minh hành vi API/UI, **không** chứng minh AI integration. Lưu phân biệt `PASS (mock)`, `PASS (live server)`, `BLOCKED`, `FAIL`, `NOT RUN`. Không dùng kết quả website để suy ra benchmark khoa học đã đạt.

**Đầu ra:** `web/tests/`, báo cáo `web/TEST_REPORT.md` và log có request ID, commit hash, cấu hình/artifact version, trường hợp lỗi.

### Giai đoạn 8 — Triển khai và bàn giao

- [ ] Chốt cấu hình server API/FE/model workers với env riêng từng module, paths lấy từ config; không hardcode đường dẫn cá nhân trong mã ứng dụng.
- [ ] Viết `.env.example` **không chứa secrets**, mô tả các biến cấu hình: API URL, upload/timeout/concurrency limits, Python executable và checkpoint/config path được xác minh.
- [ ] Cấu hình network/CORS cho đúng FE domain; giới hạn body/file size cả reverse proxy lẫn backend nếu dùng proxy.
- [ ] Đảm bảo server API có quyền đọc artifact và quyền ghi temp; cô lập môi trường expert.
- [ ] Cập nhật `/ready` kiểm tra thật sau deploy; test cold-start/warm-start nếu hợp lý với VRAM.
- [ ] Tạo README hướng dẫn clone/cài FE/BE/chạy local UI và kết nối server GPU; runbook vận hành, log, restart, cleanup, xử lý OOM/timeout.
- [ ] Review `.gitignore`: chặn `.env`, tokens, uploads, temp, log nhạy cảm, raw dataset, model binaries lớn, cache và outputs.
- [ ] Commit phần `web/` và các tài liệu liên quan, push nhánh được phép, tạo PR/review/merge theo workflow nhóm; **không push nếu chưa được cấp quyền**.
- [ ] Chạy lại E2E **trên bản deploy thật** trước khi đánh dấu hoàn thành.

**Đầu ra:** website deploy thật + repo sạch + hướng dẫn và bằng chứng smoke image/video.  
**Điều kiện qua giai đoạn:** checklist Definition of Done mục 7 bên dưới được đáp ứng hoặc blocker được báo cụ thể, không thay PASS bằng giả định.

---

## 5. Phân chia công việc để FE/BE làm song song

| Nhánh/luồng | Người phụ trách đề xuất | Có thể bắt đầu | Phụ thuộc/giao tiếp |
|---|---|---|---|
| Source + server audit | Người nắm AI/server | Ngay | Cung cấp schema thật, env, protocol, blocker cho cả nhóm. |
| Backend media/API | BE | Sau API contract dự kiến | Có thể test bằng stub để xác nhận validation; không gọi stub là AI PASS. |
| Model worker ảnh/video | Người tích hợp AI + BE | Sau source preflight | Chặn success nếu thiếu score/aggregation. |
| Frontend | FE | Ngay sau contract nháp | Mock được dùng khi phát triển UI; thay bằng API thật trước nghiệm thu. |
| Integration + E2E | FE + BE + AI/server | Khi các nhánh sẵn sàng | Xác minh luồng thực tế cả 2 loại file. |

**Ghi chú:** Handoff gán trách nhiệm FE/BE cho Astra. Bảng trên chỉ chia luồng công việc kỹ thuật để một cá nhân hoặc nhóm thực hiện; không tự thay đổi chủ sở hữu được ghi trong handoff.

### Milestone thực tế

- **M0 — Audit complete:** có verification matrix, biết exact blockers; chưa tuyên bố AI chạy.
- **M1 — API + FE skeleton:** hai đường upload và lỗi chạy qua mocked contract; chưa tuyên bố E2E thật.
- **M2 — IMAGE live:** ảnh chạy full pipeline với final continuous score thật.
- **M3 — VIDEO live:** video chạy verified extraction + per-frame full pipeline + verified aggregation.
- **M4 — Release:** FE/BE deploy, readiness thật, test report E2E image/video, README/runbook và PR/commit.

---

## 6. Các điểm chặn P0 và cách xử lý

| Blocker | Dấu hiệu | Cách xử lý hợp lệ | Tuyệt đối không làm |
|---|---|---|---|
| Thiếu expert source/entrypoint/weights | Không chạy được selected expert | Xác minh source/server; yêu cầu đúng repo/path/checkpoint; đánh dấu BLOCKED. | Tự chế mô hình/wrapper từ giả định. |
| `router_moe_cache` provider error | `Unknown score provider: router_moe_cache` | Audit runner; giữ cache chỉ cho benchmark replay; live input dùng Router4 + expert thật. | Đăng ký cache thành expert thứ 5. |
| Không có final continuous score | Chỉ có text REAL/FAKE hoặc score lỗi | Audit `lora_inference.py`/`runner.py`; sửa đường truyền/serialize dựa trên source; fail closed. | Dùng router confidence, expert score hoặc 0.5 thay thế. |
| Video preprocessing chưa rõ | Không xác minh frame selection/timestamp | Đọc đường video thật của X²-DFD/server; khóa protocol. | Chỉ lấy frame giữa hoặc random cho tiện. |
| Video aggregation chưa rõ | Không có phép tổng hợp được xác minh | Tìm source/log; đánh dấu `UNVERIFIED` và không trả final video success. | Tự dùng mean/max, số giả hay lấy frame cuối. |
| Runtime xung đột / GPU OOM | subprocess hỏng/VRAM hết | Env isolation, queue/concurrency limit, timeout, retry policy được kiểm soát. | Gộp môi trường tùy tiện, bỏ qua lỗi. |

---

## 7. Definition of Done — chỉ đánh dấu khi có chứng cứ

- [ ] Repo có thể clone và có hướng dẫn rõ ràng.
- [ ] Backend khởi động **trên server nghiên cứu**; `/health` chạy.
- [ ] `/ready` phản ánh thực tế model/runtime.
- [ ] Frontend build và mở được trên trình duyệt desktop/mobile.
- [ ] Upload ảnh và video được validate, preview và gửi qua API.
- [ ] Ảnh chạy **Router4 → 1 expert B/D/F/T → score thật → WFS → X²-DFD/LoRA → final score thật**.
- [ ] Video dùng frame extraction **đã xác minh**, mỗi frame tự chạy Router4 + pipeline, tổng hợp video **đã xác minh**.
- [ ] Video-level và frame-level được lưu/hiển thị **riêng**, timestamp và frame count đúng.
- [ ] Chỉ hiển thị REAL/FAKE và final probabilities khi điểm cuối hợp lệ; explanation chỉ khi model thực sự tạo.
- [ ] Thiếu artifact/timeout/OOM/score invalid không sinh verdict giả.
- [ ] Test image E2E browser → backend → AI → browser **PASS (live server)**.
- [ ] Test video E2E browser → backend → AI → verified aggregation → browser **PASS (live server)**; nếu bị chặn nguồn, ghi chính xác **BLOCKED**, không ghi hoàn thành.
- [ ] README, API contract, runbook, test report và cấu hình env mẫu đầy đủ.
- [ ] Commit/push đúng quyền và không chứa secrets/uploads/dataset/model binaries ngoài chủ đích.
- [ ] Không mô tả website vận hành thành công là bằng chứng benchmark khoa học thắng baseline.

---

## 8. Checklist hành động đầu tiên khi bắt tay vào code

**Làm lần lượt, không chạy tác vụ GPU dài trước preflight:**

1. [ ] Xác nhận commit repo mới nhất và đường dẫn source **trên server thực**, không chỉ dựa vào ZIP.
2. [ ] Mở `runtime_context/artifact_manifest.tsv` và `runtime_paths.env.example`; xác minh các file được ghi trong manifest bằng kiểm tra read-only.
3. [ ] Mở `X2DFD/utils/lora_inference.py` và `X2DFD/eval/infer/runner.py`; theo dấu final continuous score từ model đến JSON output.
4. [ ] Audit source/entrypoint/score conventions cho đủ B/D/F/T.
5. [ ] Xác định đường trích xuất frame và aggregation video chính xác; nếu không có, báo blocker.
6. [ ] Ghi `SOURCE_AUDIT.md` và khóa API contract.
7. [ ] Khởi tạo `web/backend`, `web/frontend`, `web/model_worker`, `web/tests`.
8. [ ] Hoàn thành và test image live path; đồng thời FE có thể làm hai màn hình bằng mock được dán nhãn development.
9. [ ] Hoàn thành và test video live path đúng protocol.
10. [ ] Test lỗi/fail-closed; chạy E2E trên browser và server; viết runbook; commit/push theo quyền.

**Lưu ý đặc biệt:** Tài liệu này là kế hoạch. Trạng thái tất cả task đang là **chưa thực hiện/chưa chứng minh** cho đến khi có mã nguồn được triển khai và kết quả test lưu lại.

---

## 9. Bảng đối chiếu với Handoff V3

| Nguồn trong Handoff V3 | Vị trí thực hiện trong plan |
|---|---|
| §1 mục tiêu website thực | §1, §4 giai đoạn 3–8 |
| §2 Router/expert, cache, score semantics | §3, §4 giai đoạn 1/4, §6 |
| §3 VIDEO P0, frame/aggregation | §3.2, §4 giai đoạn 1/5, §7 |
| §4 fail-closed continuous score | §3, §4 giai đoạn 2/4/7, §6 |
| §5 explanation thật | §4 giai đoạn 2/4/6 |
| §6–9 paths, blockers, source gate, runtime isolation | §2, §4 giai đoạn 1/3, §6 |
| §10–13 UX và API schemas | §4 giai đoạn 2/6 |
| §14–15 Backend/Frontend | §4 giai đoạn 3–6, §5 |
| §16 tests | §4 giai đoạn 7, §7 |
| §17 Git/file layout | §4 giai đoạn 0/2/8 |
| §18 Definition of Done | §7 |
| §19 execution order và §20 nguồn sự thật | §2, §4, §8 |
