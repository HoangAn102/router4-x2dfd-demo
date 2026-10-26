import React from 'react';
import { ShieldCheck, AlertOctagon, AlertTriangle, Brain, Cpu, Compass, Clock, CheckCircle2 } from 'lucide-react';

export default function ImageResultCard({ result }) {
  if (!result || !result.final) return null;

  const isFake = result.final.verdict === 'FAKE';
  const fakeProb = (result.final.fake_probability * 100).toFixed(1);
  const realProb = (result.final.real_probability * 100).toFixed(1);
  // General AI-image gate: when it overrides, `final` holds the detector score;
  // the X2DFD card keeps showing the original face-deepfake score.
  const aiGate = result.ai_detector;
  const aiOverridden = aiGate?.overridden === true;
  const aiScore = typeof aiGate?.score === 'number' ? (aiGate.score * 100).toFixed(1) : null;
  // Blending-expert gate (face-swap traces): same idea, overrides a REAL X2DFD verdict.
  const blendGate = result.blending_gate;
  const blendOverridden = blendGate?.overridden === true;
  const blendScore = typeof blendGate?.raw === 'number' ? (blendGate.raw * 100).toFixed(1) : null;
  // No face found: only the AI-image gate ran, Router4/expert/X2DFD do not apply.
  const faceless = !result.router;

  const displayProb = isFake ? fakeProb : realProb;
  const displayProbLabel = aiOverridden
    ? 'Điểm ảnh AI (detector tổng quát)'
    : blendOverridden
      ? 'Điểm ghép mặt (expert blending)'
    : isFake
      ? 'Xác suất Giả mạo (P_fake)'
      : 'Xác suất Ảnh thật (P_real)';
  const routerConf = ((result.router?.router_confidence || 0) * 100).toFixed(1);
  const margin = ((result.router?.margin || 0) * 100).toFixed(1);
  const rawExpert = ((result.expert?.raw_score || 0) * 100).toFixed(1);
  const calExpert = ((result.expert?.calibrated_score || 0) * 100).toFixed(1);

  // Strict explanation rule: Render only if explanation exists and is non-empty string
  const explanation = result.final.explanation;
  const hasExplanation = typeof explanation === 'string' && explanation.trim().length > 0;

  // Presentation-only trust flag from the backend (verdict is unchanged)
  const reliability = result.reliability;
  const isLowReliability = reliability?.level === 'low';
  const reliabilityReasons = Array.isArray(reliability?.reasons) ? reliability.reasons : [];

  // Original X2DFD probability, kept in the response when a gate replaced the final score.
  const gateX2dfd = aiOverridden ? aiGate.x2dfd_fake_probability : (blendOverridden ? blendGate.x2dfd_fake_probability : null);
  const x2dfdFakeProb = typeof gateX2dfd === 'number' ? (gateX2dfd * 100).toFixed(1) : fakeProb;
  const x2dfdRealProb = typeof gateX2dfd === 'number' ? (100 - Number(x2dfdFakeProb)).toFixed(1) : realProb;

  return (
    <div className="result-container">
      {/* 1. Verdict Banner */}
      <div className={`verdict-banner ${isFake ? 'fake' : 'real'}`}>
        <div className="verdict-badge-box">
          <div className="verdict-icon">
            {isFake ? <AlertOctagon size={36} /> : <ShieldCheck size={36} />}
          </div>
          <div>
            <div className="verdict-text-label">Phán Quyết Pháp Y AI</div>
            <div className="verdict-title">{result.final.verdict}</div>
          </div>
        </div>

        <div className="score-metric-box">
          <div className={`score-metric-val ${isFake ? 'text-fake' : 'text-real'}`}>
            {displayProb}%
          </div>
          <div className="score-metric-desc">{displayProbLabel}</div>
        </div>
      </div>

      {/* 1a. AI-image gate override */}
      {aiOverridden && (
        <div className="ai-gate-note" role="status">
          <AlertOctagon size={22} aria-hidden="true" />
          <div>
            <div className="ai-gate-score">Điểm ảnh AI: {aiScore}%</div>
            <div className="ai-gate-detail">
              Detector ảnh AI tổng quát vượt ngưỡng {(aiGate.threshold * 100).toFixed(1)}% nên kết luận là FAKE.
              {faceless
                ? ' Không tìm thấy khuôn mặt nên mô hình deepfake khuôn mặt (X2DFD) không áp dụng.'
                : ` Điểm deepfake khuôn mặt (X2DFD) chỉ đo ghép/sửa mặt: ${x2dfdFakeProb}%.`}
            </div>
          </div>
        </div>
      )}

      {/* 1a'. Blending-gate override */}
      {blendOverridden && (
        <div className="ai-gate-note" role="status">
          <AlertOctagon size={22} aria-hidden="true" />
          <div>
            <div className="ai-gate-score">Điểm ghép mặt (blending): {blendScore}%</div>
            <div className="ai-gate-detail">
              Expert blending vượt ngưỡng {(blendGate.threshold * 100).toFixed(1)}% nên kết luận là FAKE.
              {` Điểm deepfake khuôn mặt (X2DFD) ban đầu: ${x2dfdFakeProb}%.`}
            </div>
          </div>
        </div>
      )}

      {/* 1b. Low-reliability warning */}
      {isLowReliability && (
        <div className="reliability-warning" role="alert">
          <AlertTriangle size={20} className="reliability-warning-icon" aria-hidden="true" />
          <div>
            <div className="reliability-warning-title">
              Độ tin cậy thấp: ảnh có thể nằm ngoài dữ liệu mô hình đã học. Kết quả chỉ mang tính tham khảo.
            </div>
            {reliabilityReasons.length > 0 && (
              <ul className="reliability-warning-list">
                {reliabilityReasons.map((reason, i) => (
                  <li key={i}>{reason}</li>
                ))}
              </ul>
            )}
          </div>
        </div>
      )}

      {/* 2. AI Reasoning & Explanation Block (Strict Gate: Render only if non-null) */}
      {hasExplanation && (
        <div className="explanation-block">
          <div className="explanation-header">
            <Brain size={20} />
            <span>
              {aiOverridden
                ? 'Kết luận từ detector ảnh AI tổng quát'
                : blendOverridden
                  ? 'Kết luận từ expert blending (dấu vết ghép mặt)'
                  : 'Phân Tích Lý Luận Từ Mô Hình Đa Phương Thức (LLaVA Reasoning)'}
            </span>
          </div>
          <p className="explanation-body">{explanation}</p>
        </div>
      )}

      {/* 3. Detailed Multimodal Breakdown (absent when no face was found) */}
      {!faceless && (
      <div className="detail-grid">
        {/* Router4 Decision Card */}
        <div className="metric-card">
          <div className="metric-card-header">
            <span className="metric-card-title">1. Định Tuyến Router4</span>
            <Compass size={18} className="icon-accent" />
          </div>
          <div className="metric-card-val text-accent-soft">
            {result.router?.selected_alias || result.router?.selected_expert}
          </div>
          <div className="metric-card-sub">
            Độ tin cậy: <strong>{routerConf}%</strong> • Biên độ cách biệt: <strong>{margin}%</strong>
          </div>
        </div>

        {/* Selected Expert & Calibrator Card */}
        <div className="metric-card">
          <div className="metric-card-header">
            <span className="metric-card-title">2. Điểm Chuyên Gia & Hiệu Chuẩn</span>
            <Cpu size={18} className="icon-pink" />
          </div>
          <div className="metric-card-val">
            {calExpert}%
          </div>
          <div className="metric-card-sub">
            Điểm thô: <strong>{rawExpert}%</strong> • Hiệu chuẩn giáo viên: <strong>{calExpert}%</strong>
          </div>
        </div>

        {/* Final Continuous Score Card */}
        <div className="metric-card">
          <div className="metric-card-header">
            <span className="metric-card-title">3. Điểm deepfake khuôn mặt (X2DFD)</span>
            <CheckCircle2 size={18} className="icon-real" />
          </div>
          <div className="metric-card-val">
            {x2dfdFakeProb}% <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>/ 100%</span>
          </div>
          <div className="metric-card-sub">
            Xác thực: <strong>{x2dfdRealProb}% Thật</strong> • Ngưỡng: <strong>50.0%</strong>
          </div>
        </div>
      </div>
      )}

      {/* 4. Processing Timing & Traceability */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px 18px', background: 'var(--surface-faint)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)', fontSize: '0.82rem', color: 'var(--text-muted)', flexWrap: 'wrap', gap: '8px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Clock size={14} />
          <span>Thời gian xử lý: <strong>{result.timing_ms?.total || 0} ms</strong> (Định tuyến: {result.timing_ms?.routing || 0}ms, Chuyên gia: {result.timing_ms?.expert_inference || 0}ms, LLaVA: {result.timing_ms?.x2dfd_inference || 0}ms)</span>
        </div>
        <div style={{ fontFamily: 'var(--font-mono)' }}>
          Request ID: {result.request_id?.slice(0, 8)}...
        </div>
      </div>
    </div>
  );
}
