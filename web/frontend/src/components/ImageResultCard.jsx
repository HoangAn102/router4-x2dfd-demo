import React from 'react';
import { ShieldCheck, AlertOctagon, Brain, Cpu, Compass, Clock, CheckCircle2 } from 'lucide-react';

export default function ImageResultCard({ result }) {
  if (!result || !result.final) return null;

  const isFake = result.final.verdict === 'FAKE';
  const fakeProb = (result.final.fake_probability * 100).toFixed(1);
  const realProb = (result.final.real_probability * 100).toFixed(1);
  const routerConf = ((result.router?.router_confidence || 0) * 100).toFixed(1);
  const margin = ((result.router?.margin || 0) * 100).toFixed(1);
  const rawExpert = ((result.expert?.raw_score || 0) * 100).toFixed(1);
  const calExpert = ((result.expert?.calibrated_score || 0) * 100).toFixed(1);

  // Strict explanation rule: Render only if explanation exists and is non-empty string
  const explanation = result.final.explanation;
  const hasExplanation = typeof explanation === 'string' && explanation.trim().length > 0;

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
          <div className="score-metric-val" style={{ color: isFake ? '#fb7185' : '#34d399' }}>
            {fakeProb}%
          </div>
          <div className="score-metric-desc">Xác suất Giả mạo (P_fake)</div>
        </div>
      </div>

      {/* 2. AI Reasoning & Explanation Block (Strict Gate: Render only if non-null) */}
      {hasExplanation && (
        <div className="explanation-block">
          <div className="explanation-header">
            <Brain size={20} />
            <span>Phân Tích Lý Luận Từ Mô Hình Đa Phương Thức (LLaVA Reasoning)</span>
          </div>
          <p className="explanation-body">{explanation}</p>
        </div>
      )}

      {/* 3. Detailed Multimodal Breakdown */}
      <div className="detail-grid">
        {/* Router4 Decision Card */}
        <div className="metric-card">
          <div className="metric-card-header">
            <span className="metric-card-title">1. Định Tuyến Router4</span>
            <Compass size={18} color="#6366f1" />
          </div>
          <div className="metric-card-val" style={{ color: '#a5b4fc' }}>
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
            <Cpu size={18} color="#ec4899" />
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
            <span className="metric-card-title">3. Điểm Pháp Y Cuối (X²-DFD)</span>
            <CheckCircle2 size={18} color="#10b981" />
          </div>
          <div className="metric-card-val">
            {fakeProb}% <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>/ 100%</span>
          </div>
          <div className="metric-card-sub">
            Xác thực: <strong>{realProb}% Thật</strong> • Ngưỡng: <strong>50.0%</strong>
          </div>
        </div>
      </div>

      {/* 4. Processing Timing & Traceability */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px 18px', background: 'rgba(255,255,255,0.02)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)', fontSize: '0.82rem', color: 'var(--text-muted)', flexWrap: 'wrap', gap: '8px' }}>
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
