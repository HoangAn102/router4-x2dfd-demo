import React from 'react';
import { AlertTriangle, CheckCircle, Film, BarChart3, Clock, AlertCircle } from 'lucide-react';
import FrameTimeline from './FrameTimeline';

export default function VideoResultCard({ result, onSeekFrame }) {
  if (!result) return null;

  const isBlocked = result.status === 'blocked';
  const frames = result.frames || [];
  const fakeFrames = frames.filter((f) => f.verdict?.toUpperCase() === 'FAKE');
  const realFrames = frames.filter((f) => f.verdict?.toUpperCase() === 'REAL');

  const avgFakeScore =
    frames.length > 0
      ? (frames.reduce((acc, f) => acc + (f.final_frame_score || 0), 0) / frames.length) * 100
      : 0;

  return (
    <div className="result-container">
      {/* 1. Status Banner */}
      {isBlocked ? (
        <div className="verdict-banner blocked">
          <div className="verdict-badge-box">
            <div className="verdict-icon">
              <AlertTriangle size={36} />
            </div>
            <div>
              <div className="verdict-text-label">Trạng Thái Tổng Hợp Video</div>
              <div className="verdict-title">BLOCKED (CHƯA XÁC MINH)</div>
            </div>
          </div>

          <div className="score-metric-box">
            <div className="score-metric-val text-blocked">
              {fakeFrames.length}/{frames.length}
            </div>
            <div className="score-metric-desc">Frames Giả Mạo Phát Hiện</div>
          </div>
        </div>
      ) : (
        <div className={`verdict-banner ${result.final?.verdict === 'FAKE' ? 'fake' : 'real'}`}>
          <div className="verdict-badge-box">
            <div className="verdict-icon">
              <CheckCircle size={36} />
            </div>
            <div>
              <div className="verdict-text-label">Phán Quyết Tổng Hợp Video</div>
              <div className="verdict-title">{result.final?.verdict}</div>
            </div>
          </div>

          <div className="score-metric-box">
            <div className="score-metric-val">
              {((result.final?.fake_probability || 0) * 100).toFixed(1)}%
            </div>
            <div className="score-metric-desc">Xác suất Giả mạo Tổng hợp</div>
          </div>
        </div>
      )}

      {/* 2. Fail-Closed Explanatory Notice */}
      {isBlocked && (
        <div className="blocked-notice">
          <AlertCircle size={20} className="blocked-notice-icon" />
          <div>
            <strong>Nguyên tắc Trung thực Khoa học (Fail-Closed Invariant):</strong>
            <p style={{ marginTop: '4px' }}>
              Mã nguồn nghiên cứu chưa công bố thuật toán tổng hợp chính thức cho cấp độ video. Hệ thống tuyệt đối từ chối tự bịa phán quyết hoặc điểm số tổng hợp cuối cho video. Toàn bộ <strong>{frames.length} kết quả giám định của từng frame</strong> được lưu trữ và hiển thị minh bạch bên dưới.
            </p>
          </div>
        </div>
      )}

      {/* 3. Video Metadata Summary Cards */}
      <div className="detail-grid">
        <div className="metric-card">
          <div className="metric-card-header">
            <span className="metric-card-title">Thông Tin Video</span>
            <Film size={18} className="icon-accent" />
          </div>
          <div className="metric-card-val">
            {result.video?.duration_seconds || 0}s
          </div>
          <div className="metric-card-sub">
            Tốc độ khung hình: <strong>{result.video?.fps || 0} FPS</strong> • Giao thức: <strong>{result.video?.sampling_protocol || 'uniform'}</strong>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-card-header">
            <span className="metric-card-title">Phân Bổ Khung Hình</span>
            <BarChart3 size={18} className="icon-pink" />
          </div>
          <div className={`metric-card-val ${fakeFrames.length > 0 ? 'text-fake' : 'text-real'}`}>
            {fakeFrames.length} FAKE <span style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>/ {realFrames.length} REAL</span>
          </div>
          <div className="metric-card-sub">
            Trung bình điểm giả mạo từng frame: <strong>{avgFakeScore.toFixed(1)}%</strong>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-card-header">
            <span className="metric-card-title">Thời Gian Giám Định</span>
            <Clock size={18} className="icon-real" />
          </div>
          <div className="metric-card-val">
            {((result.timing_ms?.total || 0) / 1000).toFixed(1)}s
          </div>
          <div className="metric-card-sub">
            Tổng cộng 32 frames đã xử lý qua Router4 & LLaVA
          </div>
        </div>
      </div>

      {/* 4. Interactive Frame Timeline */}
      <FrameTimeline frames={frames} onSeekFrame={onSeekFrame} />
    </div>
  );
}
