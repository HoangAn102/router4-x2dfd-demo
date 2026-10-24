import React from 'react';
import { Loader2, Film, Scan, Layers } from 'lucide-react';

export default function ProcessingStatus({ isVideo, progress }) {
  const current = progress?.current || 0;
  const total = progress?.total || 32;
  const stage = progress?.stage || (isVideo ? 'queued' : 'analyzing');

  let stageLabel = 'Đang phân tích dữ liệu...';
  let stageIcon = <Scan size={24} className="icon-accent" />;

  if (isVideo) {
    if (stage === 'queued') {
      stageLabel = 'Đã nhận video, chuẩn bị trích xuất...';
      stageIcon = <Film size={24} className="icon-accent-soft" />;
    } else if (stage === 'extracting_frames') {
      stageLabel = 'Đang trích xuất 32 frame chuẩn DeepfakeBench...';
      stageIcon = <Film size={24} className="icon-accent" />;
    } else if (stage === 'routing_frames') {
      stageLabel = `Router4 đang định tuyến frame ${current} / ${total}...`;
      stageIcon = <Layers size={24} className="icon-accent" />;
    } else if (stage === 'scoring_experts') {
      stageLabel = `Đang chạy nhóm expert ${current} / ${total}...`;
      stageIcon = <Layers size={24} className="icon-blocked" />;
    } else if (stage === 'loading_x2dfd') {
      stageLabel = 'Đang chuẩn bị X²-DFD + LLaVA trên GPU...';
      stageIcon = <Loader2 size={24} className="radar-ring icon-violet" />;
    } else if (stage === 'analyzing_frames') {
      stageLabel = `X²-DFD đang giám định frame ${current} / ${total}...`;
      stageIcon = <Layers size={24} className="icon-pink" />;
    } else if (stage === 'aggregating_results') {
      stageLabel = 'Đang tổng hợp pháp y và kiểm định an toàn...';
      stageIcon = <Loader2 size={24} className="radar-ring icon-real" />;
    }
  } else {
    stageLabel = 'Đang giám định Router4 & X²-DFD LLaVA...';
    stageIcon = <Scan size={24} className="icon-violet" />;
  }

  const percent = total > 0 ? Math.round((current / total) * 100) : 0;

  return (
    <div className="processing-card glass-panel">
      <div className="radar-spinner">
        <div className="radar-ring" />
        <div className="radar-ring-inner" />
        {stageIcon}
      </div>

      <h3 style={{ fontSize: '1.2rem', fontWeight: 700, marginBottom: '8px' }}>
        {stageLabel}
      </h3>

      <p style={{ color: 'var(--text-secondary)', fontSize: '0.88rem' }}>
        {isVideo
          ? 'Router4 xử lý từng frame; các frame cùng expert được gom batch và X²-DFD tái sử dụng model đã nạp trên GPU.'
          : 'Hệ thống đang trích xuất đặc trưng pixel, tần số và suy luận đa phương thức.'}
      </p>

      {isVideo && ['routing_frames', 'scoring_experts', 'loading_x2dfd', 'analyzing_frames'].includes(stage) && (
        <>
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${percent}%` }} />
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
            Tiến độ thực tế: {current}/{total} frames ({percent}%)
          </div>
        </>
      )}
    </div>
  );
}
