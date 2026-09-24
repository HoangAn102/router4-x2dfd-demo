import React, { useState } from 'react';
import { Film, Clock, Eye, Layers } from 'lucide-react';

export default function FrameTimeline({ frames, onSeekFrame }) {
  const [selectedFrameIdx, setSelectedFrameIdx] = useState(0);

  if (!frames || frames.length === 0) return null;

  const currentFrame = frames[selectedFrameIdx] || frames[0];

  const handleFrameClick = (idx, timestamp) => {
    setSelectedFrameIdx(idx);
    if (onSeekFrame) {
      onSeekFrame(timestamp);
    }
  };

  return (
    <div className="timeline-section glass-panel">
      <div className="timeline-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Film size={22} color="#6366f1" />
          <h3 style={{ fontSize: '1.1rem', fontWeight: 700 }}>
            Dòng Thời Gian Giám Định Từng Frame ({frames.length} frames)
          </h3>
        </div>
        <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
          Nhấp vào frame để chuyển trình phát video tới giây tương ứng
        </div>
      </div>

      {/* Frame Detail Inspector Box */}
      {currentFrame && (
        <div style={{
          padding: '16px 20px',
          background: 'rgba(255, 255, 255, 0.03)',
          borderRadius: 'var(--radius-md)',
          border: '1px solid var(--border-subtle)',
          marginBottom: '20px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '16px',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            {currentFrame.thumbnail_b64 ? (
              <img
                src={currentFrame.thumbnail_b64.startsWith('data:') ? currentFrame.thumbnail_b64 : `data:image/jpeg;base64,${currentFrame.thumbnail_b64}`}
                alt={`Frame ${currentFrame.frame_index}`}
                style={{ width: '64px', height: '64px', objectFit: 'cover', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.1)' }}
              />
            ) : (
              <div style={{ width: '64px', height: '64px', background: '#000', borderRadius: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Eye size={20} color="var(--text-muted)" />
              </div>
            )}
            <div>
              <div style={{ fontSize: '1rem', fontWeight: 700 }}>
                Frame #{currentFrame.frame_index + 1} tại thời điểm <span style={{ color: '#a5b4fc', fontFamily: 'var(--font-mono)' }}>{currentFrame.timestamp_seconds.toFixed(2)}s</span>
              </div>
              <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                Chuyên gia Router4: <strong style={{ color: '#e2e8f0' }}>{currentFrame.selected_expert}</strong> • Điểm chuyên gia: <strong>{((currentFrame.expert_score || 0) * 100).toFixed(1)}%</strong>
              </div>
            </div>
          </div>

          <div style={{ textAlign: 'right' }}>
            <span className={`frame-verdict-tag ${currentFrame.verdict?.toLowerCase() === 'fake' ? 'fake' : 'real'}`} style={{ fontSize: '0.9rem', padding: '4px 12px' }}>
              {currentFrame.verdict} ({((currentFrame.final_frame_score || 0) * 100).toFixed(1)}%)
            </span>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '4px' }}>
              Điểm pháp y LLaVA
            </div>
          </div>
        </div>
      )}

      {/* 32 Frames Grid */}
      <div className="timeline-frames-grid">
        {frames.map((f) => {
          const isSelected = f.frame_index === selectedFrameIdx;
          const isFake = f.verdict?.toUpperCase() === 'FAKE';
          const scorePercent = ((f.final_frame_score || 0) * 100).toFixed(0);

          return (
            <div
              key={f.frame_index}
              className={`frame-thumbnail-card ${isSelected ? 'selected' : ''}`}
              onClick={() => handleFrameClick(f.frame_index, f.timestamp_seconds)}
            >
              {f.thumbnail_b64 ? (
                <img
                  src={f.thumbnail_b64.startsWith('data:') ? f.thumbnail_b64 : `data:image/jpeg;base64,${f.thumbnail_b64}`}
                  alt={`Frame ${f.frame_index}`}
                  className="frame-thumb-img"
                />
              ) : (
                <div className="frame-thumb-img" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Eye size={18} color="var(--text-muted)" />
                </div>
              )}

              <div className="frame-time">{f.timestamp_seconds.toFixed(1)}s</div>
              <div className={`frame-verdict-tag ${isFake ? 'fake' : 'real'}`}>
                {f.verdict} {scorePercent}%
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
