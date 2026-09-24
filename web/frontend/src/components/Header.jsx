import React from 'react';
import { ShieldCheck, AlertTriangle, Cpu, Activity } from 'lucide-react';

export default function Header({ runMode, isReady }) {
  const isMock = runMode === 'mock';

  return (
    <header className="site-header">
      <div className="header-inner">
        <div className="brand-section">
          <div className="brand-icon-wrapper">
            <ShieldCheck size={28} color="#ffffff" />
          </div>
          <div>
            <h1 className="brand-title">Router4 + X²-DFD</h1>
            <p className="brand-subtitle">Hệ thống Giám định Pháp y Deepfake Đa tầng</p>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div className={`status-pill ${isMock ? 'mock' : 'live'}`}>
            <span className="dot" />
            <span>{isMock ? 'Chế độ Mock (Dev)' : 'Máy chủ Live AI'}</span>
          </div>

          <div className={`status-pill ${isReady ? 'live' : 'mock'}`} title={isReady ? 'Hệ thống AI sẵn sàng' : 'Đang nạp mô hình'}>
            <Activity size={14} />
            <span>{isReady ? 'Sẵn sàng' : 'Khởi động...'}</span>
          </div>
        </div>
      </div>

      {isMock && (
        <div className="mock-alert-banner">
          <AlertTriangle size={20} style={{ flexShrink: 0 }} />
          <div>
            <strong>CẢNH BÁO MOCK DEVELOPMENT MODE:</strong> Website đang chạy với dữ liệu mô phỏng giả lập để kiểm thử giao diện. Để chạy AI thực tế, cần cấu hình <code>RUN_MODE=live</code> trên server GPU.
          </div>
        </div>
      )}
    </header>
  );
}
