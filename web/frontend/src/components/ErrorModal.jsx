import React from 'react';
import { AlertCircle, X } from 'lucide-react';

export default function ErrorModal({ error, onClose }) {
  if (!error) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <AlertCircle size={28} />
          <h3 style={{ fontSize: '1.25rem', fontWeight: 800 }}>Yêu Cầu Không Thể Hoàn Tất</h3>
        </div>

        <div className="modal-body">
          {error.message || 'Đã xảy ra lỗi trong quá trình xử lý.'}
        </div>

        {(error.code || error.stage || error.requestId) && (
          <div className="modal-meta-box">
            {error.code && <div>Mã lỗi: <strong>{error.code}</strong></div>}
            {error.stage && <div>Giai đoạn: <strong>{error.stage}</strong></div>}
            {error.requestId && <div>Request ID: <strong>{error.requestId}</strong></div>}
          </div>
        )}

        <div style={{ textAlign: 'right' }}>
          <button className="btn-secondary" onClick={onClose}>
            <X size={16} />
            <span>Đóng thông báo</span>
          </button>
        </div>
      </div>
    </div>
  );
}
