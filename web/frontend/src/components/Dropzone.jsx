import React, { useEffect, useRef, useState } from 'react';
import { UploadCloud, Image as ImageIcon, Video as VideoIcon, X, Sparkles, AlertCircle } from 'lucide-react';

const MAX_IMAGE_MB = 20;
const MAX_VIDEO_MB = 100;

export default function Dropzone({
  activeTab,
  onTabChange,
  selectedFile,
  onFileSelect,
  onAnalyze,
  isLoading,
  videoRef,
}) {
  const [isDragActive, setIsDragActive] = useState(false);
  const [validationError, setValidationError] = useState(null);
  const fileInputRef = useRef(null);

  // Stable preview URL.
  //
  // Previously URL.createObjectURL(selectedFile) ran inside JSX.
  // Every progress update caused a React re-render and created a new
  // blob URL, forcing the <video> decoder to reload/reseek.
  const [previewUrl, setPreviewUrl] = useState(null);

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl(null);
      return;
    }

    const url = URL.createObjectURL(selectedFile);
    setPreviewUrl(url);

    return () => {
      URL.revokeObjectURL(url);
    };
  }, [selectedFile]);

  const isImageTab = activeTab === 'image';
  const acceptedTypes = isImageTab
    ? 'image/jpeg,image/png,image/webp'
    : 'video/mp4,video/avi,video/quicktime,video/x-matroska';

  const validateAndSetFile = (file) => {
    setValidationError(null);
    if (!file) return;

    const sizeMB = file.size / (1024 * 1024);
    if (isImageTab) {
      if (!file.type.startsWith('image/')) {
        setValidationError('Vui lòng chọn tệp hình ảnh hợp lệ (JPEG, PNG, WEBP).');
        return;
      }
      if (sizeMB > MAX_IMAGE_MB) {
        setValidationError(`Dung lượng ảnh (${sizeMB.toFixed(1)}MB) vượt quá giới hạn ${MAX_IMAGE_MB}MB.`);
        return;
      }
    } else {
      if (!file.type.startsWith('video/') && !file.name.match(/\.(mp4|avi|mov|mkv)$/i)) {
        setValidationError('Vui lòng chọn tệp video hợp lệ (MP4, AVI, MOV, MKV).');
        return;
      }
      if (sizeMB > MAX_VIDEO_MB) {
        setValidationError(`Dung lượng video (${sizeMB.toFixed(1)}MB) vượt quá giới hạn ${MAX_VIDEO_MB}MB.`);
        return;
      }
    }

    onFileSelect(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragActive(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragActive(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const handleInputChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      validateAndSetFile(e.target.files[0]);
    }
  };

  const clearSelection = () => {
    onFileSelect(null);
    setValidationError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <div>
      {/* Navigation Tabs */}
      <div className="tab-nav">
        <button
          className={`tab-btn ${isImageTab ? 'active' : ''}`}
          onClick={() => {
            if (!isLoading) {
              onTabChange('image');
              clearSelection();
            }
          }}
          disabled={isLoading}
        >
          <ImageIcon size={18} />
          <span>Giám định Ảnh</span>
        </button>

        <button
          className={`tab-btn ${!isImageTab ? 'active' : ''}`}
          onClick={() => {
            if (!isLoading) {
              onTabChange('video');
              clearSelection();
            }
          }}
          disabled={isLoading}
        >
          <VideoIcon size={18} />
          <span>Giám định Video (32 Frames)</span>
        </button>
      </div>

      {/* Drag & Drop Area */}
      {!selectedFile ? (
        <div
          className={`dropzone-container glass-panel ${isDragActive ? 'drag-active' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept={acceptedTypes}
            style={{ display: 'none' }}
            onChange={handleInputChange}
          />

          <div className="dropzone-icon-box">
            <UploadCloud size={32} />
          </div>

          <h3 className="dropzone-title">
            Kéo thả {isImageTab ? 'ảnh' : 'video'} vào đây hoặc nhấp để chọn tệp
          </h3>
          <p className="dropzone-subtitle">
            Hỗ trợ giám định kỹ thuật số đa tầng với độ chính xác cao
          </p>

          <div className="dropzone-limits">
            <span>Định dạng: {isImageTab ? 'JPG, PNG, WEBP' : 'MP4, AVI, MOV, MKV'}</span>
            <span>•</span>
            <span>Tối đa: {isImageTab ? `${MAX_IMAGE_MB}MB` : `${MAX_VIDEO_MB}MB (≤ 60s)`}</span>
          </div>

          {validationError && (
            <div style={{ marginTop: '16px', color: '#fb7185', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', fontSize: '0.88rem' }}>
              <AlertCircle size={16} />
              <span>{validationError}</span>
            </div>
          )}
        </div>
      ) : (
        /* Preview Card */
        <div className="preview-card glass-panel">
          <div className="preview-media-wrapper">
            {isImageTab ? (
              <img
                src={previewUrl || undefined}
                alt="Upload preview"
              />
            ) : (
              <video
                ref={videoRef}
                src={previewUrl || undefined}
                controls
                playsInline
                preload="metadata"
              />
            )}
          </div>

          <div className="preview-meta">
            <div>
              <strong>{selectedFile.name}</strong> ({(selectedFile.size / (1024 * 1024)).toFixed(2)} MB)
            </div>

            {!isLoading && (
              <button className="btn-secondary" onClick={clearSelection}>
                <X size={16} />
                <span>Đổi tệp</span>
              </button>
            )}
          </div>

          <div style={{ marginTop: '20px' }}>
            <button
              className="btn-primary"
              onClick={onAnalyze}
              disabled={isLoading}
            >
              <Sparkles size={18} />
              <span>{isLoading ? 'Đang Phân Tích...' : 'Bắt Đầu Giám Định AI'}</span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
