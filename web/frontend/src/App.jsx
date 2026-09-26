import React, { useState, useEffect, useRef } from 'react';
import Header from './components/Header';
import Dropzone from './components/Dropzone';
import ProcessingStatus from './components/ProcessingStatus';
import ImageResultCard from './components/ImageResultCard';
import VideoResultCard from './components/VideoResultCard';
import ErrorModal from './components/ErrorModal';
import { checkHealth, checkReadiness, analyzeImage, analyzeVideo, ApiError } from './services/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('image');
  const [selectedFile, setSelectedFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(null);

  const [imageResult, setImageResult] = useState(null);
  const [videoResult, setVideoResult] = useState(null);
  const [error, setError] = useState(null);

  const [runMode, setRunMode] = useState('mock');
  const [isReady, setIsReady] = useState(false);

  const videoRef = useRef(null);

  // Initialize health & readiness check
  useEffect(() => {
    let mounted = true;

    async function fetchSystemStatus() {
      try {
        const health = await checkHealth();
        if (mounted && health?.run_mode) {
          setRunMode(health.run_mode);
        }

        const readiness = await checkReadiness();
        if (mounted) {
          setIsReady(Boolean(readiness?.all_ready));
        }
      } catch (e) {
        console.warn('System status fetch failed:', e);
      }
    }

    fetchSystemStatus();
    return () => {
      mounted = false;
    };
  }, []);

  const handleTabChange = (tab) => {
    setActiveTab(tab);
    setSelectedFile(null);
    setImageResult(null);
    setVideoResult(null);
    setError(null);
  };

  const handleFileSelect = (file) => {
    setSelectedFile(file);
    setImageResult(null);
    setVideoResult(null);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    // Do not spend phone/laptop resources decoding the local video
    // while the remote GPU is doing inference.
    if (activeTab === 'video' && videoRef.current) {
      try {
        videoRef.current.pause();
      } catch (_) {}
    }

    setIsLoading(true);
    setError(null);
    setProgress(null);

    try {
      if (activeTab === 'image') {
        const result = await analyzeImage(selectedFile);
        setImageResult(result);
      } else {
        const result = await analyzeVideo(selectedFile, (prog) => {
          setProgress(prog);
        });
        setVideoResult(result);
      }
    } catch (err) {
      console.error('Analysis error:', err);
      setError({
        message: err.message || 'Quá trình phân tích thất bại.',
        code: err.code || 'CLIENT_ERROR',
        stage: err.stage || 'request_pipeline',
        requestId: err.requestId || null,
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleSeekFrame = (timestampSeconds) => {
    if (videoRef.current) {
      videoRef.current.currentTime = timestampSeconds;
      videoRef.current.play().catch(() => {});
    }
  };

  return (
    <div className="app-container">
      {/* 1. Header with Mode Badge & Alert */}
      <Header runMode={runMode} isReady={isReady} />

      <main>
        {/* 2. Upload & Preview Zone */}
        <Dropzone
          activeTab={activeTab}
          onTabChange={handleTabChange}
          selectedFile={selectedFile}
          onFileSelect={handleFileSelect}
          onAnalyze={handleAnalyze}
          isLoading={isLoading}
          videoRef={videoRef}
        />

        {/* 3. Honest Progress & Radar Indicator */}
        {isLoading && (
          <ProcessingStatus
            isVideo={activeTab === 'video'}
            progress={progress}
          />
        )}

        {/* 4. Results */}
        {!isLoading && activeTab === 'image' && imageResult && (
          <ImageResultCard result={imageResult} />
        )}

        {!isLoading && activeTab === 'video' && videoResult && (
          <VideoResultCard
            result={videoResult}
            onSeekFrame={handleSeekFrame}
          />
        )}
      </main>

      {/* 5. Fail-Closed Error Modal */}
      <ErrorModal
        error={error}
        onClose={() => setError(null)}
      />
    </div>
  );
}
