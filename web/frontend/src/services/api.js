/**
 * API Client for Router4 + X²-DFD Web Backend.
 * Handles synchronous image analysis and asynchronous video job polling.
 */

const API_BASE = '/api/v1';

export class ApiError extends Error {
  constructor(message, errorDetail = {}) {
    super(message);
    this.name = 'ApiError';
    this.code = errorDetail.code || 'UNKNOWN_ERROR';
    this.stage = errorDetail.stage || 'client_request';
    this.requestId = errorDetail.request_id || null;
    this.details = errorDetail.details || null;
  }
}

/**
 * Perform a health check probe.
 */
export async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn('Health check error:', err);
    return { status: 'offline', run_mode: 'unknown' };
  }
}

/**
 * Check backend readiness.
 */
export async function checkReadiness() {
  try {
    const res = await fetch(`${API_BASE}/ready`);
    if (!res.ok) throw new Error(`Readiness check failed: ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn('Readiness check error:', err);
    return { ready: false, mode: 'unknown', components: {} };
  }
}

/**
 * Synchronous Image Analysis.
 */
export async function analyzeImage(file) {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/analyze/image`, {
    method: 'POST',
    body: formData,
  });

  const data = await res.json().catch(() => null);

  if (!res.ok) {
    const detail = data?.detail || {};
    throw new ApiError(detail.message || `Image analysis failed (${res.status})`, detail);
  }

  return data;
}

/**
 * Asynchronous Video Analysis with Polling Loop.
 * @param {File} file - The video file.
 * @param {Function} onProgress - Callback receiving status updates: { stage, current, total, status }.
 * @param {number} pollIntervalMs - Polling interval in ms (default 1200ms).
 */
export async function analyzeVideo(file, onProgress = () => {}, pollIntervalMs = 1200) {
  const formData = new FormData();
  formData.append('file', file);

  // 1. Submit video job (HTTP 202 Accepted)
  const submitRes = await fetch(`${API_BASE}/analyze/video`, {
    method: 'POST',
    body: formData,
  });

  const submitData = await submitRes.json().catch(() => null);

  if (!submitRes.ok) {
    const detail = submitData?.detail || {};
    throw new ApiError(detail.message || `Failed to submit video (${submitRes.status})`, detail);
  }

  const { job_id, poll_url } = submitData;
  if (!job_id) {
    throw new ApiError('Server response did not include a valid job_id.');
  }

  onProgress({
    stage: 'queued',
    current: 0,
    total: 32,
    status: 'processing',
    message: 'Đã nhận video vào hàng đợi...',
  });

  // 2. Polling Loop
  return new Promise((resolve, reject) => {
    const poll = async () => {
      try {
        const pollRes = await fetch(poll_url || `${API_BASE}/analyze/video/jobs/${job_id}`);
        const jobStatus = await pollRes.json().catch(() => null);

        if (!pollRes.ok) {
          const detail = jobStatus?.detail || {};
          reject(new ApiError(detail.message || `Polling failed (${pollRes.status})`, detail));
          return;
        }

        const { status, stage, current_frame, total_frames, result, error } = jobStatus;

        if (status === 'processing') {
          onProgress({
            stage: stage || 'analyzing_frames',
            current: current_frame || 0,
            total: total_frames || 32,
            status: 'processing',
          });
          setTimeout(poll, pollIntervalMs);
          return;
        }

        if (status === 'completed' || status === 'blocked') {
          onProgress({
            stage: 'finished',
            current: total_frames || 32,
            total: total_frames || 32,
            status,
          });
          resolve(result);
          return;
        }

        if (status === 'failed') {
          reject(new ApiError(error?.message || 'Video analysis encountered an internal failure.', error || {}));
          return;
        }

        // Unknown status fallback
        setTimeout(poll, pollIntervalMs);
      } catch (err) {
        reject(err instanceof ApiError ? err : new ApiError(err.message));
      }
    };

    setTimeout(poll, pollIntervalMs);
  });
}
