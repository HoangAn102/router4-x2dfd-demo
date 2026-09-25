/**
 * SafeVision API Client
 *
 * Frontend: Render
 * Real AI inference: PTNK GPU server
 */

const DEFAULT_API_ORIGIN = 'https://gains-how-cheque-enquiry.trycloudflare.com';

const API_ORIGIN = (
  import.meta.env.VITE_API_ORIGIN ||
  DEFAULT_API_ORIGIN
).replace(/\/+$/, '');

const API_BASE = `${API_ORIGIN}/api/v1`;

function resolveApiUrl(url) {
  if (!url) return url;

  if (/^https?:\/\//i.test(url)) {
    return url;
  }

  return `${API_ORIGIN}${url}`;
}

export class ApiError extends Error {
  constructor(message, errorDetail = {}) {
    super(message);

    this.name = 'ApiError';
    this.code =
      errorDetail.code || 'UNKNOWN_ERROR';
    this.stage =
      errorDetail.stage || 'client_request';
    this.requestId =
      errorDetail.request_id || null;
    this.details =
      errorDetail.details || null;
  }
}

export async function checkHealth() {
  try {
    const res = await fetch(
      `${API_BASE}/health`
    );

    if (!res.ok) {
      throw new Error(
        `Health check failed: ${res.status}`
      );
    }

    return await res.json();

  } catch (err) {
    console.warn(
      'Health check error:',
      err
    );

    return {
      status: 'offline',
      run_mode: 'unknown'
    };
  }
}

export async function checkReadiness() {
  try {
    const res = await fetch(
      `${API_BASE}/ready`
    );

    if (!res.ok) {
      throw new Error(
        `Readiness check failed: ${res.status}`
      );
    }

    return await res.json();

  } catch (err) {
    console.warn(
      'Readiness check error:',
      err
    );

    return {
      status: 'not_ready',
      run_mode: 'unknown',
      all_ready: false,
      components: {}
    };
  }
}

export async function analyzeImage(file) {
  const formData = new FormData();

  formData.append(
    'file',
    file
  );

  const res = await fetch(
    `${API_BASE}/analyze/image`,
    {
      method: 'POST',
      body: formData
    }
  );

  const data = await res
    .json()
    .catch(() => null);

  if (!res.ok) {
    const detail =
      data?.detail ||
      data?.error ||
      {};

    throw new ApiError(
      detail.message ||
      `Image analysis failed (${res.status})`,
      detail
    );
  }

  return data;
}

export async function analyzeVideo(
  file,
  onProgress = () => {},
  pollIntervalMs = 1200
) {
  const formData = new FormData();

  formData.append(
    'file',
    file
  );

  const submitRes = await fetch(
    `${API_BASE}/analyze/video`,
    {
      method: 'POST',
      body: formData
    }
  );

  const submitData = await submitRes
    .json()
    .catch(() => null);

  if (!submitRes.ok) {
    const detail =
      submitData?.detail ||
      submitData?.error ||
      {};

    throw new ApiError(
      detail.message ||
      `Video submit failed (${submitRes.status})`,
      detail
    );
  }

  const {
    job_id,
    poll_url
  } = submitData;

  if (!job_id) {
    throw new ApiError(
      'Backend did not return job_id.'
    );
  }

  onProgress({
    stage: 'queued',
    current: 0,
    total: 32,
    status: 'processing',
    message: 'Đã nhận video vào hàng đợi...'
  });

  return new Promise(
    (resolve, reject) => {

      const poll = async () => {
        try {
          const pollTarget = resolveApiUrl(
            poll_url ||
            `/api/v1/analyze/video/jobs/${job_id}`
          );

          const pollRes = await fetch(
            pollTarget
          );

          const job =
            await pollRes
              .json()
              .catch(() => null);

          if (!pollRes.ok) {
            const detail =
              job?.detail ||
              job?.error ||
              {};

            reject(
              new ApiError(
                detail.message ||
                `Polling failed (${pollRes.status})`,
                detail
              )
            );

            return;
          }

          const {
            status,
            stage,
            current_frame,
            total_frames,
            result,
            error
          } = job;

          if (status === 'processing') {
            onProgress({
              stage:
                stage ||
                'analyzing_frames',

              current:
                current_frame || 0,

              total:
                total_frames || 32,

              status:
                'processing'
            });

            setTimeout(
              poll,
              pollIntervalMs
            );

            return;
          }

          if (
            status === 'completed' ||
            status === 'blocked'
          ) {
            onProgress({
              stage: 'finished',
              current:
                total_frames || 32,
              total:
                total_frames || 32,
              status
            });

            resolve(result);

            return;
          }

          if (status === 'failed') {
            reject(
              new ApiError(
                error?.message ||
                'Video analysis failed.',
                error || {}
              )
            );

            return;
          }

          setTimeout(
            poll,
            pollIntervalMs
          );

        } catch (err) {
          reject(
            err instanceof ApiError
              ? err
              : new ApiError(
                  err.message
                )
          );
        }
      };

      setTimeout(
        poll,
        pollIntervalMs
      );
    }
  );
}
