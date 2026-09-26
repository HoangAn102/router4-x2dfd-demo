/**
 * SafeVision browser API.
 *
 * Browser -> SAME ORIGIN Render -> GPU proxy -> PTNK GPU.
 */

const API_BASE = '/gpu-api/api/v1';


function proxyPollUrl(url) {
  if (!url) return url;

  // GPU backend normally returns /api/v1/...
  if (url.startsWith('/api/v1/')) {
    return `/gpu-api${url}`;
  }

  // Defensive support if an absolute upstream URL is returned.
  if (/^https?:\/\//i.test(url)) {
    const parsed = new URL(url);
    return `/gpu-api${parsed.pathname}${parsed.search}`;
  }

  return url;
}


export class ApiError extends Error {
  constructor(message, detail = {}) {
    super(message);

    this.name = 'ApiError';

    this.code =
      detail.code ||
      'UNKNOWN_ERROR';

    this.stage =
      detail.stage ||
      'client_request';

    this.requestId =
      detail.request_id ||
      null;

    this.details =
      detail.details ||
      null;
  }
}


async function safeFetch(url, options = {}) {
  try {
    return await fetch(
      url,
      options
    );

  } catch (err) {

    throw new ApiError(
      'Không thể kết nối đến máy chủ giám định AI.',
      {
        code: 'NETWORK_ERROR',
        stage: 'request_pipeline',
        details:
          err?.message ||
          'Browser network error',
      }
    );
  }
}


function extractDetail(data) {
  return (
    data?.detail ||
    data?.error ||
    {}
  );
}


export async function checkHealth() {
  try {
    const res = await safeFetch(
      `${API_BASE}/health`
    );

    if (!res.ok) {
      throw new Error(
        `Health ${res.status}`
      );
    }

    return await res.json();

  } catch (err) {

    console.warn(
      'Health check:',
      err
    );

    return {
      status: 'offline',
      run_mode: 'unknown',
    };
  }
}


export async function checkReadiness() {
  try {
    const res = await safeFetch(
      `${API_BASE}/ready`
    );

    if (!res.ok) {
      throw new Error(
        `Ready ${res.status}`
      );
    }

    return await res.json();

  } catch (err) {

    console.warn(
      'Readiness:',
      err
    );

    return {
      status: 'not_ready',
      run_mode: 'unknown',
      all_ready: false,
      components: {},
    };
  }
}


export async function analyzeImage(file) {
  const body = new FormData();

  body.append(
    'file',
    file
  );

  const res = await safeFetch(
    `${API_BASE}/analyze/image`,
    {
      method: 'POST',
      body,
    }
  );

  const data = await res
    .json()
    .catch(() => null);

  if (!res.ok) {

    const detail =
      extractDetail(data);

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
  const body = new FormData();

  body.append(
    'file',
    file
  );

  const submitRes = await safeFetch(
    `${API_BASE}/analyze/video`,
    {
      method: 'POST',
      body,
    }
  );

  const submitData = await submitRes
    .json()
    .catch(() => null);

  if (!submitRes.ok) {

    const detail =
      extractDetail(
        submitData
      );

    throw new ApiError(
      detail.message ||
      `Video submit failed (${submitRes.status})`,
      detail
    );
  }

  const {
    job_id,
    poll_url,
  } = submitData;

  if (!job_id) {
    throw new ApiError(
      'Backend did not return job_id.',
      {
        code: 'MISSING_JOB_ID',
        stage: 'video_submit',
      }
    );
  }

  onProgress({
    stage: 'queued',
    current: 0,
    total: 32,
    status: 'processing',
    message: 'Đã nhận video...',
  });

  return new Promise(
    (resolve, reject) => {

      const poll = async () => {

        try {

          const target = proxyPollUrl(
            poll_url ||
            `/api/v1/analyze/video/jobs/${job_id}`
          );

          const res = await safeFetch(
            target
          );

          const job = await res
            .json()
            .catch(() => null);

          if (!res.ok) {

            const detail =
              extractDetail(job);

            reject(
              new ApiError(
                detail.message ||
                `Polling failed (${res.status})`,
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
            error,
          } = job;

          if (
            status === 'processing'
          ) {

            onProgress({
              stage:
                stage ||
                'analyzing_frames',

              current:
                current_frame || 0,

              total:
                total_frames || 32,

              status:
                'processing',
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
              status,
            });

            resolve(result);

            return;
          }

          if (
            status === 'failed'
          ) {

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
                  err?.message ||
                  'Video polling error',
                  {
                    code:
                      'VIDEO_POLL_ERROR',
                    stage:
                      'request_pipeline',
                  }
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
