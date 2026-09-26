import httpx

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from ..config import settings
from ..utils.logger import logger


router = APIRouter(
    prefix="/gpu-api",
    tags=["GPU Reverse Proxy"],
)


_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}


async def _close_upstream(
    response: httpx.Response,
    client: httpx.AsyncClient,
):
    try:
        await response.aclose()
    finally:
        await client.aclose()


@router.api_route(
    "/{path:path}",
    methods=[
        "GET",
        "POST",
        "PUT",
        "PATCH",
        "DELETE",
        "OPTIONS",
        "HEAD",
    ],
)
async def proxy_gpu_api(
    path: str,
    request: Request,
):
    """
    Same-origin browser -> Render -> GPU proxy.

    Prevents client browsers from needing to contact an ephemeral
    trycloudflare origin directly.
    """

    if not path.startswith("api/v1/"):
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "code": "PROXY_PATH_NOT_ALLOWED",
                    "stage": "render_proxy",
                    "message": "Only /api/v1/* GPU API paths are allowed.",
                }
            },
        )

    origin = (
        settings.GPU_API_ORIGIN
        .strip()
        .rstrip("/")
    )

    if not origin:
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "code": "GPU_UPSTREAM_UNCONFIGURED",
                    "stage": "render_proxy",
                    "message": "GPU upstream is not configured.",
                }
            },
        )

    upstream_url = (
        f"{origin}/{path}"
    )

    request_headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in _HOP_BY_HOP
        and k.lower() != "origin"
    }

    timeout = httpx.Timeout(
        connect=15.0,
        read=3600.0,
        write=3600.0,
        pool=15.0,
    )

    client = httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
    )

    kwargs = {
        "method": request.method,
        "url": upstream_url,
        "params": request.query_params.multi_items(),
        "headers": request_headers,
    }

    if request.method not in {
        "GET",
        "HEAD",
    }:
        kwargs["content"] = request.stream()

    upstream_request = (
        client.build_request(
            **kwargs
        )
    )

    try:

        upstream = await client.send(
            upstream_request,
            stream=True,
        )

    except httpx.HTTPError as e:

        await client.aclose()

        logger.error(
            "GPU proxy connection failed: %s",
            e,
        )

        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "code": "GPU_UPSTREAM_UNREACHABLE",
                    "stage": "render_proxy",
                    "message": (
                        "GPU inference server is temporarily unreachable."
                    ),
                    "details": type(e).__name__,
                }
            },
        )

    response_headers = {}

    for key in (
        "cache-control",
        "x-request-id",
    ):
        if key in upstream.headers:
            response_headers[key] = (
                upstream.headers[key]
            )

    media_type = (
        upstream.headers.get(
            "content-type"
        )
    )

    return StreamingResponse(
        upstream.aiter_raw(),
        status_code=upstream.status_code,
        headers=response_headers,
        media_type=media_type,
        background=BackgroundTask(
            _close_upstream,
            upstream,
            client,
        ),
    )
