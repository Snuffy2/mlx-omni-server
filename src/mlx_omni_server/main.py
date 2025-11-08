import argparse
import importlib
import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .middleware.logging import RequestResponseLoggingMiddleware
from .routers import api_router
from .utils.logger import logger, set_logger_level

app = FastAPI(title="MLX Omni Server")

# Add request/response logging middleware with custom levels
app.add_middleware(
    RequestResponseLoggingMiddleware,
    # exclude_paths=["/health"]
)

app.include_router(api_router)


def build_parser():
    """Create and configure the argument parser for the server."""
    parser = argparse.ArgumentParser(description="MLX Omni Server")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to, defaults to 0.0.0.0",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=10240,
        help="Port to bind the server to, defaults to 10240",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of workers to use, defaults to 1",
    )
    parser.add_argument(
        "--model-cache-max-size",
        type=int,
        default=3,
        help="Maximum number of MLX models to keep cached, defaults to 3",
    )
    parser.add_argument(
        "--model-cache-ttl-seconds",
        type=int,
        default=300,
        help="Time in seconds before idle MLX models expire from cache, defaults to 300. Set to 0 to disable expiration.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Set the logging level, defaults to info",
    )

    parser.add_argument(
        "--cors-allow-origins",
        type=str,
        default="",
        help='Apply origins to CORSMiddleware. This is useful for accessing the local server directly from the browser (use --cors-allow-origins="*"). Defaults to disabled',
    )
    return parser


def configure_cors_middleware(cors_allow_origins: str | None):
    """Configure CORS middleware with the provided origins, if any."""
    # Remove existing CORS middleware
    app.user_middleware = [m for m in app.user_middleware if m.cls != CORSMiddleware]
    app.middleware_stack = None  # Reset middleware stack to force rebuild

    if cors_allow_origins is None:
        origins = []
    else:
        # Add CORS middleware with provided origins or empty list
        origins = (
            [origin.strip() for origin in cors_allow_origins.split(",")]
            if cors_allow_origins
            else []
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


configure_cors_middleware(os.environ.get("MLX_OMNI_CORS", None))


def start():
    """Start the MLX Omni Server."""

    parser = build_parser()
    args = parser.parse_args()

    # Set log level through environment variable
    os.environ["MLX_OMNI_LOG_LEVEL"] = args.log_level
    # Set CORS through environment variable
    os.environ["MLX_OMNI_CORS"] = args.cors_allow_origins
    # Configure MLX model cache behavior via environment variables so the
    # global `wrapper_cache` (which reads env on import) will be created with
    # the requested settings when first imported.
    os.environ["MLX_OMNI_MODEL_CACHE_MAX_SIZE"] = str(args.model_cache_max_size)
    os.environ["MLX_OMNI_MODEL_CACHE_TTL_SECONDS"] = str(args.model_cache_ttl_seconds)

    # Import wrapper_cache lazily after env vars are set so its constructor
    # picks up the CLI configuration. Avoids needing runtime setters.
    importlib.import_module("mlx_omni_server.chat.mlx.wrapper_cache")

    set_logger_level(logger, args.log_level)
    configure_cors_middleware(args.cors_allow_origins)

    # Start server with uvicorn
    uvicorn.run(
        "mlx_omni_server.main:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        use_colors=True,
        workers=args.workers,
    )


if __name__ == "__main__":
    start()
