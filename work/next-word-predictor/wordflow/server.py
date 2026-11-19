"""Dependency-free HTTP server for the WordFlow dashboard.

Serves the static single-page front end from ``web/`` and a small JSON API backed by a
:class:`~wordflow.service.Workspace`. Built on ``http.server`` so it runs anywhere
Python does, including the unmodified LearnFast ``lfllm`` container.
"""

from __future__ import annotations

import base64
import binascii
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .corpus import MAX_UPLOAD_BYTES, CorpusError
from .service import PROJECT_DIR, TOY_CORPUS, TrainingConfig, Workspace

WEB_DIR = PROJECT_DIR / "web"
MAX_BODY_BYTES = MAX_UPLOAD_BYTES * 2          # base64 inflates uploads by ~4/3


class BadRequest(ValueError):
    """A client error whose message is shown to the user."""


# ------------------------------------------------------------- parameter parsing


def _int(params: dict, name: str, default: int | None, low: int, high: int) -> int:
    value = params.get(name, default)
    if value is None or value == "":
        if default is None:
            raise BadRequest(f"'{name}' is required")
        value = default
    try:
        if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
            raise ValueError
        number = int(value)
    except (TypeError, ValueError):
        raise BadRequest(f"'{name}' must be a whole number") from None
    if not low <= number <= high:
        raise BadRequest(f"'{name}' must be between {low} and {high}")
    return number


def _float(params: dict, name: str, default: float, low: float, high: float) -> float:
    value = params.get(name, default)
    try:
        if isinstance(value, bool):
            raise ValueError
        number = float(value)
    except (TypeError, ValueError):
        raise BadRequest(f"'{name}' must be a number") from None
    if number != number or not low <= number <= high:
        raise BadRequest(f"'{name}' must be between {low:g} and {high:g}")
    return number


def _bool(params: dict, name: str, default: bool) -> bool:
    value = params.get(name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _text(params: dict, name: str, default: str = "", max_length: int = 5000) -> str:
    value = params.get(name, default)
    if not isinstance(value, str):
        raise BadRequest(f"'{name}' must be text")
    if len(value) > max_length:
        raise BadRequest(f"'{name}' is too long (limit {max_length} characters)")
    return value


# ------------------------------------------------------------------ API routing


def _order(workspace: Workspace, params: dict, low: int = 1) -> int:
    top = workspace.state.model.order
    return _int(params, "order", min(top, 3), low, top)


def handle_api(workspace: Workspace, method: str, route: str, params: dict) -> dict:
    """Dispatch one API call. Raises :class:`BadRequest` / ``ValueError`` on bad input."""
    if route == "state" and method == "GET":
        return workspace.overview()

    if route == "predict" and method == "POST":
        smoothing = _bool(params, "smoothing", True)
        return workspace.predict(
            text=_text(params, "text"),
            order=_order(workspace, params),
            top_k=_int(params, "top_k", 5, 1, 25),
            smoothing=smoothing,
            k=_float(params, "k", 1.0, 1e-6, 100.0) if smoothing else 0.0,
        )

    if route == "score" and method == "POST":
        smoothing = _bool(params, "smoothing", True)
        return workspace.score(
            text=_text(params, "text"),
            order=_order(workspace, params),
            k=_float(params, "k", 1.0, 1e-6, 100.0) if smoothing else 0.0,
            backoff=_bool(params, "backoff", False),
        )

    if route == "generate" and method == "POST":
        seed = params.get("seed")
        return workspace.generate(
            seed_text=_text(params, "seed_text"),
            order=_order(workspace, params),
            max_words=_int(params, "max_words", 30, 1, 200),
            temperature=_float(params, "temperature", 1.0, 0.05, 5.0),
            seed=None if seed in (None, "") else _int(params, "seed", None, 0, 2**31 - 1),
            k=_float(params, "k", 0.0, 0.0, 100.0),
            stop_at_eos=_bool(params, "stop_at_eos", True),
        )

    if route == "ngrams" and method == "GET":
        return workspace.ngrams(
            order=_order(workspace, params),
            query=_text(params, "q", max_length=200),
            limit=_int(params, "limit", 50, 1, 500),
            min_count=_int(params, "min_count", 1, 1, 10**9),
            include_special=_bool(params, "include_special", True),
        )

    if route == "distribution" and method == "POST":
        return workspace.distribution(
            context_text=_text(params, "context", max_length=500),
            order=_order(workspace, params),
            k=_float(params, "k", 1.0, 1e-6, 100.0),
            limit=_int(params, "limit", 15, 1, 200),
        )

    if route == "evaluate" and method == "GET":
        return workspace.evaluation(k=_float(params, "k", 1.0, 1e-6, 100.0))

    if route == "learn" and method == "POST":
        return workspace.learn(
            corpus_text=_text(params, "corpus_text", max_length=20_000) or TOY_CORPUS,
            test_sentence=_text(params, "test_sentence", max_length=500),
            context_word=_text(params, "context_word", max_length=100),
            target_word=_text(params, "target_word", max_length=100),
            k=_float(params, "k", 1.0, 1e-6, 100.0),
        )

    if route == "train" and method == "POST":
        config = TrainingConfig(
            max_order=_int(params, "max_order", 3, 1, 5),
            min_count=_int(params, "min_count", 1, 1, 1000),
            test_fraction=_float(params, "test_fraction", 0.2, 0.0, 0.9),
            seed=_int(params, "seed", 42, 0, 2**31 - 1),
        )
        line_mode = _text(params, "line_mode", "auto", 20)
        source = _text(params, "source", "sample", 20)
        if source == "sample":
            workspace.train_sample(config)
        elif source == "upload":
            try:
                data = base64.b64decode(_text(params, "content_base64", "", MAX_BODY_BYTES), validate=True)
            except (binascii.Error, ValueError):
                raise BadRequest("the uploaded file could not be read") from None
            workspace.train_upload(_text(params, "filename", "", 300), data, config, line_mode)
        else:
            raise BadRequest("source must be 'sample' or 'upload'")
        return workspace.overview()

    raise LookupError(f"unknown endpoint: {method} /api/{route}")


# ---------------------------------------------------------------- HTTP handler


def make_handler(workspace: Workspace) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "WordFlow/1.0"

        def log_message(self, format: str, *args) -> None:      # noqa: A002 - stdlib name
            if getattr(self.server, "verbose", False):
                super().log_message(format, *args)

        # -- responses

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8")

        def _api(self, method: str, route: str, params: dict) -> None:
            try:
                self._json(HTTPStatus.OK, handle_api(workspace, method, route, params))
            except LookupError as error:
                self._json(HTTPStatus.NOT_FOUND, {"error": str(error)})
            except (BadRequest, CorpusError, ValueError) as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            except Exception as error:      # never leak a traceback to the browser
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"internal error: {error}"})

        def _static(self, path: str) -> None:
            relative = "index.html" if path in ("", "/") else path.lstrip("/")
            target = (WEB_DIR / relative).resolve()
            if WEB_DIR.resolve() not in target.parents or not target.is_file():
                self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain; charset=utf-8")
                return
            content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            if content_type.startswith("text/") or content_type.endswith(("javascript", "json")):
                content_type += "; charset=utf-8"
            self._send(HTTPStatus.OK, target.read_bytes(), content_type)

        # -- verbs

        def do_GET(self) -> None:           # noqa: N802 - stdlib naming
            url = urlparse(self.path)
            if url.path.startswith("/api/"):
                params = {key: values[-1] for key, values in parse_qs(url.query).items()}
                self._api("GET", url.path[len("/api/"):], params)
            else:
                self._static(url.path)

        def do_POST(self) -> None:          # noqa: N802 - stdlib naming
            url = urlparse(self.path)
            if not url.path.startswith("/api/"):
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if length < 0 or length > MAX_BODY_BYTES:
                self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request body too large"})
                return
            try:
                params = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(params, dict):
                    raise ValueError
            except (ValueError, UnicodeDecodeError):
                self._json(HTTPStatus.BAD_REQUEST, {"error": "request body must be a JSON object"})
                return
            self._api("POST", url.path[len("/api/"):], params)

    return Handler


def create_server(
    workspace: Workspace, host: str = "127.0.0.1", port: int = 8501, verbose: bool = False
) -> ThreadingHTTPServer:
    """Build (but do not start) the HTTP server."""
    mimetypes.add_type("application/javascript", ".js")
    mimetypes.add_type("text/css", ".css")
    server = ThreadingHTTPServer((host, port), make_handler(workspace))
    server.daemon_threads = True
    server.verbose = verbose            # type: ignore[attr-defined]
    return server
