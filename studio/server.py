"""Live backend for the studio page: real policy + ngspice runs, streamed as Server-Sent Events.

    python -m studio.server                 # then open http://localhost:8765

GET  /               the studio page (docs/demo/autoanalog_studio.html)
GET  /api/health     is the live engine up, which checkpoint, is Claude available
GET  /api/run?...    one run as an event stream; query keys are the Spec fields plus race=0|1
POST /api/parse      {"text": "..."} -> spec values from a plain-language request (needs ANTHROPIC_API_KEY)

One run at a time: starting a new run stops the one in progress.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from studio.engine import CORNERS, ROOT, Engine, Spec

PAGE = ROOT / "docs" / "demo" / "autoanalog_studio.html"

PARSE_SYSTEM = """You translate an engineer's plain-language request for a PCIe Gen 2 (5 Gbps) receiver
equalizer into specification targets for a sizing agent. Fill every field; keep the defaults for anything the
request does not mention. Defaults and sensible ranges:
- eye_height_v: minimum eye opening after the channel, default 0.25 V, range 0.15-0.50 ("wide/big/clean eye" -> 0.30-0.35)
- eye_width_ui: minimum horizontal eye opening, default 0.70 UI, range 0.50-0.95
- power_mw: power budget, default 2.0 mW, range 0.5-2.4 ("low power" -> 1.2, "ultra low power" -> 0.9)
- peaking_min_db: minimum high-frequency boost at Nyquist, default 3 dB, range 1-8 ("strong equalisation/lossy channel" -> 5)
- corner: "random" unless the request names conditions; hot/slow/worst case -> "SS_0.95V_125C", fast/cold -> "FF_1.05V_0C", typical -> "TT_1.00V_62.5C"
- note: one short sentence saying what you set and why."""


def claude_spec(text: str, model: str) -> dict:
    import anthropic
    from pydantic import BaseModel

    class Request(BaseModel):
        eye_height_v: float
        eye_width_ui: float
        power_mw: float
        peaking_min_db: float
        corner: str
        note: str

    response = anthropic.Anthropic().messages.parse(
        model=model,
        max_tokens=1000,
        system=PARSE_SYSTEM,
        messages=[{"role": "user", "content": text.strip()}],
        output_format=Request,
    )
    parsed = response.parsed_output.model_dump()
    clamp = {"eye_height_v": (0.15, 0.5), "eye_width_ui": (0.5, 0.95), "power_mw": (0.5, 2.4), "peaking_min_db": (1.0, 8.0)}
    for name, (low, high) in clamp.items():
        parsed[name] = round(min(high, max(low, float(parsed[name]))), 3)
    if parsed["corner"] not in CORNERS:
        parsed["corner"] = "random"
    parsed["source"] = f"claude:{model}"
    return parsed


class Studio:
    def __init__(self, model: str) -> None:
        self.engine = Engine()
        self.model = model
        self.busy = threading.Lock()


def make_handler(studio: Studio):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A002 - keep the console to run summaries
            return

        def _headers(self, code: int, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Cache-Control", "no-store")

        def _json(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self._headers(code, "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):  # noqa: N802
            self._headers(204, "text/plain")
            self.end_headers()

        def do_GET(self):  # noqa: N802
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                if not PAGE.exists():
                    return self._json(404, {"error": f"build the page first: python -m studio.build_page ({PAGE} missing)"})
                body = PAGE.read_bytes()
                self._headers(200, "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                return self.wfile.write(body)
            if url.path == "/api/health":
                return self._json(200, {"live": True, "checkpoint": studio.engine.checkpoint.name,
                                        "run": studio.engine.base.get("output_dir"), "corners": sorted(CORNERS),
                                        "claude": bool(os.environ.get("ANTHROPIC_API_KEY")), "busy": studio.busy.locked()})
            if url.path == "/api/run":
                return self._run({key: values[0] for key, values in parse_qs(url.query).items()})
            return self._json(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if urlparse(self.path).path != "/api/parse":
                return self._json(404, {"error": "not found"})
            length = int(self.headers.get("Content-Length") or 0)
            text = json.loads(self.rfile.read(length) or b"{}").get("text", "")
            if not os.environ.get("ANTHROPIC_API_KEY"):
                return self._json(503, {"error": "ANTHROPIC_API_KEY not set; the page falls back to its keyword parser"})
            try:
                return self._json(200, claude_spec(text, studio.model))
            except Exception as error:  # noqa: BLE001 - the page falls back to keywords on any failure
                return self._json(502, {"error": f"{type(error).__name__}: {error}"})

        def _run(self, query: dict) -> None:
            engine = studio.engine
            if not studio.busy.acquire(blocking=False):
                engine.stop.set()  # a newer request wins
                if not studio.busy.acquire(timeout=60):
                    return self._json(409, {"error": "a run is still stopping; try again"})
            engine.stop.clear()
            events: queue.Queue = queue.Queue()

            def work():
                try:
                    engine.run(Spec.from_mapping(query), events.put, race=query.get("race", "1") != "0")
                except Exception as error:  # noqa: BLE001 - surface to the page instead of hanging it
                    events.put({"type": "error", "message": f"{type(error).__name__}: {error}"})
                finally:
                    events.put(None)

            worker = threading.Thread(target=work, daemon=True)
            try:
                # HTTP/1.0 with no length: the stream ends when the connection closes.
                self._headers(200, "text/event-stream")
                self.end_headers()
                worker.start()
                while True:
                    try:
                        event = events.get(timeout=10)
                    except queue.Empty:
                        self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
                        continue
                    if event is None:
                        break
                    self.wfile.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    if event["type"] == "done" and event["lane"] == "sac" and event["best"]:
                        best = event["best"]["metrics"]
                        print(f"run: policy feasible at sim {event['first_feasible']} "
                              f"(eye {best.get('eye_vertical_v')} V)", flush=True)
            except OSError:
                engine.stop.set()  # the page went away
            finally:
                if worker.is_alive():
                    worker.join()
                studio.busy.release()

    return Handler


class QuietServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        """Browsers drop idle connections all the time; only report real failures."""
        import sys

        if isinstance(sys.exc_info()[1], (ConnectionAbortedError, ConnectionResetError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--model", default="claude-sonnet-5", help="Claude model for plain-language requests")
    args = parser.parse_args()
    print("loading the policy ...", flush=True)
    studio = Studio(args.model)
    server = QuietServer((args.host, args.port), make_handler(studio))
    print(f"AutoAnalog-RL studio live on http://localhost:{args.port}  "
          f"(policy {studio.engine.checkpoint.name}, Claude {'on' if os.environ.get('ANTHROPIC_API_KEY') else 'off'})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        studio.engine.stop.set()


if __name__ == "__main__":
    main()
