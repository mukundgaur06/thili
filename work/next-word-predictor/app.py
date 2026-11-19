"""Launch the WordFlow dashboard.

    python app.py                      # http://127.0.0.1:8501
    python app.py --port 8600
    python app.py --host 0.0.0.0       # inside Docker, so the published port works
    python app.py --corpus my.txt      # start with your own corpus instead of the sample
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wordflow.corpus import CorpusError, corpus_from_file  # noqa: E402
from wordflow.server import create_server  # noqa: E402
from wordflow.service import TrainingConfig, Workspace  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="WordFlow -- Statistical Language Modeling Laboratory")
    parser.add_argument("--host", default="127.0.0.1", help="interface to bind (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8501, help="port to listen on (default 8501)")
    parser.add_argument("--corpus", type=Path, help="train on this .txt file instead of the sample corpus")
    parser.add_argument("--open", action="store_true", help="open the dashboard in a browser")
    parser.add_argument("--verbose", action="store_true", help="log every HTTP request")
    args = parser.parse_args(argv)

    workspace = Workspace()
    try:
        if args.corpus:
            state = workspace.train(corpus_from_file(args.corpus), TrainingConfig())
        else:
            state = workspace.train_sample()
    except (CorpusError, ValueError) as error:
        print(f"Could not train the model: {error}", file=sys.stderr)
        return 2

    try:
        server = create_server(workspace, args.host, args.port, args.verbose)
    except OSError as error:
        print(f"Could not start on {args.host}:{args.port}: {error}", file=sys.stderr)
        print("Try another port, e.g. python app.py --port 8600", file=sys.stderr)
        return 2

    shown_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    url = f"http://{shown_host}:{args.port}/"
    print(f"WordFlow trained on '{state.corpus.name}': {state.model}")
    print(f"Dashboard running at {url}  (Ctrl+C to stop)")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
