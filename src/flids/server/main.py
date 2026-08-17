"""Server entrypoint: load config, run uvicorn (single worker), dump metrics."""
from __future__ import annotations

import argparse
import threading
import time

import uvicorn
import yaml

from .app import ParameterServer, make_app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    server = ParameterServer(cfg)
    app = make_app(server)

    # watchdog: stop the process once training is done or wall-clock budget hit
    def watchdog():
        while not server.done:
            time.sleep(1.0)
            if server.t0 and (time.monotonic() - server.t0) > server.max_seconds:
                with server.cond:
                    server.done = True
                    server.cond.notify_all()
        # give clients a moment to observe done, then dump and exit
        time.sleep(2.0)
        server.dump()
        import os
        os._exit(0)

    threading.Thread(target=watchdog, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="warning")


if __name__ == "__main__":
    main()
