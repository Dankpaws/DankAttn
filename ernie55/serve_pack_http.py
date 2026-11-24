#!/usr/bin/env python3
"""Multi-threaded static HTTP server for streaming DankAttn packs over LAN to Mac Studio."""
import argparse
import functools
import http.server
import socketserver


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"[pack-http] {self.address_string()} - {format % args}", flush=True)


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9876)
    args = ap.parse_args()
    handler = functools.partial(QuietHandler, directory=args.directory)
    srv = ThreadedHTTPServer((args.host, args.port), handler)
    print(f"Serving {args.directory} on http://{args.host}:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
