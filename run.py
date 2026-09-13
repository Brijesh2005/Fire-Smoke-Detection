"""
CLI Launcher for Fire & Smoke Detection Web Application.
Usage:
    python run.py
    python run.py --port 8080 --host 0.0.0.0
"""

import argparse
import sys
import os

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app import app, logger

def main():
    parser = argparse.ArgumentParser(description="Fire & Smoke Detection Web Application")
    parser.add_argument("--host", default="127.0.0.1", help="Host IP address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Port to run on (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    print("=" * 65)
    print("  [FIRE & SMOKE DETECTION - AI SURVEILLANCE WEB APP]")
    print("=" * 65)
    print(f"  * Local Web Dashboard: http://{args.host}:{args.port}/")
    print(f"  * REST API Health:     http://{args.host}:{args.port}/api/health")
    print(f"  * Samples Endpoint:    http://{args.host}:{args.port}/api/samples")
    print("=" * 65)
    print("  Press Ctrl+C to stop the server.\n")

    try:
        app.run(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        print("\n[INFO] Server stopped by user. Goodbye!")
        sys.exit(0)

if __name__ == "__main__":
    main()
