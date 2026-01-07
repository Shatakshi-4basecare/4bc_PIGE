"""
Simple local web server to serve the PIGE graphs index.
Run this after generate_graphs_index.py to view the graphs locally.
"""

import http.server
import socketserver
import os
from pathlib import Path
from typing import Dict

config = {
    'graphs_dir': "23-12-2025_pige_graph_final/PIGE_graphs",
    'port': 8003,
    'host': "localhost",
}

def serve_directory(directory: Path, port: int = 8000, host: str = "localhost"):
    """Serve a directory with a simple HTTP server."""

    # Change to the target directory
    os.chdir(directory)

    # Create server
    handler = http.server.SimpleHTTPRequestHandler

    try:
        with socketserver.TCPServer((host, port), handler) as httpd:
            print(f"Serving PIGE graphs at: http://{host}:{port}")
            print(f"Directory: {directory.absolute()}")
            print("Press Ctrl+C to stop the server")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"Port {port} is already in use. Try a different port.")
        else:
            print(f"Error starting server: {e}")


def main(config: Dict):
    graphs_dir = Path(config['graphs_dir']).resolve()

    if not graphs_dir.exists():
        raise SystemExit(f"Directory not found: {graphs_dir}")

    # Check for index.html
    index_path = graphs_dir / "index.html"
    if not index_path.exists():
        print(f"Warning: index.html not found in {graphs_dir}")
        print("Make sure to run generate_graphs_index.py first.")

    serve_directory(graphs_dir, config['port'], config['host'])


if __name__ == "__main__":
    main(config)
