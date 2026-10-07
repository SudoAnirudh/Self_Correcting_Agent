import os
import json
import glob
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Dict, Any, List
from agent import orchestrator
from agent.tools import ToolRouter
from agent.hitl import CheckpointManager


PORT = 8000


class CockpitHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler serving the Cockpit UI and live REST API endpoints."""

    def do_GET(self):
        url = self.path.split("?")[0]

        if url == "/api/traces":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            traces = []
            if os.path.exists("logs/traces.jsonl"):
                with open("logs/traces.jsonl", "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                traces.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
            self.wfile.write(json.dumps(traces).encode("utf-8"))
            return

        if url == "/api/checkpoints":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            checkpoints = []
            files = glob.glob("logs/checkpoints/*.json")
            for filepath in files:
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        checkpoints.append(json.load(f))
                except Exception:
                    pass
            self.wfile.write(json.dumps(checkpoints).encode("utf-8"))
            return

        if url == "/api/episodic":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            episodes = []
            if os.path.exists("logs/episodic_memory.json"):
                try:
                    with open("logs/episodic_memory.json", "r", encoding="utf-8") as f:
                        episodes = json.load(f)
                except Exception:
                    pass
            self.wfile.write(json.dumps(episodes).encode("utf-8"))
            return

        # Default static file handler (serves viewer/log_viewer.html)
        if url == "/" or url == "/viewer":
            self.path = "/viewer/log_viewer.html"

        return super().do_GET()

    def do_POST(self):
        if self.path == "/api/resume":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode("utf-8"))
                checkpoint_id = data.get("checkpoint_id")
                user_response = data.get("user_response", "Approved by user")

                tools = ToolRouter(seed=123, force_mocks=True)
                mem, output = orchestrator.resume_with_input(checkpoint_id, user_response, tools)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                resp_payload = {
                    "status": "resumed",
                    "checkpoint_id": checkpoint_id,
                    "final_output": output,
                    "memory": mem.snapshot(),
                }
                self.wfile.write(json.dumps(resp_payload).encode("utf-8"))
                return
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
                return

        self.send_response(404)
        self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def run_server(port: int = PORT):
    print(f"==================================================")
    print(f"🚀 Real-Time Viewer Cockpit Server running on:")
    print(f"   http://localhost:{port}/")
    print(f"==================================================")
    server = HTTPServer(("0.0.0.0", port), CockpitHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Cockpit Server...")
        server.server_close()


if __name__ == "__main__":
    run_server()
