"""Bundle Ketcher Standalone and serve its auxiliary assets on loopback."""

import base64
import functools
import glob
import http.server
import os
import re
import threading
import zipfile

import requests


KETCHER_DIR = os.path.abspath("data/ketcher_standalone")
KETCHER_ZIP_URL = "https://github.com/epam/ketcher/releases/download/v2.27.0/ketcher-standalone-2.27.0.zip"
_KETCHER_PORT = None


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()


def _ensure_ketcher_bundle():
    """
    Downloads EPAM Ketcher Standalone once, starts a silent local HTTP server for
    multi-chunk/WebWorker assets, and builds a DOM-safe base64 bundle.
    """
    global _KETCHER_PORT
    index_candidates = glob.glob(os.path.join(KETCHER_DIR, "**", "index.html"), recursive=True)
    if not index_candidates:
        os.makedirs(KETCHER_DIR, exist_ok=True)
        zip_path = os.path.join(KETCHER_DIR, "ketcher.zip")
        r = requests.get(KETCHER_ZIP_URL, timeout=60)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            f.write(r.content)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(KETCHER_DIR)
        os.remove(zip_path)
        index_candidates = glob.glob(os.path.join(KETCHER_DIR, "**", "index.html"), recursive=True)
    index_path = index_candidates[0]
    root_dir = os.path.dirname(index_path)

    if _KETCHER_PORT is None:
        try:
            handler = functools.partial(_QuietHandler, directory=root_dir)
            httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
            _KETCHER_PORT = httpd.server_address[1]
            t = threading.Thread(target=httpd.serve_forever, daemon=True)
            t.start()
        except Exception:
            _KETCHER_PORT = 0

    with open(index_path, "r", encoding="utf-8") as f:
        html_str = f.read()

    def _inline_css(m):
        rel = m.group(1).lstrip("./").lstrip("/")
        p = os.path.join(root_dir, rel)
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8", errors="ignore") as cf:
                return f"<style>{cf.read()}</style>"
        return m.group(0)

    html_str = re.sub(r'<link[^>]+href=["\']([^"\']+\.css)["\'][^>]*>', _inline_css, html_str)
    deferred_scripts = []

    def _extract_js(m):
        rel = m.group(1).lstrip("./").lstrip("/")
        p = os.path.join(root_dir, rel)
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8", errors="ignore") as jf:
                safe_js = jf.read().replace("</script>", "<\\/script>")
                deferred_scripts.append(f"<script>{safe_js}</script>")
            return ""
        return m.group(0)

    html_str = re.sub(r'<script[^>]+src=["\']([^"\']+\.js)["\'][^>]*>\s*</script>', _extract_js, html_str)
    if deferred_scripts:
        scripts_block = "\n".join(deferred_scripts)
        if "</body>" in html_str:
            html_str = html_str.replace("</body>", f"{scripts_block}\n</body>", 1)
        else:
            html_str += scripts_block
    b64_str = base64.b64encode(html_str.encode("utf-8")).decode("ascii")
    return b64_str, _KETCHER_PORT
