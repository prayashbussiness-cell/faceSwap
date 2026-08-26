import os
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

app = Flask(__name__, static_folder="static", static_url_path="")

ALLOWED = {"png", "jpg", "jpeg", "webp"}

def allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED

@app.get("/")
def home():
    return send_from_directory(app.static_folder, "index.html")

@app.post("/api/swap")
def swap():
    source = request.files.get("source")
    target = request.files.get("target")

    if not source or not target:
        return jsonify(success=False, error="Both source and target images are required."), 400

    if not allowed(source.filename) or not allowed(target.filename):
        return jsonify(success=False, error="Unsupported image format."), 400

    job_id = uuid.uuid4().hex
    source_path = UPLOAD_DIR / f"{job_id}_source_{secure_filename(source.filename)}"
    target_path = UPLOAD_DIR / f"{job_id}_target_{secure_filename(target.filename)}"
    source.save(source_path)
    target.save(target_path)

    # IMPORTANT:
    # This is only a deployment-ready demo backend. It does NOT perform
    # an actual face swap. Replace this section with your chosen face-swap
    # model/API call, then save the generated image to RESULT_DIR.
    result_name = f"{job_id}_result{target_path.suffix.lower()}"
    result_path = RESULT_DIR / result_name
    target_path.replace(result_path)

    return jsonify(success=True, result_url=f"/results/{result_name}")

@app.get("/results/<path:filename>")
def results(filename):
    return send_from_directory(RESULT_DIR, filename)

@app.get("/health")
def health():
    return jsonify(status="ok")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
