import os
import cv2
import numpy as np
import insightface
import requests
from insightface.app import FaceAnalysis
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='static')
CORS(app)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Lazy-loaded globals ---
# NOTHING heavy runs at import time anymore. Both models load on first
# use, after gunicorn has already bound the port. This is what fixes the
# "No open ports detected" / exit 137 crash.
face_app = None
swapper = None

MODEL_PATH = 'inswapper_128.onnx'
# NOTE: replace this with a URL you are legally permitted to redistribute
# the model from (your own storage, HF repo you control, etc). insightface
# does not officially host inswapper_128.onnx, so model_zoo can't auto-fetch it.
MODEL_DOWNLOAD_URL = os.environ.get('INSWAPPER_MODEL_URL', '')


def get_face_app():
    global face_app
    if face_app is None:
        face_app = FaceAnalysis(name='buffalo_l')
        # ctx_id=-1 explicitly requests CPU (no GPU available on Render),
        # det_size lowered from 640x640 to cut memory usage.
        face_app.prepare(ctx_id=-1, det_size=(320, 320))
    return face_app


def ensure_swapper_model():
    """Download inswapper_128.onnx on first use if it isn't already present."""
    if os.path.exists(MODEL_PATH):
        return
    if not MODEL_DOWNLOAD_URL:
        raise FileNotFoundError(
            "inswapper_128.onnx not found and INSWAPPER_MODEL_URL is not set. "
            "Set the INSWAPPER_MODEL_URL environment variable in Render to a "
            "direct download link for the model."
        )
    print(f"Downloading swapper model from {MODEL_DOWNLOAD_URL} ...")
    resp = requests.get(MODEL_DOWNLOAD_URL, stream=True, timeout=120)
    resp.raise_for_status()
    with open(MODEL_PATH, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Swapper model downloaded.")


def get_swapper():
    global swapper
    if swapper is None:
        ensure_swapper_model()
        swapper = insightface.model_zoo.get_model(MODEL_PATH, download=False, download_zip=False)
    return swapper


def perform_face_swap(source_path, target_path, output_path):
    source_img = cv2.imread(source_path)
    target_img = cv2.imread(target_path)

    detector = get_face_app()
    source_faces = detector.get(source_img)
    target_faces = detector.get(target_img)

    if not source_faces or not target_faces:
        raise ValueError("Could not detect faces in one or both of the provided images.")

    source_face = source_faces[0]
    target_face = target_faces[0]

    model = get_swapper()
    result_img = model.get(target_img, target_face, source_face, paste_back=True)

    cv2.imwrite(output_path, result_img)
    return output_path


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/swap', methods=['POST'])
def handle_swap():
    if 'source' not in request.files or 'target' not in request.files:
        return jsonify({'error': 'Both source and target images are required.'}), 400

    source_file = request.files['source']
    target_file = request.files['target']

    source_path = os.path.join(UPLOAD_FOLDER, 'source.jpg')
    target_path = os.path.join(UPLOAD_FOLDER, 'target.jpg')
    output_path = os.path.join(OUTPUT_FOLDER, 'result.jpg')

    source_file.save(source_path)
    target_file.save(target_path)

    try:
        perform_face_swap(source_path, target_path, output_path)
        return jsonify({
            'success': True,
            'result_url': '/output/result.jpg'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/output/<filename>')
def serve_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


@app.route('/healthz')
def healthz():
    # Cheap endpoint that doesn't touch the models — useful for confirming
    # the port is bound and the app is alive, separate from model state.
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
