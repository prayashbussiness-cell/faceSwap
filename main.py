import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='static')
CORS(app)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize InsightFace Analysis Engine
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Load Swapper Model
MODEL_PATH = 'inswapper_128.onnx'
swapper = None

def get_swapper():
    global swapper
    if swapper is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("inswapper_128.onnx not found. Please ensure build script downloaded it.")
        swapper = insightface.model_zoo.get_model(MODEL_PATH, download=False, download_zip=False)
    return swapper

def perform_face_swap(source_path, target_path, output_path):
    source_img = cv2.imread(source_path)
    target_img = cv2.imread(target_path)

    source_faces = face_app.get(source_img)
    target_faces = face_app.get(target_img)

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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)