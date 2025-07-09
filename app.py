# async_image_upscaler.py
import os
import gc
import uuid
import threading
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image
import numpy as np
import torch
from realesrgan import RealESRGANer

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
except ImportError:
    from realesrgan.archs.rrdbnet_arch import RRDBNet

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
WEIGHTS_FOLDER = 'weights'
TASKS = {}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    torch.set_num_threads(1)

def get_model_architecture(model_name, scale):
    if model_name == 'RealESRGAN_x4plus':
        return RRDBNet(3, 3, 64, 23, 32, scale)
    elif model_name == 'RealESRGAN_x4plus_anime_6B':
        return RRDBNet(3, 3, 64, 6, 32, scale)
    else:
        return RRDBNet(3, 3, 64, 23, 32, scale)

def get_upsampler(model_name, scale, tile, tile_pad, use_half):
    model_path = os.path.join(WEIGHTS_FOLDER, f"{model_name}.pth")
    if not os.path.exists(model_path):
        return None, f"Model file not found: {model_path}"
    try:
        model = get_model_architecture(model_name, scale)
        upsampler = RealESRGANer(
            scale=scale,
            model_path=model_path,
            model=model,
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=0,
            half=use_half and torch.cuda.is_available(),
            device=device
        )
        return upsampler, None
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

def enhance_async(task_id, image_path, model_name):
    try:
        img = Image.open(image_path).convert('RGB')
        max_size = 400 if not torch.cuda.is_available() else 800
        img.thumbnail((max_size, max_size))
        img_np = np.array(img)

        upsampler, error = get_upsampler(model_name, 4, 128, 8, True)
        if error:
            TASKS[task_id] = {'status': 'failed', 'error': error}
            return

        output_np, _ = upsampler.enhance(img_np, outscale=4)
        output_img = Image.fromarray(output_np)

        output_filename = f"result_{os.path.basename(image_path)}"
        output_path = os.path.join(RESULT_FOLDER, output_filename)
        output_img.save(output_path)

        TASKS[task_id] = {'status': 'done', 'result': output_filename}

    except Exception as e:
        TASKS[task_id] = {'status': 'failed', 'error': str(e)}
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")

@app.route('/enhance', methods=['POST'])
def enhance():
    image_file = request.files.get('image')
    model_name = request.form.get('model', 'RealESRGAN_x4plus')
    if not image_file:
        return jsonify({'error': 'No image uploaded'}), 400

    filename = f"{uuid.uuid4().hex}_{image_file.filename}"
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    image_file.save(input_path)

    task_id = uuid.uuid4().hex
    TASKS[task_id] = {'status': 'processing'}
    threading.Thread(target=enhance_async, args=(task_id, input_path, model_name)).start()
    return jsonify({'task_id': task_id}), 202

@app.route('/status/<task_id>')
def task_status(task_id):
    return jsonify(TASKS.get(task_id, {'status': 'not_found'}))

@app.route('/result/<filename>')
def get_result(filename):
    return send_from_directory(RESULT_FOLDER, filename)

@app.route('/health')
def health():
    return {'status': 'ok', 'device': str(device)}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
