# app.py
import os
import gc
import uuid
import threading
from flask import Flask, render_template, request, jsonify
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
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    torch.set_num_threads(1)

task_status = {}

def get_model_architecture(model_name, scale):
    if model_name == 'RealESRGAN_x4plus':
        return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
    elif model_name == 'RealESRGAN_x4plus_anime_6B':
        return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=scale)
    else:
        return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)

def enhance_image(task_id, input_path, output_path, model_name):
    try:
        img = Image.open(input_path).convert('RGB')
        img.thumbnail((800, 800))
        img_np = np.array(img)

        model = get_model_architecture(model_name, scale=4)
        model_path = os.path.join(WEIGHTS_FOLDER, f"{model_name}.pth")

        upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=256,
            tile_pad=10,
            pre_pad=0,
            half=True and torch.cuda.is_available(),
            device=device
        )

        output_np, _ = upsampler.enhance(img_np, outscale=4)
        output_img = Image.fromarray(output_np)
        output_img.save(output_path)

        task_status[task_id] = {
            "status": "done",
            "input_url": f"/{input_path}",
            "output_url": f"/{output_path}"
        }
    except Exception as e:
        task_status[task_id] = {"status": "error", "error": str(e)}
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/enhance', methods=['POST'])
def enhance():
    image_file = request.files.get('image')
    model_name = request.form.get('model')

    if not image_file:
        return jsonify({"status": "error", "error": "No image uploaded"}), 400

    task_id = uuid.uuid4().hex
    input_path = os.path.join(UPLOAD_FOLDER, f"{task_id}.jpg")
    output_path = os.path.join(RESULT_FOLDER, f"{task_id}.jpg")
    image_file.save(input_path)

    task_status[task_id] = {"status": "processing"}
    threading.Thread(target=enhance_image, args=(task_id, input_path, output_path, model_name)).start()

    return jsonify({"task_id": task_id}), 202

@app.route('/status/<task_id>', methods=['GET'])
def check_status(task_id):
    return jsonify(task_status.get(task_id, {"status": "unknown"}))

@app.route('/result/<task_id>', methods=['GET'])
def result(task_id):
    task = task_status.get(task_id)
    if not task or task.get("status") != "done":
        return "Result not available or still processing", 404
    return render_template("result.html", input_url=task["input_url"], output_url=task["output_url"])

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
