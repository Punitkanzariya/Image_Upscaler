import os
import gc
import uuid
import time
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

# Folders
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
WEIGHTS_FOLDER = 'weights'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    torch.set_num_threads(1)

# Task tracking
task_status = {}

# ============ Helpers ============

def delete_file_later(path, delay=600):
    def _delete():
        time.sleep(delay)
        try:
            os.remove(path)
        except:
            pass
    threading.Thread(target=_delete).start()

def delete_task_status_later(task_id, delay=600):
    def _delete():
        time.sleep(delay)
        task_status.pop(task_id, None)
    threading.Thread(target=_delete).start()

def get_model_architecture(name, scale):
    if name == 'RealESRGAN_x4plus_anime_6B':
        return RRDBNet(3, 3, 64, 6, 32, scale)
    return RRDBNet(3, 3, 64, 23, 32, scale)

def enhance_image(task_id, input_path, output_path, preview_input_path, model_name):
    try:
        # Load and shrink image
        img = Image.open(input_path).convert('RGB')
        img.thumbnail((512, 512))
        img_np = np.array(img)

        # Save input preview
        img.save(preview_input_path, format="WEBP", quality=80)

        # Load model
        model = get_model_architecture(model_name, scale=4)
        model_path = os.path.join(WEIGHTS_FOLDER, f"{model_name}.pth")
        upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=128,
            tile_pad=10,
            pre_pad=0,
            half=torch.cuda.is_available(),
            device=device
        )

        # Enhance
        output_np, _ = upsampler.enhance(img_np, outscale=4)
        output_img = Image.fromarray(output_np)
        output_img.thumbnail((1024, 1024))
        output_img.save(output_path, format="WEBP", quality=85)

        task_status[task_id] = {
            "status": "done",
            "input_url": f"/{preview_input_path}",
            "output_url": f"/{output_path}"
        }

        # Schedule cleanup
        delete_file_later(preview_input_path)
        delete_file_later(output_path)
        delete_task_status_later(task_id)

    except Exception as e:
        task_status[task_id] = {"status": "error", "error": str(e)}
    finally:
        try:
            os.remove(input_path)
        except:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ============ Routes ============

@app.route('/')
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
    preview_input_path = os.path.join(RESULT_FOLDER, f"{task_id}_input.webp")
    output_path = os.path.join(RESULT_FOLDER, f"{task_id}_output.webp")
    image_file.save(input_path)

    task_status[task_id] = {"status": "processing"}
    threading.Thread(
        target=enhance_image,
        args=(task_id, input_path, output_path, preview_input_path, model_name)
    ).start()

    return jsonify({"task_id": task_id}), 202

@app.route('/status/<task_id>')
def check_status(task_id):
    return jsonify(task_status.get(task_id, {"status": "unknown"}))

@app.route('/result/<task_id>')
def result(task_id):
    task = task_status.get(task_id)
    if not task or task.get("status") != "done":
        return "Processing or result not found.", 404
    return render_template("result.html", input_url=task["input_url"], output_url=task["output_url"])

# ============ Entry Point ============
@app.route('/health')
def health():
    return "OK", 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
