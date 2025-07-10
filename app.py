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

# Config
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
WEIGHTS_FOLDER = 'weights'
MAX_IMAGE_SIZE = 2 * 1024 * 1024  # 2 MB upload limit
app.config['MAX_CONTENT_LENGTH'] = MAX_IMAGE_SIZE

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    torch.set_num_threads(1)

task_status = {}

# === AUTO CLEANUP FUNCTIONS ===

def delete_file_later(path, delay_seconds=600):
    """Delete a file after a delay (default: 10 min)."""
    def delete():
        time.sleep(delay_seconds)
        try:
            if os.path.exists(path):
                os.remove(path)
        except:
            pass
    threading.Thread(target=delete).start()

def delete_task_status_later(task_id, delay_seconds=600):
    """Remove task status from memory."""
    def delete():
        time.sleep(delay_seconds)
        task_status.pop(task_id, None)
    threading.Thread(target=delete).start()

# === MODEL LOADING ===

def get_model_architecture(model_name, scale):
    if model_name == 'RealESRGAN_x4plus':
        return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
    elif model_name == 'RealESRGAN_x4plus_anime_6B':
        return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=scale)
    else:
        return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)

# === ENHANCEMENT FUNCTION ===

def enhance_image(task_id, input_path, output_path, preview_input_path, model_name):
    try:
        with Image.open(input_path).convert('RGB') as img:
            img.thumbnail((384, 384))  # Smaller thumbnail to save RAM
            img_np = np.array(img)

            # Save compressed input preview (WebP)
            img.save(preview_input_path, format="WEBP", optimize=True, quality=75)

        # Free original image
        del img
        gc.collect()

        # === Load lightweight model only ===
        model = get_model_architecture(model_name, scale=4)
        model_path = os.path.join(WEIGHTS_FOLDER, f"{model_name}.pth")

        upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=64,               # 🔥 Critical: smaller tiles save RAM
            tile_pad=5,
            pre_pad=0,
            half=False,            # 🔥 Critical: set to False on CPU
            device=device
        )

        output_np, _ = upsampler.enhance(img_np, outscale=4)

        # Save enhanced image as compressed WebP
        output_img = Image.fromarray(output_np)
        output_img.thumbnail((768, 768))  # Cap output size
        output_img.save(output_path, format="WEBP", optimize=True, quality=80)

        # ✅ Update task status
        task_status[task_id] = {
            "status": "done",
            "input_url": f"/{preview_input_path}",
            "output_url": f"/{output_path}"
        }

        # 🔁 Schedule cleanup
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

# === ROUTES ===

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
    output_path = os.path.join(RESULT_FOLDER, f"{task_id}_output.webp")
    preview_input_path = os.path.join(RESULT_FOLDER, f"{task_id}_input.webp")

    try:
        image_file.save(input_path)
    except Exception as e:
        return jsonify({"status": "error", "error": f"Upload failed: {str(e)}"}), 400

    task_status[task_id] = {"status": "processing"}

    # Start enhancement in a background thread
    threading.Thread(target=enhance_image, args=(task_id, input_path, output_path, preview_input_path, model_name)).start()

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

# === RUN APP ===

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
    