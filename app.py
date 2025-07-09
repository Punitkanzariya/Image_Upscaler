import os
import gc
from flask import Flask, render_template, request
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

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    torch.set_num_threads(1)

# ✅ Correct architecture per model
def get_model_architecture(model_name, scale):
    if model_name == 'RealESRGAN_x4plus':
        return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
    elif model_name == 'RealESRGAN_x4plus_anime_6B':
        return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=scale)
    else:
        return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)

# Build upsampler
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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image_file = request.files.get('image')
        model_name = request.form.get('model', 'RealESRGAN_x4plus')
        scale = 4
        tile_size = 256
        tile_pad = 10
        use_half = True

        if not image_file:
            return render_template("index.html", error="Please upload an image.")

        filename = image_file.filename
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        image_file.save(input_path)

        try:
            img = Image.open(input_path).convert('RGB')
            max_size = 400 if not torch.cuda.is_available() else 800
            img.thumbnail((max_size, max_size))
            img_np = np.array(img)
        except Exception as e:
            return render_template("index.html", error=f"Invalid image: {str(e)}")

        upsampler, error = get_upsampler(model_name, scale, tile_size, tile_pad, use_half)
        if error:
            return render_template("index.html", error=error)

        try:
            output_np, _ = upsampler.enhance(img_np, outscale=scale)
            output_img = Image.fromarray(output_np)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                return render_template("index.html", error="Image too large. Try smaller.")
            return render_template("index.html", error=f"Enhancement failed: {str(e)}")
        except Exception as e:
            return render_template("index.html", error=f"Enhancement failed: {str(e)}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        output_path = os.path.join(RESULT_FOLDER, f"result_{filename}")
        output_img.save(output_path)

        del img, img_np, output_np, output_img
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return render_template("result.html",
                               input_url=f"/{UPLOAD_FOLDER}/{filename}",
                               output_url=f"/{output_path}")
    return render_template("index.html")

@app.route('/health')
def health_check():
    return {
        'status': 'healthy',
        'device': str(device),
        'cuda_available': torch.cuda.is_available(),
        'weights_folder_exists': os.path.exists(WEIGHTS_FOLDER)
    }

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
