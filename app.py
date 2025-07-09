import os
import gc
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import numpy as np
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# Flask setup
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
WEIGHTS_FOLDER = 'weights'

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load default model once at startup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Default model architecture
base_model = RRDBNet(
    num_in_ch=3, num_out_ch=3,
    num_feat=64,
    num_block=23,
    num_grow_ch=32,
    scale=4
)

# Function to create upsampler per model/scale/tile setting
def get_upsampler(model_name, scale, tile, tile_pad, use_half):
    model_path = os.path.join(WEIGHTS_FOLDER, f"{model_name}.pth")
    if not os.path.exists(model_path):
        return None, f"Model file not found: {model_path}"
    
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64,
        num_block=6 if 'anime' in model_name else 23,
        num_grow_ch=32,
        scale=scale
    )
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

# Index Route: Image upload form
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image_file = request.files.get('image')
        model_name = request.form.get('model', 'RealESRGAN_x4plus')
        scale = int(request.form.get('scale', 4))
        tile_size = int(request.form.get('tile_size', 256))
        tile_pad = int(request.form.get('tile_pad', 10))
        use_half = True

        if not image_file:
            return render_template("index.html", error="Please upload an image.")

        # Save uploaded file
        filename = image_file.filename
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        image_file.save(input_path)

        # Load image
        try:
            img = Image.open(input_path).convert('RGB')
            img.thumbnail((512, 512))  # Resize to reduce RAM usage
            img_np = np.array(img)
        except:
            return render_template("index.html", error="Invalid image file.")

        # Load model
        upsampler, error = get_upsampler(model_name, scale, tile_size, tile_pad, use_half)
        if error:
            return render_template("index.html", error=error)

        try:
            # Enhance image
            output_np, _ = upsampler.enhance(img_np)
            output_img = Image.fromarray(output_np)
        except Exception as e:
            return render_template("index.html", error=f"Enhancement failed: {str(e)}")

        # Save output image
        output_path = os.path.join(RESULT_FOLDER, f"result_{filename}")
        output_img.save(output_path)

        # Free memory
        del img, img_np, output_np, output_img, upsampler
        gc.collect()
        torch.cuda.empty_cache()

        return render_template("result.html",
                               input_url=f"/{UPLOAD_FOLDER}/{filename}",
                               output_url=f"/{output_path}")
    return render_template("index.html")

# Run app

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
