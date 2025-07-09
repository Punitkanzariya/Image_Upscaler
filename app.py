import os
import gc
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import numpy as np
import torch
from realesrgan import RealESRGANer
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
except ImportError:
    # Fallback for older versions
    from realesrgan.archs.rrdbnet_arch import RRDBNet

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

# Optimize for CPU deployment
if not torch.cuda.is_available():
    torch.set_num_threads(1)  # Limit CPU threads for memory efficiency

# Function to get the right model architecture based on model name
def get_model_architecture(model_name, scale):
    """Get the appropriate model architecture based on model name"""
    if 'anime' in model_name.lower():
        # For anime models, use fewer blocks
        return RRDBNet(
            num_in_ch=3, num_out_ch=3,
            num_feat=64,
            num_block=6,
            num_grow_ch=32,
            scale=scale
        )
    elif 'gfpgan' in model_name.lower() or 'face' in model_name.lower():
        # For face restoration models
        return RRDBNet(
            num_in_ch=3, num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=scale
        )
    else:
        # Default RealESRGAN model
        return RRDBNet(
            num_in_ch=3, num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=scale
        )

# Function to create upsampler per model/scale/tile setting
def get_upsampler(model_name, scale, tile, tile_pad, use_half):
    model_path = os.path.join(WEIGHTS_FOLDER, f"{model_name}.pth")
    if not os.path.exists(model_path):
        return None, f"Model file not found: {model_path}"
    
    try:
        # Get the appropriate model architecture
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

        # Load image with size limits for memory safety
        try:
            img = Image.open(input_path).convert('RGB')
            # More aggressive size limiting for Render
            max_size = 400 if not torch.cuda.is_available() else 800
            img.thumbnail((max_size, max_size))
            img_np = np.array(img)
        except Exception as e:
            return render_template("index.html", error=f"Invalid image file: {str(e)}")

        # Load model
        upsampler, error = get_upsampler(model_name, scale, tile_size, tile_pad, use_half)
        if error:
            return render_template("index.html", error=error)

        try:
            # Enhance image with memory monitoring
            output_np, _ = upsampler.enhance(img_np, outscale=scale)
            output_img = Image.fromarray(output_np)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                return render_template("index.html", error="Image too large for processing. Please upload a smaller image.")
            else:
                return render_template("index.html", error=f"Enhancement failed: {str(e)}")
        except Exception as e:
            return render_template("index.html", error=f"Enhancement failed: {str(e)}")
        finally:
            # Clean up upsampler
            if 'upsampler' in locals():
                del upsampler
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # Save output image
        output_path = os.path.join(RESULT_FOLDER, f"result_{filename}")
        output_img.save(output_path)

        # Free memory
        del img, img_np, output_np, output_img
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return render_template("result.html",
                               input_url=f"/{UPLOAD_FOLDER}/{filename}",
                               output_url=f"/{output_path}")
    return render_template("index.html")

# Route to list available models
@app.route('/models')
def list_models():
    """List available model files"""
    if not os.path.exists(WEIGHTS_FOLDER):
        return {'models': [], 'error': 'Weights folder not found'}
    
    models = []
    for file in os.listdir(WEIGHTS_FOLDER):
        if file.endswith('.pth'):
            models.append(file.replace('.pth', ''))
    
    return {'models': models}

# Health check route
@app.route('/health')
def health_check():
    return {
        'status': 'healthy', 
        'device': str(device),
        'cuda_available': torch.cuda.is_available(),
        'weights_folder_exists': os.path.exists(WEIGHTS_FOLDER)
    }

# Run app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)