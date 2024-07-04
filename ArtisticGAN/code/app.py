from flask import Flask, render_template, request
from io import BytesIO
import torch
import torch.nn as nn
import torchvision.utils as vutils
from PIL import Image
import numpy as np
import base64

# Create a Flask app
app = Flask(__name__)

# Assuming 'latent_size' is defined appropriately
latent_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to generate and return image bytes based on selected category
def generate_image_bytes(category):
    # Load the corresponding generator model based on the selected category
    generator_path = f'G_{category}.pth'
    generator = nn.Sequential(
        nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(True),
        nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(True),
        nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(True),
        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(True),
        nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
        nn.Tanh()
    ).to(device)
    generator.load_state_dict(torch.load(generator_path, map_location='cpu'))

    # Generate fake images
    fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)
    fake_images = generator(fixed_latent)
    fake_images_denorm = (fake_images + 1) / 2
    grid = vutils.make_grid(fake_images_denorm.cpu(), nrow=8, padding=2, normalize=False)
    grid_np = (grid.permute(1, 2, 0).detach().numpy() * 255).astype('uint8')
    img = Image.fromarray(grid_np)
   
    # Convert image to bytes
    img_bytes_io = BytesIO()
    img.save(img_bytes_io, format='PNG')
    img_bytes_io.seek(0)
    return img_bytes_io.getvalue()

# Route to render the form and handle image generation
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        category = request.form['category']
        image_bytes = generate_image_bytes(category)
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        return render_template('index.html', image=image_base64)
    else:
        return render_template('index.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)