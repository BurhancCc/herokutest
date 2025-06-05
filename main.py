from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import base64

# FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model definieren
class ConvAutoencoderGray(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

# Getrainde modellen inladen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clf_model = ConvAutoencoderGray().to(device)
clf_model.load_state_dict(torch.load("autoencoder.pth", map_location=device))
clf_model.eval()

denoise_model = ConvAutoencoderGray().to(device)
denoise_model.load_state_dict(torch.load("autoencoder_denoise.pth", map_location=device))
denoise_model.eval()

# Treshhold definieren en preprocessing van foto's
THRESHOLD = 0.000087  # for anomaly detection based on MSE
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Endpoint om threshold door te geven aan de front-end
@app.get("/threshold")
def get_threshold():
    return {"threshold": THRESHOLD}

# Voorspellen! (En metrics meegeven)
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and preprocess image
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("L")
    img = transform(image).unsqueeze(0).to(device)

    # Compute errors
    with torch.no_grad():
        recon = clf_model(img)
        mse = torch.mean((recon - img) ** 2).item()
        mae = torch.mean(torch.abs(recon - img)).item()
        rmse = mse ** 0.5

    is_outlier = mse > THRESHOLD
    return {"is_outlier": is_outlier, "mse": mse, "mae": mae, "rmse": rmse}

# Denoisen! (En metrics meegeven)
@app.post("/denoise")
async def denoise(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(BytesIO(contents)).convert("L")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        recon = denoise_model(tensor)
        mse = torch.mean((recon - tensor) ** 2).item()
        mae = torch.mean(torch.abs(recon - tensor)).item()
        rmse = mse ** 0.5
    # Encode reconstruction as base64 PNG
    recon_img = recon.squeeze(0).cpu()
    buf = BytesIO()
    transforms.ToPILImage()(recon_img).save(buf, format="PNG")
    recon_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "reconstruction": recon_b64
    }