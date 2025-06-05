import streamlit as st
import requests
from PIL import Image, ImageFilter
import io
import base64
from ultralytics import YOLO
import numpy as np
import os

# FastAPI backend
API_URL = "http://127.0.0.1:8000"

# App titel en uitleg
st.title("Autoencoder Anomaliedetectie Dashboard van Burhan")
st.markdown(
    "Upload een afbeelding of bekijk de voorbeelden hieronder. Voor elke afbeelding wordtde MSE, MAE en RMSE van de reconstructie getoond. "
    "Bij een gedetecteerde anomalie kan optioneel een gezicht worden gedetecteerd met objectdetectie en worden geblurred met Gaussian Blur."
)

# Denoiser
st.header("1️⃣ Image denoising")
uploaded_denoise = st.file_uploader(
    "Upload afbeelding voor denoising", type=["png","jpg","jpeg"], key="uploader_denoise"
)
if uploaded_denoise:
    img = Image.open(uploaded_denoise).convert("RGB")
    st.image(img, caption="Origineel (noisy)", use_container_width=True)

    if st.button("Denoise Afbeelding", key="btn_denoise"):
        files = {"file": (uploaded_denoise.name, uploaded_denoise.getvalue(), uploaded_denoise.type)}
        resp = requests.post(f"{API_URL}/denoise", files=files)
        if resp.ok:
            data = resp.json()
            st.write(f"**MSE:** {data['mse']:.6f}")
            st.write(f"**MAE:** {data['mae']:.6f}")
            st.write(f"**RMSE:** {data['rmse']:.6f}")
            recon = base64.b64decode(data['reconstruction'])
            recon_img = Image.open(io.BytesIO(recon))
            st.image(recon_img, caption="Denoised Output", use_container_width=True)
        else:
            st.error(f"Fout bij denoising: {resp.status_code}")

# Voorbeelden outliers met metrics
st.header("2️⃣ Voorbeelden van Anomaliedetectie")
example_dir = "examples/"
if os.path.isdir(example_dir):
    files = [f for f in os.listdir(example_dir) if os.path.isfile(os.path.join(example_dir, f))]
    if files:
        cols = st.columns(4)
        for idx, fname in enumerate(files[:8]):
            path = os.path.join(example_dir, fname)
            try:
                img = Image.open(path).convert("RGB")
            except Exception as e:
                cols[idx%4].warning(f"Kon {fname} niet openen: {e}")
                continue
            cols[idx%4].image(img, caption=fname, use_container_width=True)
            buf = io.BytesIO()
            img_gray = img.convert("L").resize((64,64))
            img_gray.save(buf, format="PNG")
            try:
                resp = requests.post(f"{API_URL}/predict", files={"file": (fname, buf.getvalue(), "image/png")}, timeout=5)
                resp.raise_for_status()
                d = resp.json()
                cols[idx%4].write(f"MSE: {d['mse']:.6f}")
                cols[idx%4].write(f"MAE: {d['mae']:.6f}")
                cols[idx%4].write(f"RMSE: {d['rmse']:.6f}")
                cols[idx%4].write(f"Anomalie: {'Ja' if d['is_outlier'] else 'Nee'}")
            except Exception as e:
                cols[idx%4].write(f"Error: {e}")
    else:
        st.write("Geen voorbeelden gevonden in `examples/`.")
else:
    st.write("Maak `examples/` dir met testbeelden voor anomaliedetectie.")

# Voorspelling!
st.header("3️⃣ Voorspelling anomaliedetectie")
if 'anom_done' not in st.session_state:
    st.session_state['anom_done'] = False

uploaded_anom = st.file_uploader("Upload afbeelding voor anomaliedetectie", type=["png","jpg","jpeg"], key="uploader_anom")
if uploaded_anom:
    st.session_state['full_image'] = Image.open(uploaded_anom).convert("RGB")
    st.image(st.session_state['full_image'], caption="Origineel", use_container_width=True)

    if st.button("Voorspel Anomalie", key="predict_anom_btn"):
        buf2 = io.BytesIO()
        img_gray = st.session_state['full_image'].convert("L").resize((64,64))
        img_gray.save(buf2, format="PNG")
        resp2 = requests.post(f"{API_URL}/predict", files={"file": (uploaded_anom.name, buf2.getvalue(), "image/png")}, timeout=5)
        if resp2.ok:
            out = resp2.json()
            st.session_state.update({
                'mse': out['mse'], 'mae': out['mae'], 'rmse': out['rmse'],
                'is_outlier': out['is_outlier'], 'face_image': None, 'anom_done': True
            })
        else:
            st.error(f"Fout bij anomaliedetectie: {resp2.status_code}")

# Prestatiemetrics tonen na voorspelling
if st.session_state['anom_done']:
    st.write(f"**MSE:** {st.session_state['mse']:.6f}")
    st.write(f"**MAE:** {st.session_state['mae']:.6f}")
    st.write(f"**RMSE:** {st.session_state['rmse']:.6f}")
    if st.session_state['is_outlier']:
        st.error("Anomalie gedetecteerd.")
        use_face = st.checkbox("Gezichtsdetectie inschakelen voor blurring (optioneel)", key="face_detect_checkbox")
        if use_face:
            if st.session_state['face_image'] is None:
                yolo = YOLO("yolov8n-face.pt")
                arr = np.array(st.session_state['full_image'])
                res = yolo(arr)
                if res and len(res[0].boxes):
                    x1,y1,x2,y2 = res[0].boxes.xyxy[0].cpu().numpy().astype(int)
                    st.session_state['face_image'] = st.session_state['full_image'].crop((x1,y1,x2,y2))
                else:
                    st.warning("Geen gezicht gevonden — gebruik volledige afbeelding.")
                    st.session_state['face_image'] = st.session_state['full_image']
            st.image(st.session_state['face_image'], caption="Gecropt Gezicht", use_container_width=True)
            if st.button("Blur Anomalie Gezicht", key="blur_face_btn"):
                blurred = st.session_state['face_image'].filter(ImageFilter.GaussianBlur(15))
                st.image(blurred, caption="Blurred Anomalie Gezicht", use_container_width=True)
    else:
        # Indien geen outlier, alleen weergeven want geen blurring nodig
        st.success("Geen anomalie gedetecteerd — afbeelding is normaal.")