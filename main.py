import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import List
import json
from pathlib import Path
import shutil
import tempfile
import cv2
import numpy as np
import re
import easyocr
from ultralytics import YOLO

app = FastAPI()

DATA_FILE = Path("kenteken_mapping.json")

class Mapping(BaseModel):
    kenteken: str
    locatiecode: str
    omschrijving: str

#--------------------------------------------------------------------------------------
# Tussendoor opslaan van ingevoerde kentekens, plek, omschrijving

# === Helperfuncties voor opslag ===
def load_data():
    if DATA_FILE.exists():
        with open(DATA_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

# === Start met geladen data ===
kenteken_mapping = load_data()

#-----------------------------------------------------------------------------
# Communicatie met streamlit over registratie van kentekens en waar ze horen

# === API endpoints ===

@app.get("/mappings", response_model=List[Mapping])
def get_mappings():
    return kenteken_mapping

@app.post("/mappings")
def add_mapping(item: Mapping):
    kenteken_mapping.append(item.dict())
    save_data(kenteken_mapping)
    return {"status": "toegevoegd", "item": item}

@app.delete("/mappings")
def clear_all():
    kenteken_mapping.clear()
    save_data(kenteken_mapping)
    return {"status": "alle mappings verwijderd"}

@app.delete("/mappings/{kenteken}")
def delete_mapping(kenteken: str):
    global kenteken_mapping
    initial_len = len(kenteken_mapping)
    kenteken_mapping = [m for m in kenteken_mapping if m["kenteken"].upper() != kenteken.upper()]
    if len(kenteken_mapping) == initial_len:
        raise HTTPException(status_code=404, detail="Kenteken niet gevonden")
    save_data(kenteken_mapping)
    return {"status": f"{kenteken} verwijderd"}

#----------------------------------------------------------------------
# Verwerken van afbeelding met kenteken herkenning

# === OCR Setup ===
# model = YOLO("best.pt")  # Pad naar jouw getrainde model
# reader = easyocr.Reader(['en', 'nl'])

# def clean_kenteken(text):
#     return re.sub(r'[^A-Z0-9]', '', text.upper())

# def herken_kenteken_from_image(image_path):
#     results = model.predict(source=image_path, conf=0.5)
#     image = cv2.imread(image_path)
#     boxes = results[0].boxes.xyxy.cpu().numpy()

#     pad = 50
#     ocr_teksten = []

#     for box in boxes:
#         x1, y1, x2, y2 = map(int, box)
#         x1 = max(x1 - pad, 0)
#         y1 = max(y1 - pad, 0)
#         x2 = min(x2 + pad, image.shape[1])
#         y2 = min(y2 + pad, image.shape[0])
#         cropped = image[y1:y2, x1:x2]

#         gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
#         _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#         result_ocr = reader.readtext(thresh, contrast_ths=0.05, adjust_contrast=0.7)

#         for t in result_ocr:
#             tekst = clean_kenteken(t[1])
#             if tekst:
#                 ocr_teksten.append(tekst)

#     return ocr_teksten