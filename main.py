from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import numpy as np
import cv2
import joblib
from color_utils import calculate_average_deltaE, calculate_avg_hsb

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("models/color_quality_model.pkl")
@app.get("/")
def read_root():
    return {"message": "Color API is running 🚀"}
@app.post("/analyze")
async def analyze_color(
    reference_r: int = Form(...),
    reference_g: int = Form(...),
    reference_b: int = Form(...),
    files: List[UploadFile] = File(...)
):
    reference_rgb = (reference_r, reference_g, reference_b)

    delta_e_list = []
    hue_diff_list = []
    sat_diff_list = []
    bright_diff_list = []

    for file in files:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (300, 300))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        avg_delta_e = calculate_average_deltaE(reference_rgb, img_rgb)
        delta_e_list.append(avg_delta_e)

        h2, s2, b2 = calculate_avg_hsb(img)
        ref_bgr = np.uint8([[list(reference_rgb)]])
        ref_hsv = cv2.cvtColor(ref_bgr, cv2.COLOR_RGB2HSV)[0][0]
        h1, s1, b1 = ref_hsv

        hue_diff_list.append(abs(h1 - h2))
        sat_diff_list.append(abs(s1 - s2))
        bright_diff_list.append(abs(b1 - b2))

    # Average features across all images
    avg_delta_e_final = np.mean(delta_e_list)
    avg_hue_diff = np.mean(hue_diff_list)
    avg_sat_diff = np.mean(sat_diff_list)
    avg_bright_diff = np.mean(bright_diff_list)

    features = [avg_delta_e_final, avg_hue_diff, avg_sat_diff, avg_bright_diff]
    prediction = model.predict([features])[0]
    accuracy = max(0, 100 - (avg_delta_e_final / 20) * 100)

    return {
        "delta_e": avg_delta_e_final,
        "accuracy": accuracy,
        "features": {
            "hue_diff": avg_hue_diff,
            "sat_diff": avg_sat_diff,
            "bright_diff": avg_bright_diff
        },
        "prediction": prediction
    }
