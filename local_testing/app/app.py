from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

model = YOLO("model/best.pt")   

@app.get("/")
def root():
    return {"status": "Helmet Detection API running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    results = model(img)
    boxes = results[0].boxes

    detections = []
    for box in boxes:
        detections.append({
            "cls": int(box.cls),
            "conf": float(box.conf),
            "xyxy": box.xyxy.tolist()
        })

    return {"detections": detections}

