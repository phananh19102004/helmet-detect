from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from ultralytics import YOLO
from PIL import Image, ImageDraw
import io
import os

app = FastAPI()

MODEL_PATH = os.getenv("MODEL_PATH", "/models/best.pt")
model = YOLO(MODEL_PATH)


# ----------website---------------
@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
    <head>
        <title>YOLO Helmet Detection Demo</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; padding-top: 50px; }
            h2 { color: #4CAF50; }
            .container { width: 50%; margin: auto; padding: 20px; }
            input[type="file"] { padding: 10px; margin: 10px; }
            button { padding: 10px 20px; background-color: #4CAF50; color: white; border: none; cursor: pointer; }
            button:hover { background-color: #45a049; }
            img { margin-top: 20px; max-width: 100%; height: auto; border: 3px solid #4CAF50; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>YOLO Helmet Detection</h2>
            <form action="/predict-image" enctype="multipart/form-data" method="post" id="upload-form">
                <input type="file" name="file" accept="image/*" id="file-input" required>
                <br><br>
                <button type="submit">Detect</button>
            </form>
            <p>Upload an image to see detection result</p>
            <div id="image-result"></div>
        </div>

        <script>
            document.getElementById('file-input').addEventListener('change', function(event) {
                const reader = new FileReader();
                reader.onload = function() {
                    const img = document.createElement('img');
                    img.src = reader.result;
                    img.id = 'image-preview';
                    const resultDiv = document.getElementById('image-result');
                    resultDiv.innerHTML = '';  // Clear previous preview
                    resultDiv.appendChild(img);  // Show the image preview
                };
                reader.readAsDataURL(event.target.files[0]);
            });

            // Optional: Prevent the form from submitting to show the image preview
            document.getElementById('upload-form').addEventListener('submit', function(event) {
                event.preventDefault();
                const formData = new FormData(event.target);
                fetch('/predict-image', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.blob())
                .then(imageBlob => {
                    const imageUrl = URL.createObjectURL(imageBlob);
                    const img = document.createElement('img');
                    img.src = imageUrl;
                    document.getElementById('image-result').innerHTML = '';
                    document.getElementById('image-result').appendChild(img);
                });
            });
        </script>
    </body>
    </html>
    """


# Predict & Draw Bounding Box
# -------------------------
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    results = model(image)
    boxes = results[0].boxes

    draw = ImageDraw.Draw(image)
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf)
        cls = int(box.cls)
        label = f"helmet {conf:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 10), label, fill="red")

    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/jpeg")


# API Health
# -------------------------
@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_PATH}
