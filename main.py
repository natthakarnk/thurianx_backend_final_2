from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import numpy as np
import cv2
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # หรือใส่ domain frontend ของคุณ เช่น "https://your-frontend.vercel.app"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app = FastAPI()
model = YOLO("model.pt")

@app.get("/")
def read_root():
    return {"message": "ThurianX API is running on Render!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    results = model(img)
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = model.names[int(box.cls[0])]
        # ⭐ เพิ่ม confidence ลงใน response ⭐
        confidence = float(box.conf[0]) if hasattr(box, "conf") else None
        detections.append({
            "label": label,
            "confidence": confidence,  # <--- เพิ่มตรงนี้
            "x": x1,
            "y": y1,
            "width": x2 - x1,
            "height": y2 - y1
        })

    return JSONResponse(content={"results": detections})
