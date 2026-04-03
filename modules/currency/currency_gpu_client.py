import base64
import requests
import cv2

GPU_URL = "https://9069663790960.notebooks.jarvislabs.net/proxy/8000/"   # ← change this

def detect_currency_gpu(frame):
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    img_base64 = base64.b64encode(buffer).decode()

    response = requests.post(GPU_URL, json={"image": img_base64}, timeout=10)
    return response.json()["result"]