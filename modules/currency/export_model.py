# run this once as a standalone script — save it as export_model.py
# place it next to your best.onnx (which is actually best.pt)
from ultralytics import YOLO

# load it as pt (even though it's named .onnx, it's actually pytorch)
model = YOLO("best.pt")

# export as real ONNX
model.export(format="onnx", imgsz=640, opset=12)