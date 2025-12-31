from ultralytics import YOLO
import cv2
import numpy as np

# Step 1: Convert YOLO to ONNX
model = YOLO("yolo11n.pt")
model.export(format="onnx")  # Creates yolo11n.onnx

print("✅ YOLO model converted to ONNX format!")

# Step 2: Test the ONNX model
onnx_model = YOLO("yolo11n.onnx")

# Test with an image
results = onnx_model("image.png")
results[0].show()
results[0].save("output_onnx.jpg")

print("✅ ONNX model tested successfully!")