from ultralytics import YOLO
import cv2

# Load the model
model = YOLO("yolo11n.pt")

# Perform object detection on an image
results = model("image.png")

# Show results with bounding boxes (this will display the image)
results[0].show()  # This will open a window with the image and bounding boxes

# OR you can save the result to a file
results[0].save("output.jpg") 