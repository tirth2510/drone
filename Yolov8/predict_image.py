import os
from ultralytics import YOLO
import cv2

# Provide the full path to the image file
image_path = r'C:\Users\tejal\OneDrive\Desktop\Yolo\data\images\test\White-Wagtail_999.jpg'
output_image_path = 'output_example.jpg'

# Load the image
image = cv2.imread(image_path)

# Check if the image is loaded successfully
if image is None:
    print(f"Error: Couldn't open image file at {image_path}")
    exit()

H, W, _ = image.shape

image = cv2.resize(image, (640, 448))

# Load the YOLO model
model_path = r"C:\Users\tejal\OneDrive\Desktop\Yolo\runs\detect\train5\weights\last.pt"
model = YOLO(model_path)

# Set the detection threshold
threshold = 0.5

# Perform object detection
results = model(image)[0]

# Draw bounding boxes on the image
for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score > threshold:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        cv2.putText(image, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

# Save the output image
cv2.imwrite(output_image_path, image)

# Display the image
cv2.imshow('Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
