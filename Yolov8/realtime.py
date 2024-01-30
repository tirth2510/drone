from ultralytics import YOLO
import cv2

custom_model_path = "runs/detect/train5/weights/last.pt"
model = YOLO("yolov8s")

source = 0

# Keep track of previously detected labels
previous_detections = set()

while True:
    results = model.predict(source=source, show=True)

    # Check if "bottle" is detected
    bottle_detected = False
    for pred in results.pred[0]:
        label = model.names[pred[5]]
        confidence = pred[4].item()

        if label == "bottle" and confidence > 0.5:
            print("Bottle detected.")
            bottle_detected = True
            break  # No need to continue checking if bottle is already detected

    # If you want to save the image with bounding boxes, you can use the following line
    # results.save(save_dir="output/")

    if bottle_detected:
        print("Terminating the code.")
        break

    # If you want to save the image with bounding boxes, you can use the following line
    # results.save(save_dir="output/")

    # Wait for any key press and check if it's not -1 (indicating a key press)
    key = cv2.waitKey(1) & 0xFF  # Use bitwise AND to extract the ASCII value of the key

    if key != 255:  # Check if a key is pressed (not equal to default -1)
        print(f"Key pressed: {chr(key)}")
        print("Quitting the camera window.")
        break

# Release the camera and close the OpenCV window
cv2.destroyAllWindows()



