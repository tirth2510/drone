from ultralytics import YOLO

# Specify the path to the model weights
model_path = r'C:\Users\tejal\OneDrive\Desktop\Yolo\runs\detect\train7\weights\last.pt'

# Load the model
model = YOLO(model_path)

# Display the model architecture
print(model.model)

# Extract the input size from the model's configuration
input_size = model.model[0].yaml['height'], model.model[0].yaml['width']
print(f"Expected input size: {input_size}")

