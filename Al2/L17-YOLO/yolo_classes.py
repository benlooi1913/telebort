from ultralytics import YOLO

# Load the trained model
model = YOLO('yolov8n.pt')

# Get class names
class_names = model.names

# Print class names
print(class_names)

# Save class names to file
with open('class_names.txt', 'w') as f:
    for index in class_names:
        f.write(f'{index}: {class_names[index]}\n')
