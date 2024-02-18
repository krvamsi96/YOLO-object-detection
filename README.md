### YOLO-object-detection

from ultralytics import YOLO

### Load a COCO-pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

### Display model information (optional)
model.info()

### Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

### Run inference with the YOLOv8n model on the 'bus.jpg' image
results = model('bus.jpg')


Certainly! Here's the explanation of the Python code provided:

```python
from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

# Run inference with the YOLOv8n model on the 'bus.jpg' image
results = model('path/to/bus.jpg')
```

### Explanation:
1. We import the `YOLO` class from the `ultralytics` library, which provides functionalities for YOLO object detection.
2. We load a COCO-pretrained YOLOv8n model by instantiating the `YOLO` class with the path to the pretrained model file (`yolov8n.pt`).
3. Optionally, we can display information about the loaded model using the `info()` method, which provides details about the architecture, number of parameters, and other relevant information.
4. We can further train the loaded model on a custom dataset using the `train` method. Here, we specify the dataset configuration file (`coco8.yaml`), the number of training epochs (100), and the input image size (`imgsz=640`).
5. Finally, we perform object detection inference on an image (`bus.jpg`) using the trained model. We pass the path to the image as an argument to the model instance, which returns the detected objects along with their bounding boxes and confidence scores.
