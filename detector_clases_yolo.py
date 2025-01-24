from ultralytics import YOLO
model = YOLO('yolov8m.pt')
print(model.names)