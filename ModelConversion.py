from ultralytics import YOLO
model = YOLO(r"/home/gregory/Documents/RailCV/MultiClassRailModelV8.pt")
model.export(format="onnx")
