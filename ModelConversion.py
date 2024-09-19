from ultralytics import YOLO
model = YOLO(r"/home/gregory/RailCV/RailModelBest.pt")
model.export(format="onnx")