from ultralytics import YOLO, solutions 
model = YOLO(r"\\wsl.localhost\Ubuntu\home\gmoore2\RailCV\RailModelBest.pt")

results = model.predict(r"\\wsl.localhost\Ubuntu\home\gmoore2\RailCV\train2.mp4", show = True, save=True)
#results.save("annotated.mov")
#Video Source
# cap = cv2.VideoCapture("wichita.MOV")
# assert 