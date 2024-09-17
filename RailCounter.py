import cv2
from ultralytics import YOLO, solutions 

model = YOLO(r"/home/gregory/Documents/RailCV/RailModelBest.pt")

cap = cv2.VideoCapture(r"/home/gregory/Documents/RailCV/train-short.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

#Define points that the line is on
line_points = [(350,0), (350,1000)]

#Video Writer
video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

#Initialize object counter
counter = solutions.ObjectCounter(
    view_img = True,
    reg_pts = line_points,
    count_reg_color = (255, 255, 255),
    count_txt_color = (0,0,0),
    view_out_counts = False,
    names = model.names,
    draw_tracks = False,
    line_thickness = 2,
    region_thickness=2
)
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    tracks = model.track(im0, persist=True, show=True)

    im0 = counter.start_counting(im0, tracks)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()

# results = model.predict(r"\\wsl.localhost\Ubuntu\home\gmoore2\RailCV\train2.mp4", show = True, save=True)

