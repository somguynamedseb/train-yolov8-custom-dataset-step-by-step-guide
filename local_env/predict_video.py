import os

from ultralytics import YOLO
import cv2 as cv


# VIDEOS_DIR = os.path.join('.', 'videos')

video_path = ('walking1.mp4')

# video_path_out = '{}_out.mp4'.format('walking2.mp4')
# cv.imshow
cap = cv.VideoCapture(video_path)
if (cap.isOpened() == False):
    print("Error opening video stream or file")

# Read until video is completed
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

        # Display the resulting frame
        cv.imshow('Frame', frame)

        # Press Q on keyboard to  exit
        if cv.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
ret, frame = cap.read()
# H, W, _ = frame.shape
out = cv.VideoWriter(video_path_out, cv.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv.CAP_PROP_FPS)), (frame_height, frame_width))

model_path = os.path.join('.', 'runs', 'detect', 'train2', 'weights', 'last.pt')

cv.imshow("test",cap)
cv.waitKey()

# Load a model
model = YOLO(model_path)  # load a custom model
print("model loaded")
threshold = 0.5
print(ret)
print(cap.isOpened())
while ret:
    print(ret)
    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv.LINE_AA)

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv.destroyAllWindows()
