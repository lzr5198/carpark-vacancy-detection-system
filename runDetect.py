import cv2
import time
from datetime import datetime
import os


imagesFolder = "/Users/thomasyin/Desktop"
cap = cv2.VideoCapture("rtsp://192.168.0.158:554/stream1")

frameRate = cap.get(5) #frame rate

cur_time = time.time()  # Get current time

# start_time_24h measures 24 hours
start_time_24h = cur_time

# start_time_1min measures 1 minute
start_time_1min = cur_time - 59

while cap.isOpened():
    frameId = cap.get(1)  # current frame number
    ret, frame = cap.read()

    if (ret != True):
        break

    cur_time = time.time()  # Get current time
    elapsed_time_1min = cur_time - start_time_1min  # Time elapsed from previous image saving.

    if elapsed_time_1min >= 1:
        # Reset the timer that is used for measuring
        start_time_1min = cur_time

        filename = imagesFolder + "/image.jpg"
        # filename = "stream.txt"
        cv2.imwrite(filename, frame)

        os.system("python3 ./yolov5/detect.py --exist-ok --source " + filename)

        cv2.waitKey(1)

    elapsed_time_24h = time.time() - start_time_24h

    if elapsed_time_24h > 24*60*60:
        break

cap.release()
print ("Done!")

cv2.destroyAllWindows()