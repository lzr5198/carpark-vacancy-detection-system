import cv2
import time
from datetime import datetime
import os


imagesFolder = "/Users/kevinlin/Desktop"
#cap = cv2.VideoCapture("rtsp://username:password@cameraIP/axis-media/media.amp")

# Use public RTSP Streaming for testing:
# cap = cv2.VideoCapture("rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov")
cap = cv2.VideoCapture("rtsp://192.168.3.11:554/stream1")

#cap = cv2.VideoCapture("test2.mp4")
frameRate = cap.get(5) #frame rate

cur_time = time.time()  # Get current time

# start_time_24h measures 24 hours
start_time_24h = cur_time

# start_time_1min measures 1 minute
start_time_1min = cur_time - 59  # Subtract 59 seconds for start grabbing first frame after one second (instead of waiting a minute for the first frame).

while cap.isOpened():
    frameId = cap.get(1)  # current frame number
    ret, frame = cap.read()

    if (ret != True):
        break

    cur_time = time.time()  # Get current time
    elapsed_time_1min = cur_time - start_time_1min  # Time elapsed from previous image saving.

    # If 60 seconds were passed, reset timer, and store image.
    if elapsed_time_1min >= 3:
        # Reset the timer that is used for measuring 60 seconds
        start_time_1min = cur_time

        filename = imagesFolder + "/image.jpg"
        # _" + str(datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p"))  + "
        #filename = "image_" + str(datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p"))  + ".jpg"
        cv2.imwrite(filename, frame)

        # os.system("python3 ./yolov5/detect.py --source " + filename)
        os.system("python3 ./yolov5/detect.py --exist-ok --source " + filename)
        # os.system("python3 ./yolov5/detect.py --source 'rtsp://192.168.3.11:554/stream1'")


        # Show frame for testing
        # cv2.imshow('frame', frame)
        cv2.waitKey(1)

    elapsed_time_24h = time.time() - start_time_24h

    #Break loop after 24*60*60 seconds
    if elapsed_time_24h > 24*60*60:
        break

    #time.sleep(60 - elapsed_time) # Sleeping is a bad idea - we need to grab all the frames.


cap.release()
print ("Done!")

cv2.destroyAllWindows()