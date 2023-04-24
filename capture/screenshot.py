import cv2
import os
import requests
from undistort import compute_remap

def get_filenames(directory):
    filenames = []
    for file in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, file)):
            if file.split('.')[1] == 'jpg':
                filenames.append(file)
    return filenames

def get_index(directory):
    tmp = get_filenames(directory)
    if tmp:
        return len(tmp)
    else:
        return 0

if __name__ == '__main__':
    endpoint = "http://localhost:8000/drawRect/undistort/"
    data = requests.get(url = endpoint).json()

    cfp=os.path.abspath(os.path.dirname(__file__))
    f = open(os.getcwd() + '/stream.txt', "r")
    streamPath = f.readline().rstrip()
    f.close()

    currentIndex = str(get_index(cfp + '/screenshots')+1)
    savePath = cfp + '/screenshots/output' + currentIndex + '.jpg'
    # saveRawPath = os.getcwd() + '/drawRect/DrawRectangle/raw_imgs/spot' + rawImgIndex + '.jpg'

    cap= cv2.VideoCapture(streamPath)
    while True:
        ret, frame = cap.read()
        if ret == False:
            print("Error: failed to capture frame from RTSP stream")
            break

        if data["undistort"]:
            frame = compute_remap(frame)
        cv2.imwrite(savePath, frame)
        break

    cap.release()
    cv2.destroyAllWindows()