import cv2
import os
import requests
from undistort import compute_remap

def get_filenames(directory):
    filenames = []
    for file in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, file)):
            if file.split('.')[1] == 'avi':
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
    f = open(os.getcwd()+'/stream.txt', "r")
    streamPath = f.readline().rstrip()
    f.close()

    currentIndex = str(get_index(cfp + '/recordings')+1)
    savePath = cfp + '/recordings/output' + currentIndex + '.avi'

    cap = cv2.VideoCapture(streamPath)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(savePath, fourcc, 20.0, size)

    while(True):
        _, frame = cap.read()
        # if data["undistort"]:
        #     frame = compute_remap(frame)
        cv2.imshow('Recording...', frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()