import cv2
import os

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
    cfp=os.path.abspath(os.path.dirname(__file__))
    f = open(os.getcwd() + '/stream.txt', "r")
    streamPath = f.readline().rstrip()
    f.close()

    currentIndex = str(get_index(cfp + '/screenshots')+1)
    rawImgIndex = str(get_index(os.getcwd() + '/drawRect/DrawRectangle/raw_imgs'))
    savePath = cfp + '/screenshots/output' + currentIndex + '.jpg'
    # saveRawPath = os.getcwd() + '/drawRect/DrawRectangle/raw_imgs/spot' + rawImgIndex + '.jpg'

    cap= cv2.VideoCapture(streamPath)
    while True:
        ret, frame = cap.read()
        if ret == False:
            print("Error: failed to capture frame from RTSP stream")
            break

        # cv2.imshow("frame", frame)

        # Press 'c' key to capture screenshot
        # if cv2.waitKey(1) & 0xFF == ord('c'):
        cv2.imwrite(savePath, frame)
        # cv2.imwrite(saveRawPath, frame)
        print("Screenshot captured")
        break

    cap.release()
    cv2.destroyAllWindows()