import copy
import cv2
import numpy as np
import os
import tkinter as tk
import sys
import requests
from multiprocessing import Process, Queue


WIN_NAME = 'draw_rect'
cfp=os.path.abspath(os.path.dirname(__file__))

carslot_ids = []

def get_filenames(directory):
    filenames = []
    for file in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, file)):
            if file.split('.')[1] == 'jpg':
                filenames.append(file)
    return filenames

def get_index():
    tmp = get_filenames(cfp + '/raw_imgs')
    if tmp:
        return len(tmp)
    else:
        return 0

def get_carslot_id(q, x):
    root = tk.Tk()

    root.title("Carslot ID")
    root.geometry("300x200")

    input_label = tk.Label(root, text="Enter the carslot ID:")
    input_entry = tk.Entry(root)
    input_entry.focus_set()
    def get_input(event=None):
        user_input = input_entry.get()
        q.put(user_input)
        root.destroy()

    submit_button = tk.Button(root, text="Submit", command=get_input)

    input_label.pack()
    input_entry.pack()
    submit_button.pack()

    root.bind("<Return>", get_input)

    root.lift()
    root.mainloop()

class Rect(object):
    def __init__(self):
        self.tl = (0, 0)
        self.br = (0, 0)

    def regularize(self):
        """
        make sure tl = TopLeft point, br = BottomRight point
        """
        pt1 = (min(self.tl[0], self.br[0]), min(self.tl[1], self.br[1]))
        pt2 = (max(self.tl[0], self.br[0]), max(self.tl[1], self.br[1]))
        self.tl = pt1
        self.br = pt2

class DrawRects(object):
    def __init__(self, img_loc, color, thickness=1):
        image=cv2.imread(img_loc)
        self.img_loc=os.path.basename(img_loc)
        self.img_name=self.img_loc.split('.')[0]
        self.original_image = image
        self.image_for_show = image.copy()
        self.color = color
        self.thickness = thickness
        self.rects = []
        self.current_rect = Rect()
        self.left_button_down = False

    @staticmethod
    def __clip(value, low, high):
        """
        clip value between low and high

        Parameters
        ----------
        value: a number
            value to be clipped
        low: a number
            low limit
        high: a number
            high limit

        Returns
        -------
        output: a number
            clipped value
        """
        output = max(value, low)
        output = min(output, high)
        return output

    def shrink_point(self, x, y):
        """
        shrink point (x, y) to inside image_for_show

        Parameters
        ----------
        x, y: int, int
            coordinate of a point

        Returns
        -------
        x_shrink, y_shrink: int, int
            shrinked coordinate
        """
        height, width = self.image_for_show.shape[0:2]
        x_shrink = self.__clip(x, 0, width)
        y_shrink = self.__clip(y, 0, height)
        return (x_shrink, y_shrink)

    def append(self):
        """
        add a rect to rects list
        """
        self.rects.append(copy.deepcopy(self.current_rect))

    def pop(self):
        """
        pop a rect from rects list
        """
        rect = Rect()
        if self.rects:
            rect = self.rects.pop()
        return rect

    def reset_image(self):
        """
        reset image_for_show using original image
        """
        self.image_for_show = self.original_image.copy()

    def draw(self):
        """
        draw rects on image_for_show
        """
        for rect in self.rects:
            cv2.rectangle(self.image_for_show, rect.tl, rect.br,
                          color=self.color, thickness=self.thickness)

    def draw_current_rect(self):
        """
        draw current rect on image_for_show
        """
        cv2.rectangle(self.image_for_show,
                      self.current_rect.tl, self.current_rect.br,
                      color=self.color, thickness=self.thickness)


def onmouse_draw_rect(event, x, y, flags, draw_rects):
    global carslot_ids
    # create the file anyway
    # f=open(cfp+'/box_coordinates/'+draw_rects.img_name+'.txt','w')
    # f.close()

    if event == cv2.EVENT_LBUTTONDOWN:
        # pick first point of rect
        print('pt1: x = %d, y = %d' % (x, y))
        draw_rects.left_button_down = True
        draw_rects.current_rect.tl = (x, y)
    if draw_rects.left_button_down and event == cv2.EVENT_MOUSEMOVE:
        # pick second point of rect and draw current rect
        draw_rects.current_rect.br = draw_rects.shrink_point(x, y)
        draw_rects.reset_image()
        draw_rects.draw()
        draw_rects.draw_current_rect()
    if event == cv2.EVENT_LBUTTONUP:
        # finish drawing current rect and append it to rects list
        draw_rects.left_button_down = False
        draw_rects.current_rect.br = draw_rects.shrink_point(x, y)
        print('pt2: x = %d, y = %d' % (draw_rects.current_rect.br[0],
                                       draw_rects.current_rect.br[1]))
        draw_rects.current_rect.regularize()
        draw_rects.append()

        # pop dialog
        carslot_id = Queue()
        p = Process(target=get_carslot_id, args=(carslot_id, 1))
        p.start()
        p.join() # this blocks until the process terminates
        carslot_id = carslot_id.get()
        carslot_ids.append(carslot_id)
        
        f=open(cfp+'/box_coordinates/'+draw_rects.img_name+'.txt','w')

        for i in range(len(draw_rects.rects)):
            f.write(carslot_ids[i] + '\n')
            f.write('%d,%d,%d,%d\n'%(draw_rects.rects[i].tl[0],draw_rects.rects[i].tl[1],draw_rects.rects[i].br[0],draw_rects.rects[i].br[1]))
        f.close()
    if (not draw_rects.left_button_down) and event == cv2.EVENT_RBUTTONDOWN:
        # pop the last rect in rects list
        draw_rects.pop()
        draw_rects.reset_image()
        draw_rects.draw()

        carslot_ids.pop()

        f=open(cfp+'/box_coordinates/'+draw_rects.img_name+'.txt','w')

        for i in range(len(draw_rects.rects)):
            f.write(carslot_ids[i] + '\n')
            f.write('%d,%d,%d,%d\n'%(draw_rects.rects[i].tl[0],draw_rects.rects[i].tl[1],draw_rects.rects[i].br[0],draw_rects.rects[i].br[1]))
        f.close()

def compute_remap(image):
        R = image.shape[0]//2
        W = int(2*np.pi*R)
        H = R
        mapx = np.zeros([H,W], dtype=np.float32)
        mapy = np.zeros([H,W], dtype=np.float32)
        for i in range(mapx.shape[0]):
            for j in range(mapx.shape[1]):
                angle = j/W*np.pi*2
                radius = H-i
                mapx[i,j]=R+np.sin(angle)*radius
                mapy[i,j]=R-np.cos(angle)*radius
        image_remap = cv2.remap(image, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) 
        # self.show_image(image_remap)
        return image_remap

def compute_remap(image):
    R = image.shape[0]//2
    W = int(2*np.pi*R)
    H = R
    mapx = np.zeros([H,W], dtype=np.float32)
    mapy = np.zeros([H,W], dtype=np.float32)

    for i in range(mapx.shape[0]):
        for j in range(mapx.shape[1]):
            angle = j/W*np.pi*2
            radius = H-i
            mapx[i,j]=R+np.sin(angle)*radius
            mapy[i,j]=R-np.cos(angle)*radius
        
    image_remap = cv2.remap(image, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) 
    return image_remap

if __name__ == '__main__':
    ############################################ using existing image ############################################
    # currentIndex = str(get_index())
    # # img_loc = cfp + '/raw_imgs/spot' + currentIndex + '.jpg'
    # img_loc = cfp+'/raw_imgs/spot0.jpg'
    ############################################ using existing image ############################################

    ############################################ using rtsp stream ############################################
    endpoint = "http://localhost:8000/drawRect/undistort/"
    data = requests.get(url = endpoint).json()
    
    f = open(os.getcwd() + '/stream.txt', "r")
    streamPath = f.readline().rstrip()
    f.close()
    currentIndex = str(get_index())
    img_loc = cfp + '/raw_imgs/spot' + currentIndex + '.jpg'

    cap= cv2.VideoCapture(streamPath)
    ret, frame = cap.read()
    if ret != False:
        if data["undistort"]:
            frame = compute_remap(frame)
        cv2.imwrite(img_loc, frame)
    else:
        print("Error: failed to capture frame from RTSP stream")

    cap.release()
    ############################################ using rtsp stream ############################################

    draw_rects = DrawRects(img_loc, (0, 255, 0), 3)
    cv2.namedWindow(WIN_NAME, 0)
    cv2.setMouseCallback(WIN_NAME, onmouse_draw_rect, draw_rects)
    while True:
        cv2.imshow(WIN_NAME,draw_rects.image_for_show)
        key = cv2.waitKey(30)
        
        if key == 27:  # ESC
            cv2.imwrite(cfp+'/processed_imgs/'+draw_rects.img_name+'.jpg', draw_rects.image_for_show)
            break
    cv2.destroyAllWindows()
