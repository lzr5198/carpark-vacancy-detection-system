import copy
import cv2
import numpy as np
import os
from tkinter import *
WIN_NAME = 'draw_rect'
cfp=os.path.abspath(os.path.dirname(__file__))

class Rect(object):
    def __init__(self):
        self.tl = (0, 0)
        self.br = (0, 0)
        self.base_index=-1
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
        #print(img_name)
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

def pop_up_box():
    """
    使用tkinter弹出输入框输入数字, 具有确定输入和清除功能, 可在函数内直接调用num(文本框的值)使用
    """

    import tkinter

    
    def inputint():
        nonlocal num
        try:
            num = int(var.get().strip())
        except:
            num = 'Not a valid integer.'


    def inputclear():
        nonlocal num
        var.set('')
        num = ''

    num = 0
    root = tkinter.Tk(className='Set Up Base')  # 弹出框框名
    root.geometry('270x60')     # 设置弹出框的大小 w x h

    var = tkinter.StringVar()   # 这即是输入框中的内容
    var.set('Iput Base Index') # 通过var.get()/var.set() 来 获取/设置var的值
    entry1 = tkinter.Entry(root, textvariable=var)  # 设置"文本变量"为var
    entry1.pack()   # 将entry"打上去"
    btn1 = tkinter.Button(root, text='Input', command=inputint)     # 按下此按钮(Input), 触发inputint函数
    btn2 = tkinter.Button(root, text='Clear', command=inputclear)   # 按下此按钮(Clear), 触发inputclear函数

    # 按钮定位
    btn2.pack(side='right')
    btn1.pack(side='right')

    # 上述完成之后, 开始真正弹出弹出框
    root.mainloop()
    return num

def onmouse_draw_rect(event, x, y, flags, draw_rects):
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
        base_index=pop_up_box()
        draw_rects.current_rect.base_index=base_index
        draw_rects.append()


        f=open(cfp+'/boundingBoxes/'+draw_rects.img_name+'.txt','w')
        for rect in draw_rects.rects:
            f.write('%d,%d,%d,%d,%d\n'%(rect.tl[0],rect.tl[1],rect.br[0],rect.br[1],rect.base_index))
        f.close()
    if (not draw_rects.left_button_down) and event == cv2.EVENT_RBUTTONDOWN:
        # pop the last rect in rects list
        draw_rects.pop()
        draw_rects.reset_image()
        draw_rects.draw()
        f=open(cfp+'/boundingBoxes/'+draw_rects.img_name+'.txt','w')
        for rect in draw_rects.rects:
            f.write('%d,%d,%d,%d,%d\n'%(rect.tl[0],rect.tl[1],rect.br[0],rect.br[1],rect.base_index))
        f.close()

def drawRectangleAPI(img_loc):
    draw_rects = DrawRects(img_loc, (0, 255, 0), 1)
    cv2.namedWindow(WIN_NAME, 0)
    cv2.setMouseCallback(WIN_NAME, onmouse_draw_rect, draw_rects)
    while True:
        cv2.imshow(WIN_NAME,draw_rects.image_for_show)
        key = cv2.waitKey(30)
        if key == 27:  # ESC
            cv2.imwrite(cfp+'/Img_Boxes/'+draw_rects.img_name+'_bboxes'+'.jpg', draw_rects.image_for_show)
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    img_loc = cfp+'/test0.jpg'
    draw_rects = DrawRects(img_loc, (0, 255, 0), 1)
    cv2.namedWindow(WIN_NAME, 0)
    cv2.setMouseCallback(WIN_NAME, onmouse_draw_rect, draw_rects)
    while True:
        cv2.imshow(WIN_NAME,draw_rects.image_for_show)
        key = cv2.waitKey(30)
        if key == 27:  # ESC
            cv2.imwrite(cfp+'/Img_Boxes/'+draw_rects.img_name+'_bboxes'+'.jpg', draw_rects.image_for_show)
            break
    cv2.destroyAllWindows()
