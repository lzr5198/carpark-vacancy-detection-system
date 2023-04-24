import numpy as np
import cv2


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