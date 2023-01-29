from django.shortcuts import render,redirect
from django.contrib.auth import authenticate,login,logout
from django.contrib.auth.models import User
from django.contrib import messages
from .DrawRectangle.DrawBoundingBox import *
import os
cfp=os.path.abspath(os.path.dirname(__file__))
# Create your views here.
def loginPage(request):
    if request.method=='POST':
        username=request.POST.get('username')
        password=request.POST.get('password')
        try:
            user=User.objects.get(username=username)
        except:
            messages.error(request,'User does not exist')
        user=authenticate(request,username=username,password=password)
        if user is not None:
            login(request,user)
            return redirect('home')
        else:
            messages.error(request,'Username OR password does not exist')
    return render(request,'base/login_register.html')

def logoutUser(request):
    logout(request)
    return redirect('home')


def home(request):
    return render(request,'base/home.html')

def draw_rectangles(request):
    #从base_imgs文件夹遍历base 图片，把文件名传入render
    #context={"bases":bases}
    bases=[]
    for root, dirs, files in os.walk(cfp+"/DrawRectangle/base_imgs"):
        for file in files:
            bases.append(file)
    context={"bases":bases}
    return render(request,'base/draw_rectangles.html',context)

def draw(request,base):

    img_loc = cfp+'/DrawRectangle/base_imgs/'+base
    draw_rects = DrawRects(img_loc, (0, 255, 0), 1)
    cv2.namedWindow(WIN_NAME, 0)
    cv2.setMouseCallback(WIN_NAME, onmouse_draw_rect, draw_rects)
    while True:
        cv2.imshow(WIN_NAME,draw_rects.image_for_show)
        key = cv2.waitKey(30)
        if key == 27:  # ESC
            cv2.imwrite(cfp+'/DrawRectangle/Img_Boxes/'+draw_rects.img_name+'_bboxes'+'.jpg', draw_rects.image_for_show)
            break
    cv2.destroyAllWindows()

    bases=[]
    for root, dirs, files in os.walk(cfp+"/DrawRectangle/base_imgs"):
        for file in files:
            bases.append(file)
    context={"bases":bases}
    return render(request,'base/draw_rectangles.html',context)

    