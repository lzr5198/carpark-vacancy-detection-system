from django.shortcuts import render,redirect
from django.contrib.auth import authenticate,login,logout
from django.contrib.auth.models import User
from django.contrib import messages
from .DrawRectangle.DrawBoundingBox import *
import os
import subprocess
from django.http import HttpResponse


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
            return redirect('carslots/')
        else:
            messages.error(request,'Username OR password does not exist')
    return render(request,'login_register.html')

def logoutUser(request):
    logout(request)
    return render(request,'login_register.html')



def home(request):
    return render(request,'home.html')

def draw_rectangles(request):
    return render(request,'draw_rectangles.html')

def draw(request):
    print(os.getcwd())
    subprocess.run(["python3", "base/DrawRectangle/DrawBoundingBox.py"])
    return render(request,'draw_rectangles.html')

    