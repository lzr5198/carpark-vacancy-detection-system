from django.shortcuts import render,redirect
from django.contrib.auth import authenticate,login,logout
from django.contrib.auth.models import User
from django.contrib import messages
from .DrawRectangle.DrawBoundingBox import *
import os
import subprocess
from django.urls import reverse
from django.http import HttpResponse
from django.conf import settings
import json

cfp=os.path.abspath(os.path.dirname(__file__))

undistortOrNot = False

def spot_list(request):
    image_names = None
    context = None
    files = os.listdir(settings.STATICFILES_DIRS[0])
    image_names = sorted([f for f in files if f.endswith('.jpg')])
    if image_names:
        context = {'image_names': image_names}
    return render(request,'spot_list.html', context)

def draw(request):
    print(os.getcwd())
    subprocess.run(["python3", "drawRect/DrawRectangle/DrawBoundingBox.py"])
    return redirect(reverse('spot_list'))

def spot(request):
    image_name = str(request.GET.get('image_name'))

    context = {'image_name': image_name}
    return render(request, 'spot.html', context)

def undistort(request):
    global undistortOrNot

    if request.method == "GET":
        return HttpResponse(json.dumps({'undistort':undistortOrNot}))
    if request.method == "POST":
        undistortOrNot = not undistortOrNot
        return HttpResponse("Mode changed")