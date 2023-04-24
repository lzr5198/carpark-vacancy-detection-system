from django.shortcuts import render,redirect
from django.contrib.auth import authenticate,login,logout
from django.contrib.auth.models import User
from django.contrib import messages
import os
from django.urls import reverse
from django.http import HttpResponse
from django.conf import settings
import cv2
import subprocess

cfp=os.path.abspath(os.path.dirname(__file__))

def record(request):
    print(os.getcwd())
    subprocess.run(["python3", os.getcwd() + "/capture/record.py"])
    return redirect(reverse("process_data"))

def screenshot(request):
    subprocess.run(["python3", os.getcwd() + "/capture/screenshot.py"])
    return redirect(reverse("process_data"))
