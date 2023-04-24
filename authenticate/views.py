from django.shortcuts import render,redirect
from django.urls import reverse
from django.contrib.auth import authenticate,login,logout
from django.contrib.auth.models import User
from django.contrib import messages
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
    return render(request,'main.html')

def logoutUser(request):
    logout(request)
    return redirect(reverse('login'))

def home(request):
    return render(request,'home.html')