from django.shortcuts import render, redirect
from .forms import *
from django.contrib.auth import login, authenticate
from django.contrib import messages
from django.contrib.auth.forms import AuthenticationForm
from django.http.response import StreamingHttpResponse
from .camera import LiveWebCam
import cv2

# Create your views here.
def register_request(request):
	if request.method == "POST":
		form = NewUserForm(request.POST)
		if form.is_valid():
			user = form.save()
			login(request, user)
			messages.success(request, "Registration successful." )
			return redirect("home")
		messages.error(request, "Unsuccessful registration. Invalid information.")
	form = NewUserForm()
	return render (request=request, template_name="register.html", context={"register_form":form})

def login_request(request):
	if request.method == "POST":
		form = AuthenticationForm(request, data=request.POST)
		if form.is_valid():
			username = form.cleaned_data.get('username')
			password = form.cleaned_data.get('password')
			user = authenticate(username=username, password=password)
			if user is not None:
				login(request, user)
				messages.info(request, f"You are now logged in as {username}.")
				return redirect("home")
			else:
				messages.error(request,"Invalid username or password.")
		else:
			messages.error(request,"Invalid username or password.")
	form = AuthenticationForm()
	return render(request=request, template_name="login.html", context={"login_form":form})

def gen(camera=None):
    cap = cv2.VideoCapture("rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4")
    
    while True:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        frame = cv2.resize(frame, (640, 480), interpolation = cv2.INTER_LINEAR) 
        frame = frame.tobytes()
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        #frame = camera.get_frame()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        
def carpark_map(request):
    return render(request=request, template_name="home.html", context={})

def cam_feed(request):
    
	return StreamingHttpResponse(gen(),
					content_type='multipart/x-mixed-replace; boundary=frame')
    #return render(request=request, template_name="home.html", context={})


# Predict functions

def cnn_predict(img, bb_coor):
	
	# return a dict 
	# {
	# 	"B120" : 0,
	# 	"B121" : 1,
	# }
	pass

def yolo_predict(img, bb_coor):
	pass