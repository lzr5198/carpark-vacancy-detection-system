from django.shortcuts import render
from django.http import HttpResponse
from .models import Carslot
from django.shortcuts import redirect
import os
from django.template.loader import render_to_string
import json

# Create your views here.
result = None

def process_data(request):
    global result

    if request.method == "POST":
        data = {
            "object_name": [],
            "x1": [],
            "y1": [],
            "x2": [],
            "y2": []
        }
        for key in request.POST.keys():
            data[key] = request.POST.getlist(key)
        # print(data)
        result = helper_function(data)
        return HttpResponse()
    
    if request.method == "GET":
        print(result)
        print("INGet")
        json_result = json.dumps(result)
        # html = render_to_string('carslots.html', {"json_result": json_result})
        return render(request, "carslots.html", {"json_result": json_result})

def ajax_function(request):
    if request.method == "GET":
        json_result = json.dumps(result)
        return HttpResponse(json_result)

def helper_function(data):
    result = {
        "A1": False,
        "A2": False,
        "A3": False,
        "A4": False,
    }
    for i in range(len(data["object_name"])):
        # if data["object_name"][i] != "person":
        #     continue
        for carslot in Carslot.objects.all():
            if int(data["x1"][i]) > carslot.x1 and int(data["y1"][i]) > carslot.y1 and int(data["x2"][i]) < carslot.x2 and int(data["y2"][i]) < carslot.y2:
                result[carslot.slotId] = True
    
    # print(result)
    # print("\n")
    return result