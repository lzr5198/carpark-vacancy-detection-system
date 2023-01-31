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
        result = compare_bounding_box(data)
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

def compare_bounding_box(data):
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
            car_coor = [int(data['x1'][i]), int(data['y1'][i]), int(data['x2'][i]), int(data['y2'][i])]
            carslots_coor = [int(carslot.x1), int(carslot.y1), int(carslot.x2), int(carslot.y2)]
            # print(IOU(car_coor, carslots_coor))
            if IOU(car_coor, carslots_coor) > 0.7:
                result[carslot.slotId] = True


            # if int(data["x1"][i]) > carslot.x1 and int(data["y1"][i]) > carslot.y1 and int(data["x2"][i]) < carslot.x2 and int(data["y2"][i]) < carslot.y2:
            #     result[carslot.slotId] = True
    return result

def IOU(box1, box2):
	x1, y1, x2, y2 = box1
	x3, y3, x4, y4 = box2
	x_inter1 = max(x1, x3)
	y_inter1 = max(y1, y3)
	x_inter2 = min(x2, x4)
	y_inter2 = min(y2, y4)
	width_inter = abs(x_inter2 - x_inter1)
	height_inter = abs(y_inter2 - y_inter1)
	area_inter = width_inter * height_inter
	width_box1 = abs(x2 - x1)
	height_box1 = abs(y2 - y1)
	width_box2 = abs(x4 - x3)
	height_box2 = abs(y4 - y3)
	area_box1 = width_box1 * height_box1
	area_box2 = width_box2 * height_box2
	area_union = area_box1 + area_box2 - area_inter
	iou = area_inter / area_union
	return iou