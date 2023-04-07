from django.shortcuts import render
from django.http import HttpResponse
from .models import Carslot
from django.shortcuts import redirect
import os
from django.template.loader import render_to_string
import json
from django.urls import reverse
from django.conf import settings
from carslots.models import Carslot
import urllib.parse

# Create your views here.
result = None
cnn_result = None

# get carslot_id for A and B, get yolo and cnn results (ready to be sent to template)
def get_context():
    global result
    global cnn_result
    carslots = Carslot.objects.all()
    slot_idsA = []
    slot_idsB = []
    for carslot in carslots:
        if carslot.slotId[0] == 'a' or carslot.slotId[0] == 'A':
            slot_idsA.append(carslot.slotId)
        else:
            slot_idsB.append(carslot.slotId)
    # result = create_carslot_map()
    json_yolo_result, json_cnn_result = json.dumps(result), json.dumps(cnn_result)
    slot_idsA, slot_idsB = json.dumps(slot_idsA), json.dumps(slot_idsB)
    context = {
        'slot_idsA': slot_idsA,
        'slot_idsB': slot_idsB,
        'cnn_result': json_cnn_result,
        'yolo_result': json_yolo_result
    }

    print("In get context")
    print(result)
    print(cnn_result)
    print()

    return context

def create_carslot_map():
    # Get all the carslots in database
    carslots = Carslot.objects.all()
    result = {}
    # map them to false
    for carslot in carslots:
        result[carslot.slotId] = False
    return result

def process_data(request):
    global result
    global cnn_result
    if request.method == "POST":
        # print("POST DATA RECEIVED")
        data = {
            "object_name": [],
            "x1": [],
            "y1": [],
            "x2": [],
            "y2": [],
            "CNN_result": None,
        }
        for key in request.POST.keys():
            data[key] = request.POST.getlist(key)
        
        # processed cnn and yolo results
        result = compare_bounding_box(data)
        cnn_result_encoded = data["CNN_result"][0]
        cnn_result = json.loads(cnn_result_encoded)

        print("In process_data POST")
        print(cnn_result)
        print(result)
        print()

        return HttpResponse()
    
    if request.method == "GET":
        context = get_context()
        print("CONTEXT IN GET IN PROCESS_DATA")
        print(context)
        print()
        return render(request, "carslots.html", context)

def ajax_function(request):
    global result
    global cnn_result
    if request.method == "GET":
        json_yolo_result = json.dumps(result)
        json_cnn_result = json.dumps(cnn_result)

        context = {
            'yolo_result': json_yolo_result,
            'cnn_result': json_cnn_result,
        }

        print("In AJAX")
        print(context)
        
        return HttpResponse(json.dumps(context), content_type="application/json")

def overwrite_obj(request):
    '''
    1.  Delete all objects in the database and create new ones
    2.  Clear coordinates in /yolov5/CNN/boundingBoxes/spot.txt 
        and write new one
    '''

    # Read the coordinate stored in /drawRect/DrawRectangle/box_coordinates/ with the name
    # image_name.txt
    image_name = str(request.GET.get('image_name')).split('.')[0] + '.txt'
    f = open(os.getcwd() + '/drawRect/DrawRectangle/box_coordinates/' + image_name, 'r')

    # Put the coordinates and slot_ids into list
    carslot_ids = []
    box_coor = []
    for i, line in enumerate(f):
        line = line.rstrip()
        if i % 2 == 0: carslot_ids.append(line)
        else: box_coor.append(line)
    f.close()
    # print(os.getcwd())
    # print("carslot_ids: ", carslot_ids)
    # print("box_coor", box_coor)

    # overwrite files in cnn
    f=open(os.getcwd() + '/yolov5/CNN/boundingBoxes/spot.txt','w')
    for i in range(len(carslot_ids)):
        f.write(carslot_ids[i]+'\n')
        f.write(box_coor[i]+'\n')
    f.close()

    # delete all objects in carslot model, and create
    Carslot.objects.all().delete()

    box_coor = [box.split(',') for box in box_coor]
    box_coor = [[int(coor) for coor in slot] for slot in box_coor]

    print("box_coor", box_coor)

    # create objects
    for i in range(len(carslot_ids)):
        carslot = Carslot(slotId = carslot_ids[i])
        carslot.x1 = box_coor[i][0]
        carslot.y1 = box_coor[i][1]
        carslot.x2 = box_coor[i][2]
        carslot.y2 = box_coor[i][3]
        carslot.save()
    
    return redirect(reverse("process_data"))


def compare_bounding_box(data):
    result = create_carslot_map()
    
    for i in range(len(data["object_name"])):
        if data["object_name"][i] != "car" and data["object_name"][i] != "truck":
            continue
        for carslot in Carslot.objects.all():
            car_coor = [int(data['x1'][i]), int(data['y1'][i]), int(data['x2'][i]), int(data['y2'][i])]
            carslots_coor = [int(carslot.x1), int(carslot.y1), int(carslot.x2), int(carslot.y2)]
            iou = IOU(car_coor, carslots_coor)

            # If there is a car, set to true
            if iou > 0.3 and iou <= 1:
                result[carslot.slotId] = True
    return result

def IOU(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    # No intersection cases
    if (x2 < x3) or (x4 < x1) or (y2 < y3) or (y4 < y1):
        return 0
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

    # area_min = area_box1 if area_box1 < area_box2 else area_box2
    # print("in iou")
    # print(area_box1)
    # print(area_box2)

    iou = area_inter / area_union
    # iou = area_inter/area_min
    # print(iou)
    return iou
