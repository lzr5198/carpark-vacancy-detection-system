from django.shortcuts import render
from django.http import HttpResponse
from .models import Carslot

# Create your views here.

data = {
    "object_name": [],
    "x1": [],
    "y1": [],
    "x2": [],
    "y2": []
}
def process_data(request):
    # field_name = 'slotId'
    # obj = Carslot.objects.first()
    # field_object = Carslot._meta.get_field(field_name)
    # field_value = field_object.value_from_object(obj)


    # should get coordinates and compare to generate the results and pass it to a global var
    # if request.method == "POST":

    #     global object_name
    #     object_name = request.POST["object_name"]
    #     global x1
    #     x1 = request.POST["x1"]
    #     global y1
    #     y1 = request.POST["y1"]
    #     global x2
    #     x2 = request.POST["x2"]
    #     global y2
    #     y2 = request.POST["y2"]

    #     print(object_name)
    #     print(x1)
    #     print(y1)
    #     print(x2)
    #     print(y2)
    #     data = {
    #         'x1': "none",
    #         'y1': "none",
    #         'x2': "none",
    #         'y2': "none",
    #         "object_name": "none"
    #     }
    #     return render(request, "carslots.html", data)
    
    if request.method == "POST":
        global data
        for key in request.POST.keys():
            data[key] = request.POST.getlist(key)
        print(data)
        return HttpResponse("<h1>allgood</h1>")

    if request.method == "GET":
        helper_function()
        # data = {
        #     'x1': x1,
        #     'y1': y1,
        #     'x2': x2,
        #     'y2': y2,
        #     "object_name": object_name
        # }
        return render(request, "carslots.html", data)
    

def helper_function():
    global data
    result = {}
    for i in range(len(data["object_name"])):
        # if data["object_name"][i] != "person":
        #     continue
        for carslot in Carslot.objects.all():
            if int(data["x1"][i]) > carslot.x1 and int(data["y1"][i]) > carslot.y1 and int(data["x2"][i]) < carslot.x2 and int(data["y2"][i]) < carslot.y2:
                result[carslot.slotId] = True
    print(result)

    # for carslot in Carslot.objects.all():

    #     print(carslot.slotId)
    #     print(carslot.x1)
    #     print(carslot.y1)
    #     print(carslot.x2)
    #     print(carslot.y2)

